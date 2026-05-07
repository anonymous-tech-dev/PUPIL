"""
==============================================================================
Contrastive SFT Training Entry Point
==============================================================================
Wires together:
  - Model loading (Qwen3-VL / Qwen2.5-VL / Qwen2-VL)
  - LoRA configuration
  - Freeze/unfreeze settings (knobs from CLI)
  - ContrastiveSFTDataset + collator
  - ContrastiveSFTTrainer with contrastive regularization
  - Projection head (for vector contrastive mode)

This file mirrors train_sft.py but adds ContrastiveArguments parsing and
the contrastive training components. The vanilla SFT pipeline in
train_sft.py is NOT modified.

Stage: Training Script Orchestration
==============================================================================
"""

import os
import ast
import pathlib

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoProcessor,
    BitsAndBytesConfig,
    HfArgumentParser,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

from src.params import ContrastiveArguments, DataArguments, ModelArguments, TrainingArguments
from src.dataset.contrastive_sft_dataset import make_contrastive_data_module
from src.trainer.contrastive_sft_trainer import ContrastiveSFTTrainer
from src.contrastive_utils import ContrastiveProjectionHead
from src.train.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
)
from src.train.monkey_patch_forward import (
    replace_qwen3_with_mixed_modality_forward,
    replace_qwen2_5_with_mixed_modality_forward,
    replace_qwen_2_with_mixed_modality_forward,
    replace_qwen3_vl_moe_with_mixed_modality_forward,
)
from src.train.monkey_patch_vision import replace_qwen2_5_vision


local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank == "0" or local_rank is None:
        print(*args)


def find_target_linear_names(
    model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True
):
    """Find all linear/embedding layer names for LoRA targeting."""
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(
            f"Found {len(lora_module_names)} lora modules: {lora_module_names}"
        )
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    """Stage: Freeze/unfreeze vision tower and merger (knob-controlled)."""
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

    if hasattr(model.visual, "deepstack_merger_list"):
        deepstack_params = model.visual.deepstack_merger_list.parameters()
        set_requires_grad(deepstack_params, not training_args.freeze_merger)


def configure_llm(model, training_args):
    """Stage: Freeze/unfreeze LLM backbone (knob-controlled)."""
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def unfreeze_topk_layers(model, k_llm: int = 0, k_vis: int = 0):
    """Stage: Selectively unfreeze top-k layers (knob-controlled)."""
    if k_llm and hasattr(model, "language_model") and hasattr(
        model.language_model, "layers"
    ):
        for layer in model.language_model.layers[-k_llm:]:
            for p in layer.parameters():
                p.requires_grad = True

    if k_vis and hasattr(model, "visual") and hasattr(model.visual, "blocks"):
        for blk in model.visual.blocks[-k_vis:]:
            for p in blk.parameters():
                p.requires_grad = True


def train():
    global local_rank

    # ══════════════════════════════════════════════════════════════
    # Stage 1: Parse arguments (4 argument groups)
    # ══════════════════════════════════════════════════════════════
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ContrastiveArguments)
    )
    model_args, data_args, training_args, contrastive_args = (
        parser.parse_args_into_dataclasses()
    )

    # ── Validation ──
    if data_args.nframes is not None and data_args.fps is not None:
        raise ValueError(
            "You cannot set both `nframes` and `fps` at the same time."
        )
    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError(
            "If `lora_enable` is True, `freeze_llm` must also be True."
        )
    if not training_args.lora_enable:
        assert not training_args.vision_lora, (
            "training_args.lora_enable is not enabled, but vision_lora is."
        )
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError(
            "If `vision_lora` is True, `freeze_vision_tower` must also be True."
        )
    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(
                training_args.lora_namespan_exclude
            )
        else:
            training_args.lora_namespan_exclude = []
        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # ══════════════════════════════════════════════════════════════
    # Stage 2: Log contrastive configuration
    # ══════════════════════════════════════════════════════════════
    rank0_print("═" * 70)
    rank0_print("  CONTRASTIVE SFT TRAINING")
    rank0_print(f"  Experiment:         {contrastive_args.negative_strategy}")
    rank0_print(f"  Contrastive Mode:   {contrastive_args.contrastive_mode}")
    rank0_print(f"  Lambda (weight):    {contrastive_args.contrastive_weight}")
    rank0_print(f"  Alpha (grounding):  {contrastive_args.alpha_grounding_penalty}")
    rank0_print(f"  Temperature (τ):    {contrastive_args.contrastive_temperature}")
    rank0_print("═" * 70)

    # ══════════════════════════════════════════════════════════════
    # Stage 3: Load model
    # ══════════════════════════════════════════════════════════════
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["visual", "lm_head"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            )
        )

    config = AutoConfig.from_pretrained(model_args.model_id)

    if config.model_type == "qwen3_vl_moe":
        replace_qwen3_vl_moe_with_mixed_modality_forward()
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation=(
                "flash_attention_2"
                if not training_args.disable_flash_attn2
                else "sdpa"
            ),
            **bnb_model_from_pretrained_args,
        )
    elif config.model_type == "qwen3_vl":
        replace_qwen3_with_mixed_modality_forward()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation=(
                "flash_attention_2"
                if not training_args.disable_flash_attn2
                else "sdpa"
            ),
            **bnb_model_from_pretrained_args,
        )
    elif config.model_type == "qwen2_5_vl":
        replace_qwen2_5_with_mixed_modality_forward()
        replace_qwen2_5_vision()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation=(
                "flash_attention_2"
                if not training_args.disable_flash_attn2
                else "sdpa"
            ),
            **bnb_model_from_pretrained_args,
        )
    else:
        replace_qwen_2_with_mixed_modality_forward()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation=(
                "flash_attention_2"
                if not training_args.disable_flash_attn2
                else "sdpa"
            ),
            **bnb_model_from_pretrained_args,
        )

    model.config.use_cache = False
    model_to_configure = model

    # ══════════════════════════════════════════════════════════════
    # Stage 4: Configure freeze/unfreeze (knob-controlled)
    # ══════════════════════════════════════════════════════════════
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(
        model_to_configure, training_args, compute_dtype, training_args.device
    )
    unfreeze_topk_layers(
        model_to_configure,
        k_llm=getattr(training_args, "unfreeze_topk_llm", 0),
        k_vis=getattr(training_args, "unfreeze_topk_vision", 0),
    )

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        model.enable_input_require_grads()

    if training_args.bits in [4, 8]:
        model.config.dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs,
        )

    # ══════════════════════════════════════════════════════════════
    # Stage 5: LoRA setup
    # ══════════════════════════════════════════════════════════════
    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(
                model,
                lora_namespan_exclude=lora_namespan_exclude,
                num_lora_modules=training_args.num_lora_modules,
            ),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True
        if not training_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_token" in name:
                if hasattr(module, "weight"):
                    if (
                        training_args.bf16
                        and module.weight.dtype == torch.float32
                    ):
                        module = module.to(torch.bfloat16)

    # ══════════════════════════════════════════════════════════════
    # Stage 6: Create projection head (vector contrastive mode only)
    # ══════════════════════════════════════════════════════════════
    projection_head = None
    if contrastive_args.contrastive_mode == "vector":
        # Get the hidden size from the model config
        # Qwen3VL uses a nested config: text_config.hidden_size
        if hasattr(model.config, "hidden_size"):
            hidden_size = model.config.hidden_size
        elif hasattr(model.config, "text_config"):
            hidden_size = model.config.text_config.hidden_size
        else:
            raise ValueError(
                "Cannot determine hidden_size from model config. "
                f"Available: {list(model.config.__dict__.keys())}"
            )
        projection_dim = 512  # Standard projection dimension
        projection_head = ContrastiveProjectionHead(
            hidden_size=hidden_size,
            projection_dim=projection_dim,
        )
        # Move to same device/dtype as model
        projection_head = projection_head.to(
            dtype=compute_dtype, device=training_args.device
        )
        # Register as a submodule of the model so DeepSpeed tracks it.
        # DeepSpeed's ZeRO optimizer requires all optimizer parameters to
        # be part of the model that gets wrapped.
        model.contrastive_projection_head = projection_head
        rank0_print(
            f"Created projection head: {hidden_size} → {projection_dim} "
            f"(vector contrastive mode, registered as model submodule)"
        )

    # ══════════════════════════════════════════════════════════════
    # Stage 7: Create dataset + collator
    # ══════════════════════════════════════════════════════════════
    data_module = make_contrastive_data_module(
        model_id=model_args.model_id,
        processor=processor,
        data_args=data_args,
        max_seq_length=training_args.max_seq_length,
        cgbench_train_vids_dir=contrastive_args.cgbench_train_vids_dir,
        max_samples_cgbench=contrastive_args.max_samples_cgbench,
        max_samples_finevideo=contrastive_args.max_samples_finevideo,
        max_samples_edubench=contrastive_args.max_samples_edubench,
        use_reasoning_traces=contrastive_args.use_reasoning_traces,
        max_val_samples=contrastive_args.max_val_samples,
        cgbench_anchors_path=contrastive_args.cgbench_anchors_path,
    )

    # ══════════════════════════════════════════════════════════════
    # Stage 8: Create trainer
    # ══════════════════════════════════════════════════════════════
    trainer = ContrastiveSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        contrastive_args=contrastive_args,
        projection_head=projection_head,
        **data_module,
    )

    # ══════════════════════════════════════════════════════════════
    # Stage 9: Train (with checkpoint resume support)
    # ══════════════════════════════════════════════════════════════
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    # ══════════════════════════════════════════════════════════════
    # Stage 10: Save ONLY adapters (not merged model — disk space)
    # ══════════════════════════════════════════════════════════════
    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )

        if local_rank == 0 or local_rank == -1:
            # Save adapter config + weights (NOT the full merged model)
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(
                training_args.output_dir, state_dict=state_dict
            )
            processor.save_pretrained(training_args.output_dir)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_state_dict.bin"),
            )

            # Save projection head if it exists (for vector mode testing)
            if projection_head is not None:
                torch.save(
                    projection_head.state_dict(),
                    os.path.join(training_args.output_dir, "projection_head.bin"),
                )
                rank0_print("Saved projection head to output_dir")

            rank0_print(
                f"Saved LoRA adapters (NOT merged model) to "
                f"{training_args.output_dir}"
            )
    else:
        safe_save_model_for_hf_trainer(
            trainer, output_dir=training_args.output_dir
        )


if __name__ == "__main__":
    train()
