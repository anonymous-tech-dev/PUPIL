"""DPO entry-point for Qwen-VL.  Mirrors :mod:`src.train.train_sft`.

Launch with DeepSpeed::

    deepspeed --num_gpus 4 src/train/train_dpo.py \\
        --model_id Qwen/Qwen3-VL-8B-Instruct \\
        --data_path /path/to/dpo_train.json \\
        --output_dir outputs/dpo_run \\
        --bf16 True --beta 0.1 ...

For LoRA runs, the policy model gets adapters and the reference is the same
model with adapters disabled (``ref_model=None``).  For full-finetune runs we
load a second frozen copy as the reference.
"""
from __future__ import annotations

import ast
import os
import pathlib

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoProcessor,
    BitsAndBytesConfig,
    HfArgumentParser,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

from src.dataset import make_dpo_data_module
from src.params import DataArguments, DPOArguments, ModelArguments
from src.train.monkey_patch_forward import (
    replace_qwen2_5_with_mixed_modality_forward,
    replace_qwen3_vl_moe_with_mixed_modality_forward,
    replace_qwen3_with_mixed_modality_forward,
    replace_qwen_2_with_mixed_modality_forward,
)
from src.train.monkey_patch_vision import replace_qwen2_5_vision
from src.train.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
)
from src.trainer import QwenDPOTrainer

local_rank = None


def rank0_print(*args):
    if local_rank in (0, "0", None):
        print(*args)


# ─────────────────────────────────────────────────────────────────────────────
# helpers (lifted verbatim from train_sft.py to keep behaviour identical)
# ─────────────────────────────────────────────────────────────────────────────
def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=(),
                             verbose=True):
    linear_cls = torch.nn.Linear
    embedding_cls = torch.nn.Embedding
    names = []
    for name, module in model.named_modules():
        if any(ex in name for ex in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            names.append(name)
    if num_lora_modules > 0:
        names = names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(names)} LoRA modules")
    return names


def set_requires_grad(parameters, requires_grad: bool):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    vt = model.visual
    vt.to(dtype=compute_dtype, device=device)
    set_requires_grad(vt.parameters(), not training_args.freeze_vision_tower)
    set_requires_grad(model.visual.merger.parameters(), not training_args.freeze_merger)
    if hasattr(vt, "deepstack_merger_list"):
        set_requires_grad(vt.deepstack_merger_list.parameters(),
                          not training_args.freeze_merger)


def configure_llm(model, training_args):
    set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    set_requires_grad(model.language_model.parameters(), not training_args.freeze_llm)


def unfreeze_topk_layers(model, k_llm: int = 0, k_vis: int = 0):
    if k_llm and hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        for layer in model.language_model.layers[-k_llm:]:
            for p in layer.parameters():
                p.requires_grad = True
    if k_vis and hasattr(model, "visual") and hasattr(model.visual, "blocks"):
        for blk in model.visual.blocks[-k_vis:]:
            for p in blk.parameters():
                p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# model-loading helper (returns a freshly-loaded backbone)
# ─────────────────────────────────────────────────────────────────────────────
def _load_backbone(model_args, training_args, compute_dtype,
                   bnb_kwargs):
    config = AutoConfig.from_pretrained(model_args.model_id)
    attn = "flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa"
    common = dict(dtype=compute_dtype, attn_implementation=attn, **bnb_kwargs)
    if config.model_type == "qwen3_vl_moe":
        replace_qwen3_vl_moe_with_mixed_modality_forward()
        return Qwen3VLMoeForConditionalGeneration.from_pretrained(model_args.model_id, **common)
    if config.model_type == "qwen3_vl":
        replace_qwen3_with_mixed_modality_forward()
        return Qwen3VLForConditionalGeneration.from_pretrained(model_args.model_id, **common)
    if config.model_type == "qwen2_5_vl":
        replace_qwen2_5_with_mixed_modality_forward()
        replace_qwen2_5_vision()
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_id, **common)
    replace_qwen_2_with_mixed_modality_forward()
    return Qwen2VLForConditionalGeneration.from_pretrained(model_args.model_id, **common)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, DPOArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.nframes is not None and data_args.fps is not None:
        raise ValueError("Set only one of --nframes or --fps.")

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If lora_enable=True you must also set freeze_llm=True.")
    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "vision_lora requires lora_enable=True."
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("vision_lora requires freeze_vision_tower=True.")

    if training_args.lora_namespan_exclude is not None:
        training_args.lora_namespan_exclude = ast.literal_eval(
            training_args.lora_namespan_exclude
        )
    else:
        training_args.lora_namespan_exclude = []
    if not training_args.vision_lora:
        training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16
                     else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_kwargs = {}
    if training_args.bits in (4, 8):
        bnb_kwargs.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["visual", "lm_head"],
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            ),
        ))

    # ── policy model ────────────────────────────────────────────────────
    model = _load_backbone(model_args, training_args, compute_dtype, bnb_kwargs)
    model.config.use_cache = False
    configure_llm(model, training_args)
    configure_vision_tower(model, training_args, compute_dtype, training_args.device)
    unfreeze_topk_layers(
        model,
        k_llm=getattr(training_args, "unfreeze_topk_llm", 0),
        k_vis=getattr(training_args, "unfreeze_topk_vision", 0),
    )

    if training_args.gradient_checkpointing:
        # use_reentrant=False matches what train_sft.py does — required to
        # avoid silent zero-grad bugs on DeepSpeed ZeRO-2.
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        model.enable_input_require_grads()

    if training_args.bits in (4, 8):
        model.config.dtype = compute_dtype
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs,
        )

    # ── reference model ─────────────────────────────────────────────────
    ref_model = None
    if not training_args.lora_enable:
        rank0_print("Full-finetune DPO → loading frozen reference model.")
        ref_model = _load_backbone(model_args, training_args, compute_dtype, bnb_kwargs)
        ref_model.config.use_cache = False
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()

    # ── LoRA ────────────────────────────────────────────────────────────
    if training_args.lora_enable:
        lora_cfg = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(
                model,
                lora_namespan_exclude=training_args.lora_namespan_exclude,
                num_lora_modules=training_args.num_lora_modules,
            ),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
        )
        if training_args.bits == 16:
            model.to(torch.bfloat16 if training_args.bf16 else torch.float16)
        rank0_print("Adding LoRA adapters …")
        model = get_peft_model(model, lora_cfg)

        if not training_args.freeze_vision_tower:
            for n, p in model.named_parameters():
                if "visual" in n:
                    p.requires_grad = True
        if not training_args.freeze_merger:
            for n, p in model.named_parameters():
                if "merger" in n:
                    p.requires_grad = True

    # ── processor + data ────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(model_args.model_id)
    data_module = make_dpo_data_module(
        model_id=model_args.model_id,
        processor=processor,
        data_args=data_args,
        max_seq_length=training_args.max_seq_length,
    )

    # TRL's DPOConfig sets remove_unused_columns=False internally; force it.
    training_args.remove_unused_columns = False

    trainer = QwenDPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=getattr(training_args, "beta", 0.1),
        loss_type=getattr(training_args, "dpo_loss", "sigmoid"),
        label_smoothing=getattr(training_args, "label_smoothing", 0.0) or 0.0,
        is_peft_model=training_args.lora_enable,
        args=training_args,
        processing_class=processor,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"],
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True

    # ── save ────────────────────────────────────────────────────────────
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )
        if local_rank in (0, -1):
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict,
                       os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
