import os
# Force decord-only video backend to prevent torchvision full-video RAM blowups.
try:
    import decord_only_guard  # noqa: F401
except ImportError:
    pass
import torch
from peft import LoraConfig, get_peft_model
import ast
from transformers import (
    AutoProcessor,
    AutoConfig,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration, 
    HfArgumentParser, 
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from src.trainer.sof_dpo_trainer import QwenDPOTrainer
from src.dataset.sof_dpo_dataset import make_dpo_data_module
from src.params import DataArguments, ModelArguments, DPOArguments
from src.train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
from monkey_patch_forward import (
    replace_qwen2_5_with_mixed_modality_forward, 
    replace_qwen_2_with_mixed_modality_forward,
    replace_qwen3_with_mixed_modality_forward,
    replace_qwen3_vl_moe_with_mixed_modality_forward
)
from monkey_patch_vision import replace_qwen2_5_vision

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
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
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

    if hasattr(model.visual, "deepstack_merger_list"):
        deepstack_merger_list_params = model.visual.deepstack_merger_list.parameters()
        set_requires_grad(deepstack_merger_list_params, not training_args.freeze_merger)

def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def unfreeze_topk_layers(model, k_llm: int = 0, k_vis: int = 0):
    if k_llm and hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        for layer in model.language_model.layers[-k_llm:]:
            for p in layer.parameters():
                p.requires_grad = True

    if k_vis and hasattr(model, "visual") and hasattr(model.visual, "blocks"):
        for blk in model.visual.blocks[-k_vis:]:
            for p in blk.parameters():
                p.requires_grad = True

def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, DPOArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if data_args.nframes is not None and data_args.fps is not None:
        raise ValueError("You cannot set both `nframes` and `fps` at the same time. Please set only one of them.")

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual", "lm_head"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    ref_model = None

    config = AutoConfig.from_pretrained(model_args.model_id)

    if config.model_type == "qwen3_vl_moe":
        replace_qwen3_vl_moe_with_mixed_modality_forward()
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
            **bnb_model_from_pretrained_args
        )
        if not training_args.lora_enable:
            ref_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_args.model_id,
                dtype=compute_dtype,
                attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
                **bnb_model_from_pretrained_args
            )

    elif config.model_type == "qwen3_vl":
        replace_qwen3_with_mixed_modality_forward()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
            **bnb_model_from_pretrained_args
        )
        if not training_args.lora_enable:
            ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_id,
                dtype=compute_dtype,
                attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
                **bnb_model_from_pretrained_args
            )

    elif config.model_type == "qwen2_5_vl":
        replace_qwen2_5_with_mixed_modality_forward()
        replace_qwen2_5_vision()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
            **bnb_model_from_pretrained_args
        )
        if not training_args.lora_enable:
            ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_id,
                dtype=compute_dtype,
                attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
                **bnb_model_from_pretrained_args
            )
        
    else:
        replace_qwen_2_with_mixed_modality_forward()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
            **bnb_model_from_pretrained_args
        )
        if not training_args.lora_enable:
            ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_args.model_id,
                dtype=compute_dtype,
                attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
                **bnb_model_from_pretrained_args
            )

    model.config.use_cache = False

    # ── SoF warm-start: merge SFT LoRA into base BEFORE adding the DPO LoRA ──
    # Without this, DPO trains a fresh LoRA on the raw base model and the
    # KL-reference (ref_model, or the LoRA-disabled policy) is *also* the
    # raw base — the SFT signal is silently discarded.
    if getattr(model_args, "sft_adapter_path", None):
        from peft import PeftModel
        import os as _os
        import torch as _torch
        sft_path = model_args.sft_adapter_path

        # ── CRITICAL: also load the trained full-rank weights for modules
        # excluded from LoRA (default: everything under `visual.*`, i.e. the
        # merger and deepstack mergers).  Without this, the SFT-trained
        # merger (~160M params) is silently dropped — see the eval-loader
        # patch in mllm_evaluation/models/qwen3_vl_finetuned.py for
        # background.  Same loader contract: keys may carry the
        # "base_model.model." prefix from PEFT-wrapped capture.
        def _apply_non_lora(m, sft_path: str, label: str) -> None:
            non_lora_path = _os.path.join(sft_path, "non_lora_state_dict.bin")
            if not _os.path.isfile(non_lora_path):
                rank0_print(f"   ℹ️  no non_lora_state_dict.bin in {sft_path} "
                            f"({label} merger uses base weights — only safe if "
                            f"freeze_merger=True at SFT time)")
                return
            sd = _torch.load(non_lora_path, map_location="cpu", weights_only=False)
            current_keys = {n for n, _ in m.named_parameters()}
            ren = {}
            for k, v in sd.items():
                if k in current_keys:
                    ren[k] = v
                elif k.replace("base_model.model.", "") in current_keys:
                    ren[k.replace("base_model.model.", "")] = v
                else:
                    ren[k] = v  # let load_state_dict report it as unexpected
            missing, unexpected = m.load_state_dict(ren, strict=False)
            rank0_print(f"   🔧 {label}: loaded {len(ren)} non-LoRA tensors "
                        f"(unexpected={len(unexpected)})")
            if unexpected:
                rank0_print(f"      first 3 unexpected: {list(unexpected)[:3]}")

        rank0_print(f"🔌 Merging SFT adapter from: {sft_path}")
        model = PeftModel.from_pretrained(model, sft_path, is_trainable=False)
        _apply_non_lora(model, sft_path, label="policy")
        model = model.merge_and_unload()
        # peft leaves `peft_config` on the base model after merge_and_unload;
        # if we don't drop it, get_peft_model() below warns about "Already
        # found a peft_config attribute" (harmless, but noisy).
        for _m in (model, getattr(model, "base_model", None)):
            if _m is not None and hasattr(_m, "peft_config"):
                try: delattr(_m, "peft_config")
                except Exception: pass
        if ref_model is not None:
            rank0_print(f"🔌 Merging SFT adapter into ref_model from: {sft_path}")
            ref_model = PeftModel.from_pretrained(ref_model, sft_path, is_trainable=False)
            _apply_non_lora(ref_model, sft_path, label="ref")
            ref_model = ref_model.merge_and_unload()
            for _m in (ref_model, getattr(ref_model, "base_model", None)):
                if _m is not None and hasattr(_m, "peft_config"):
                    try: delattr(_m, "peft_config")
                    except Exception: pass
        model.config.use_cache = False

    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    unfreeze_topk_layers(
        model_to_configure,
        k_llm=getattr(training_args, "unfreeze_topk_llm", 0),
        k_vis=getattr(training_args, "unfreeze_topk_vision", 0),
    )

    if training_args.gradient_checkpointing:
        # Honor an explicit CLI override; otherwise fall back to the
        # vision_lora-conditioned default. DPO + LoRA + ZeRO-2 requires
        # use_reentrant=False to avoid double grad-reduction on shared
        # LoRA params during the chosen+rejected concatenated backward.
        if not getattr(training_args, "gradient_checkpointing_kwargs", None):
            if training_args.vision_lora:
                training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
            else:
                training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
        model.enable_input_require_grads()

    if training_args.bits in [4,8]:
        model.config.dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

        # Peft maodel makes vision tower and merger freezed again.
        # Configuring fuction could be called here, but sometimes it does not work properly.
        # So I just made it this way.
        # Need to be fixed in the future.

        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True

        if not training_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length

    if ref_model is not None:
        ref_model.eval()
        ref_model.config.use_cache = False

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    dataset_module = make_dpo_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args)
    
    training_args.padding_value = processor.tokenizer.pad_token_id

    trainer = QwenDPOTrainer(
        model=model,
        ref_model = ref_model,
        train_dataset=dataset_module["train_dataset"],
        eval_dataset = dataset_module["eval_dataset"],
        data_collator= dataset_module["data_collator"],
        processing_class=processor,
        args=training_args,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()