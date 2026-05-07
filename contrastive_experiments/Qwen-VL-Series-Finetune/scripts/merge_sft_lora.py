"""
Merge an SFT LoRA adapter into the base Qwen3-VL weights and save a
self-contained HF-format checkpoint that can be used as MODEL_ID for
downstream DPO training.

Why a dedicated script (vs src/merge_lora_weights.py)?
    - Loads in bf16 (matches SFT/DPO training dtype). The existing utility
      hardcodes fp16, which would silently truncate a bf16-trained adapter.
    - Writes a merge_info.json with provenance so we never lose track of
      which SFT checkpoint a merged base came from.
    - No flash-attn / device_map magic — pure CPU merge, then save.

Usage:
    python scripts/merge_sft_lora.py \\
        --base_model    Qwen/Qwen3-VL-8B-Instruct \\
        --adapter_path  /workspace/.../T-04_.../checkpoint-200 \\
        --output_dir    /workspace/.../outputs/merged/sft-T04-a5-ck200-merged
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)


def _load_base_model(base_model: str, lora_cfg, dtype: torch.dtype):
    common = dict(low_cpu_mem_usage=True, torch_dtype=dtype, config=lora_cfg)
    mt = lora_cfg.model_type
    if mt == "qwen3_vl_moe":
        return Qwen3VLMoeForConditionalGeneration.from_pretrained(base_model, **common)
    if mt == "qwen3_vl":
        return Qwen3VLForConditionalGeneration.from_pretrained(base_model, **common)
    if mt == "qwen2_5_vl":
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model, **common)
    return Qwen2VLForConditionalGeneration.from_pretrained(base_model, **common)


def merge(args):
    adapter_path = Path(args.adapter_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not (adapter_path / "adapter_config.json").exists():
        sys.exit(f"[merge_sft_lora] FATAL: no adapter_config.json at {adapter_path}")
    if not (adapter_path / "adapter_model.safetensors").exists():
        sys.exit(f"[merge_sft_lora] FATAL: no adapter_model.safetensors at {adapter_path}")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            sys.exit(
                f"[merge_sft_lora] FATAL: {output_dir} exists and is non-empty. "
                f"Pass --overwrite to clobber."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"[merge_sft_lora] dtype           : {args.dtype}")
    print(f"[merge_sft_lora] base model      : {args.base_model}")
    print(f"[merge_sft_lora] adapter         : {adapter_path}")
    print(f"[merge_sft_lora] output          : {output_dir}")

    # Use the adapter's saved config (carries any architectural tweaks the SFT run made).
    print("[merge_sft_lora] loading adapter config ...")
    lora_cfg = AutoConfig.from_pretrained(str(adapter_path))
    if hasattr(lora_cfg, "quantization_config"):
        del lora_cfg.quantization_config

    print("[merge_sft_lora] loading base model on CPU ...")
    t0 = time.time()
    model = _load_base_model(args.base_model, lora_cfg, dtype)
    print(f"[merge_sft_lora]   base loaded in {time.time()-t0:.1f}s")

    # SFT trainer saves trainable non-LoRA params (e.g. merger biases) separately.
    non_lora_path = adapter_path / "non_lora_state_dict.bin"
    if non_lora_path.exists():
        print(f"[merge_sft_lora] loading non-LoRA trainables from {non_lora_path.name} ...")
        nlt = torch.load(str(non_lora_path), map_location="cpu", weights_only=False)
        # Strip wrapper prefixes added by PEFT/Trainer.
        nlt = {(k[11:] if k.startswith("base_model.") else k): v for k, v in nlt.items()}
        if any(k.startswith("model.model.") for k in nlt):
            nlt = {(k[6:] if k.startswith("model.") else k): v for k, v in nlt.items()}
        # Cast to target dtype to avoid dtype-mismatch warnings on attach.
        nlt = {k: (v.to(dtype) if torch.is_floating_point(v) else v) for k, v in nlt.items()}
        missing, unexpected = model.load_state_dict(nlt, strict=False)
        print(f"[merge_sft_lora]   loaded {len(nlt)} tensors  "
              f"(missing-from-nlt={len(missing)}  unexpected={len(unexpected)})")
        if unexpected:
            print(f"[merge_sft_lora]   WARNING unexpected keys (first 5): {unexpected[:5]}")
    else:
        print("[merge_sft_lora] no non_lora_state_dict.bin — skipping")

    print("[merge_sft_lora] attaching PEFT LoRA ...")
    model = PeftModel.from_pretrained(model, str(adapter_path), torch_dtype=dtype)

    print("[merge_sft_lora] merge_and_unload ...")
    t0 = time.time()
    model = model.merge_and_unload()
    print(f"[merge_sft_lora]   merged in {time.time()-t0:.1f}s")

    print("[merge_sft_lora] saving merged model ...")
    t0 = time.time()
    model.save_pretrained(str(output_dir), safe_serialization=True)
    print(f"[merge_sft_lora]   model saved in {time.time()-t0:.1f}s")

    print("[merge_sft_lora] saving processor (from base) ...")
    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.save_pretrained(str(output_dir))

    # CRITICAL: processor.save_pretrained() expands video_preprocessor_config.json
    # and preprocessor_config.json with extra default fields (e.g. do_sample_frames=True,
    # fps=2, max_frames=768) that the upstream Qwen3-VL HF release does NOT ship.
    # Those defaults override the dataset's own frame-sampling logic at runtime and
    # produce wrong-shaped video tensors → reshape errors at training time.
    # Fix: copy the upstream processor JSONs verbatim, replacing the bloated saves.
    try:
        from huggingface_hub import snapshot_download
        upstream = Path(snapshot_download(
            args.base_model,
            allow_patterns=["video_preprocessor_config.json", "preprocessor_config.json"],
        ))
        for fname in ("video_preprocessor_config.json", "preprocessor_config.json"):
            src = upstream / fname
            if src.exists():
                # back up the bloated save, then overwrite with upstream
                bloated = output_dir / fname
                if bloated.exists():
                    bloated.rename(output_dir / f"{fname}.saved_by_processor.bak")
                import shutil
                shutil.copy(src, output_dir / fname)
                print(f"[merge_sft_lora]   restored {fname} from upstream")
    except Exception as e:
        print(f"[merge_sft_lora]   WARNING could not restore upstream processor configs: {e}")
        print(f"[merge_sft_lora]   You may hit reshape errors at training time. "
              f"Manually copy {args.base_model}'s {{video_,}}preprocessor_config.json into {output_dir}.")

    # Provenance — read this whenever you wonder what's in this dir.
    info = {
        "kind": "sft_lora_merged",
        "base_model": args.base_model,
        "adapter_path": str(adapter_path),
        "merge_dtype": args.dtype,
        "merged_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "merge_script": str(Path(__file__).resolve()),
        "intended_use": "Use as MODEL_ID for DPO training. Reference policy in DPO will be this merged model with adapters disabled.",
    }
    with open(output_dir / "merge_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"[merge_sft_lora] wrote {output_dir / 'merge_info.json'}")
    print("[merge_sft_lora] DONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True,
                   help="HF id or local path of the original base (e.g. Qwen/Qwen3-VL-8B-Instruct)")
    p.add_argument("--adapter_path", required=True,
                   help="Path to SFT LoRA checkpoint dir (must contain adapter_config.json + adapter_model.safetensors)")
    p.add_argument("--output_dir", required=True,
                   help="Where to write the merged HF-format model")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                   help="Save dtype. Match your training dtype (bf16 for our pipeline).")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow writing into a non-empty --output_dir")
    args = p.parse_args()
    merge(args)
