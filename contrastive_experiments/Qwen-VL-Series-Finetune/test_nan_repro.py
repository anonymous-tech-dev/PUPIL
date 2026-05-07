#!/usr/bin/env python3
"""Reproduce the NaN crash from the FPS-based vanilla SFT run."""
import torch
import json
import os
import sys
import gc

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from src.dataset.sft_dataset import SupervisedDataset, DataCollatorForSupervisedDataset
from src.params import DataArguments

torch.manual_seed(42)

# ── Dataset ──
data_args = DataArguments()
data_args.data_path = '/workspace/Pupil/contrastive_experiments/final_sft_data/train.json'
data_args.fps = 1
data_args.nframes = None
data_args.video_max_pixels = 66560
data_args.video_min_pixels = 16640

processor = AutoProcessor.from_pretrained('Qwen/Qwen3-VL-8B-Instruct')
ds = SupervisedDataset(
    data_path=data_args.data_path, processor=processor,
    data_args=data_args, model_id='Qwen/Qwen3-VL-8B-Instruct',
    max_seq_length=65536,
)

# ── Model + LoRA ──
model = Qwen3VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen3-VL-8B-Instruct', dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = False
model.enable_input_require_grads()

for p in model.parameters():
    p.requires_grad = False
for p in model.visual.merger.parameters():
    p.requires_grad = True

lora_config = LoraConfig(
    r=128, lora_alpha=128, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.to('cuda:0')
model.train()

# Use gradient checkpointing like training
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

# ── Optimizer (matching training) ──
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-5, weight_decay=0.01,
)

# ── Test suspect samples ──
# Long video samples from the crash region (step 337, rank 0)
suspect_indices = [18495, 4776, 6961, 13782, 11724, 6967, 7729, 1603]

print(f"{'step':>4} {'idx':>6} {'tokens':>6} {'labels':>6} {'loss':>10} {'grad_norm':>10} {'nan_grad':>8}")
print("-" * 65)

for step_i, idx in enumerate(suspect_indices):
    sample = ds[idx]
    n_tok = len(sample['input_ids'])
    n_lab = (sample['labels'] != -100).sum().item()
    
    batch = collator([sample])
    batch = {k: v.to('cuda:0') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    if 'pixel_values_videos' in batch:
        batch['pixel_values_videos'] = batch['pixel_values_videos'].to(dtype=torch.bfloat16)

    out = model(**batch)
    loss = out.loss
    loss_val = loss.item()
    
    optimizer.zero_grad()
    loss.backward()
    
    # Grad norm
    total_norm = 0.0
    nan_grads = 0
    for p in model.parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                nan_grads += 1
            total_norm += p.grad.data.float().norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    # Clip grads
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    print(f"{step_i:4d} {idx:6d} {n_tok:6d} {n_lab:6d} {loss_val:10.4f} {total_norm:10.4f} {nan_grads:8d}")
    
    if nan_grads > 0:
        print(f"  *** NaN gradients detected at step {step_i}! ***")
        # Check if weights are now NaN
        for name, p in model.named_parameters():
            if p.requires_grad and torch.isnan(p).any():
                print(f"  NaN weights: {name}")
                break
        break
    
    del batch, out, loss
    gc.collect()
    torch.cuda.empty_cache()

print("\nDone!")
