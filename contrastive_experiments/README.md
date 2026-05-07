# contrastive_experiments/

Training-time experiments for the PUPIL paper: vanilla SFT, DPO, and the
**SoF-DPO** (Source-of-Failure DPO) contrastive objective on Qwen-VL
backbones, plus all data-prep needed to feed those trainers.

## Layout

```
contrastive_experiments/
├── Qwen-VL-Series-Finetune/   # Trainer fork (SFT / DPO / contrastive losses)
│   ├── scripts/               # Launchers for each training mode
│   ├── src/                   # Custom datasets, losses, trainer subclasses
│   └── tools/                 # evaluate_model.py, etc.
├── sof_dpo/                   # SoF-DPO data pipeline + launchers
│   ├── build_pairs/           # Generate negatives, score margins, assemble pairs
│   ├── scripts/               # v2_*.sh stage launchers + run_all orchestrator
│   ├── data*/                 # Pair / preference data (axis-ablated)
│   └── frames_diag/           # Frame-budget diagnostics for the audio axis
│
├── activity_qa_setup/         # ActivityNet-QA data prep (baseline comparison)
├── cgbench_setup/             # CG-Bench data prep
├── finevid_setup/             # FineVideo SFT data prep
├── edu_train_setup/           # PUPIL-Train SFT data prep (downloader + formatter)
│
├── dpo_data/                  # Assembled DPO pair files (CG-Bench + smoke sets)
├── final_sft_data/            # Final SFT mix (train / val / test JSONs)
├── final_sft_data_cgb_only/   # CG-Bench-only SFT split
└── outputs/                   # Per-run training output dirs (checkpoints etc.)
```

## What runs where

The training entry points all live in
[Qwen-VL-Series-Finetune/scripts/](Qwen-VL-Series-Finetune/scripts/):

| Script | Mode | Notes |
|---|---|---|
| [train_vanilla_sft.sh](Qwen-VL-Series-Finetune/scripts/train_vanilla_sft.sh) | SFT (frame-count) | Frozen vision tower, fixed `nframes` |
| [train_vanilla_sft_fps.sh](Qwen-VL-Series-Finetune/scripts/train_vanilla_sft_fps.sh) | SFT (FPS-based) | FPS-driven sampling, dynamic seq-length |
| [train_cl_sft.sh](Qwen-VL-Series-Finetune/scripts/train_cl_sft.sh), [train_cl_sft_fps.sh](Qwen-VL-Series-Finetune/scripts/train_cl_sft_fps.sh) | Contrastive SFT | Adds the contrastive auxiliary loss on top of SFT |
| [train_dpo.sh](Qwen-VL-Series-Finetune/scripts/train_dpo.sh) | DPO | Sigmoid-DPO on assembled preference pairs |
| [merge_sft_lora.sh](Qwen-VL-Series-Finetune/scripts/merge_sft_lora.sh) | LoRA merge | Merges adapters into the base checkpoint |
| [eval_baseline.sh](Qwen-VL-Series-Finetune/scripts/eval_baseline.sh), [test*.sh](Qwen-VL-Series-Finetune/scripts/) | Eval | Quick BLEU/ROUGE/judge eval (see [tools/evaluate_model.py](Qwen-VL-Series-Finetune/tools/evaluate_model.py)) |

All trainer scripts read defaults from environment variables, e.g.:

```bash
NUM_GPUS=8 GLOBAL_BATCH=64 LR=2e-5 EPOCHS=3 \
    bash Qwen-VL-Series-Finetune/scripts/train_vanilla_sft_fps.sh
```

## SoF-DPO (the contrastive method)

[sof_dpo/](sof_dpo/) is a self-contained, multi-stage pipeline that builds
preference pairs by ablating one source-of-failure axis at a time (audio,
visual, priority, time) and using the resulting wrong answers as
**chosen-vs-rejected** pairs.

End-to-end pipeline orchestrated by
[sof_dpo/scripts/v2_run_all.sh](sof_dpo/scripts/v2_run_all.sh):

```bash
bash sof_dpo/scripts/v2_run_all.sh           # all stages 0..6
STAGES=0,1,2 bash sof_dpo/scripts/v2_run_all.sh   # subset
```

| Stage | Script | Description |
|---|---|---|
| 0 | [v2_00_build_negatives.sh](sof_dpo/scripts/v2_00_build_negatives.sh) | Generate negative responses under axis-ablated context (8-GPU sharded; ~5–6 h) |
| 1 | [v2_01_filter.sh](sof_dpo/scripts/v2_01_filter.sh) | ROUGE / keyword / abstain filter |
| 2 | [v2_02_score_margins.sh](sof_dpo/scripts/v2_02_score_margins.sh) | Reference-policy margins (8-GPU; ~2–3 h) |
| 3 | [v2_03_assemble.sh](sof_dpo/scripts/v2_03_assemble.sh) | Assemble pre-judge DPO + SFT files |
| 4 | [v2_04_judge.sh](sof_dpo/scripts/v2_04_judge.sh) | GPT-5 judge (Azure) over the assembled pairs |
| 5 | [v2_05_apply_judge.sh](sof_dpo/scripts/v2_05_apply_judge.sh) | Apply judge → judged DPO + SFT |
| 6 | [v2_06_curriculum.sh](sof_dpo/scripts/v2_06_curriculum.sh) | Easy → hard curriculum re-order (Run-2 dataset) |

Then SFT warm-start + DPO:

```bash
bash sof_dpo/scripts/10_sof_sft_warmstart.sh    # SFT warm-start
bash sof_dpo/scripts/20_sof_dpo.sh              # SoF-DPO from the warm-start
```

## Data-prep helpers

These take third-party raw datasets and convert them into the Qwen-VL
trainer's expected JSON layout:

- `activity_qa_setup/` — ActivityNet-QA → SFT JSON
- `cgbench_setup/` — CG-Bench → SFT + DPO JSONs
- `finevid_setup/` — FineVideo → long-form SFT
- `edu_train_setup/` — PUPIL-Train video downloader + formatter

The post-prep mixes live in [final_sft_data/](final_sft_data/) and
[final_sft_data_cgb_only/](final_sft_data_cgb_only/).

## Setup

```bash
cd Qwen-VL-Series-Finetune
pip install -r requirements.txt   # see also environment.yaml
```

Hardware assumed by defaults: 4×–8× B200 / H100 (192 GB). Most launchers
auto-recompute `gradient_accumulation_steps` from `NUM_GPUS` so the global
batch size is preserved when scaling.

## Notes

- Output checkpoints, training logs, and result JSONs were stripped from
  the [outputs/](outputs/) and [sof_dpo/outputs/](sof_dpo/outputs/) folders
  for size; only the directory skeletons remain.
- Any reference to an Azure judge endpoint in shell scripts uses a
  placeholder (`https://<AZURE_OPENAI_ENDPOINT>`).
- The vendored [Qwen-VL-Series-Finetune/](Qwen-VL-Series-Finetune/) trainer
  is based on a publicly available open-source fine-tuning toolkit; we
  add custom losses (SoF-DPO, contrastive SFT) under
  [src/loss/](Qwen-VL-Series-Finetune/src/loss/) and
  [src/trainer/](Qwen-VL-Series-Finetune/src/trainer/).
