# PUPIL

Code release for **PUPIL**, a benchmark and method suite for long-form
educational video question answering. The codebase covers the full pipeline:
dataset curation, frame-sampling baselines, contrastive (SFT / DPO) training
of video-language models, and an open-ended evaluation harness with an
LLM-as-judge.

> **Anonymous submission.** All identifying information has been removed.
> Configuration values that point to internal infrastructure (e.g. Azure
> OpenAI endpoints) are placeholders such as `https://<AZURE_OPENAI_ENDPOINT>`
> and must be overridden before running.

## Repository layout

```
PUPIL/
├── dataset_curation/            # Video collection → transcripts → QA generation
├── frame_sampling_experiments/  # Frame-sampling baselines (TCoT / EduCoT / scene-detect)
├── contrastive_experiments/     # SFT, DPO, SoF-DPO training of Qwen-VL backbones
├── mllm_evaluation/             # Open-ended QA evaluation + LLM-as-judge
└── figures/                     # Paper figure-generation scripts
```

Each top-level directory has its own README with detailed instructions.

| Folder | Purpose | README |
|---|---|---|
| [dataset_curation/](dataset_curation/) | Build the PUPIL benchmark from raw long-form lecture videos | [README](dataset_curation/README.md) |
| [frame_sampling_experiments/](frame_sampling_experiments/) | Evaluate frame-sampling strategies (uniform, scene-detect, EduCoT, TCoT) on LVBench / PUPIL | [README](frame_sampling_experiments/README.md) |
| [contrastive_experiments/](contrastive_experiments/) | Train Qwen-VL with vanilla SFT, DPO, and the SoF-DPO contrastive objective | [README](contrastive_experiments/README.md) |
| [mllm_evaluation/](mllm_evaluation/) | Run any model from a plug-in zoo on PUPIL, then score with an LLM judge | [README](mllm_evaluation/README.md) |

## Setup

The codebase is split across multiple sub-projects, each with its own
environment (different model families have incompatible dependencies). Pick the
sub-project you want to run and follow its README. Common requirements:

- Python ≥ 3.10
- CUDA-capable GPUs (most experiments assume 4×–8× H100/A100/B200)
- An Azure OpenAI deployment for the LLM-as-judge (set
  `AZURE_OPENAI_ENDPOINT`)

Reference environment specs for individual model families live in
[mllm_evaluation/envs/](mllm_evaluation/envs/) (`qwen_vl_reqs.txt`,
`intern_reqs.txt`, `videollama_reqs.txt`, `videosalmonn_reqs.txt`).

## Reproducing the paper

A typical end-to-end pass through the codebase:

1. **Curate the benchmark** — see [dataset_curation/README.md](dataset_curation/README.md)
   to download videos, generate transcripts, and produce the QA queries.
2. **Frame-sampling baselines** — see [frame_sampling_experiments/README.md](frame_sampling_experiments/README.md)
   to run TCoT / EduCoT pipelines on LVBench-v2 and PUPIL.
3. **Train the contrastive model** — see [contrastive_experiments/README.md](contrastive_experiments/README.md)
   for SFT warm-start, DPO, and SoF-DPO.
4. **Evaluate** — see [mllm_evaluation/README.md](mllm_evaluation/README.md) to
   run any baseline or fine-tuned model on PUPIL and produce judged
   leaderboards.

## Notes for reviewers

- Some scripts hard-code historical absolute paths (e.g.
  `/home/Pupil/...`); these reflect the layout on the original training
  cluster and should be replaced with paths under your local
  checkout.
- The third-party MMCTAgent toolkit used in
  [dataset_curation/generation/MMCTAgent/](dataset_curation/generation/MMCTAgent/)
  is a public dependency and is included verbatim for convenience.
- Large data and result artifacts (`*.json`, `*.jsonl` outputs, model
  checkpoints, training logs, video files, subtitles) have been stripped
  from this submission to stay under the size limit. Only source code,
  shell launchers, configs, and `requirements*.txt` files are shipped.
