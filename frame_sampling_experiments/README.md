# frame_sampling_experiments/

Frame-sampling baselines and ablations for long-form video QA. Each
sub-pipeline is a Hydra-based runner that takes a dataset (LVBench-v2 or
PUPIL), samples frames according to some strategy, runs a Qwen-VL backbone,
and writes per-sample result JSONLs.

## Layout

```
frame_sampling_experiments/
├── temporal_cot_gdm/         # TCoT (Temporal-CoT) baseline + scene-detect ablations
├── tcot_eduvideobench/       # TCoT pipeline run on PUPIL
├── edu_cot/                  # EduCoT (scene-aware frame selection) — main pipeline
├── edu_cot_2nd_uniform_k768/ # Re-run with k=768 uniform context (ablation)
├── edu_cot_best_sd_k768/     # Best scene-detect config at k=768
└── edu_cot_eduvideobench/    # EduCoT pipeline run on PUPIL
```

The first four (`temporal_cot_gdm`, `edu_cot*`) target **LVBench-v2**; the
two `*_eduvideobench` variants target **PUPIL** with open-ended QA + LLM
judge.

## Pipeline structure

Every sub-pipeline follows the same template:

```
<pipeline>/
├── main.py            # Hydra entry point (single-shard runner)
├── pipeline.py        # Per-sample driver: load → segment → sample → answer
├── config.yaml        # Hyperparameters (model, dataset, frame budget)
├── stages/            # segmentation.py, video_loading.py, ...
├── models/            # Backbone wrappers (qwen3_vl, qwen25_vl, gpt_azure, ...)
├── utils/             # Shared helpers (metrics, IO, dataset loaders)
├── run_parallel.sh    # Multi-GPU sharded launcher (where present)
└── results/           # Per-sample *_results.jsonl outputs
```

## Running

### Single-GPU smoke test

```bash
cd frame_sampling_experiments/edu_cot_eduvideobench
python main.py num_samples=5 cuda_visible_devices=0
```

### Multi-GPU (data-parallel sharding)

```bash
cd frame_sampling_experiments/edu_cot_eduvideobench
bash run_parallel.sh 8        # 8 shards, one per GPU; merges on completion
```

### Fine-tuned LoRA adapter

```bash
ADAPTER_DIR=/path/to/checkpoint-200 \
ADAPTER_TAG=T04_gradfix_ckpt200 \
bash run_parallel.sh 8
```

### Hydra config overrides

Any value in [config.yaml](edu_cot_eduvideobench/config.yaml) can be
overridden on the command line:

```bash
python main.py model.qwen_model_id=Qwen/Qwen2.5-VL-7B-Instruct \
                num_segments=24 frames_per_segment=64 \
                context_budget_frames=1024
```

Important knobs (TCoT/EduCoT shorthand):

| Hydra key | Symbol | Meaning |
|---|---|---|
| `num_segments` | `l` | Number of video segments to split into |
| `frames_per_segment` | `s` | Frames sampled per segment for the *selection* call |
| `context_budget_frames` | `k` | Total answering-context budget in frames |
| `uniform_context_frames` | `u` | Uniform-sampled frames added alongside model-selected ones |

## VLMEvalKit benchmark adapter

[temporal_cot_gdm/vlmevalkit_bench/](temporal_cot_gdm/vlmevalkit_bench/)
contains a thin wrapper that registers our LVBench-v2 setup as a
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit) dataset, so any
VLMEvalKit-supported model can be evaluated using our metadata. Setup:

```bash
cd frame_sampling_experiments/temporal_cot_gdm/vlmevalkit_bench
bash 01_install.sh                 # clones VLMEvalKit and installs deps
bash run_qwen3vl.sh                # runs Qwen3-VL on LVBench-v2
```

## Setup

```bash
# Same env as the main contrastive experiments works fine
pip install -r ../contrastive_experiments/Qwen-VL-Series-Finetune/requirements.txt
pip install hydra-core scenedetect    # extra deps used here
```

Edit absolute dataset paths in each `config.yaml` (e.g.
`lvbench_v2_video_dir`) before running.

## Notes

- Result JSONLs in each pipeline's `results/` folder were stripped from
  the submission for size; the [results/](edu_cot/results/),
  [results/lvbench_v2/](edu_cot/results/lvbench_v2/), etc. directory
  trees remain.
- The four `edu_cot*` folders are intentionally near-duplicates: each
  represents a different ablation (different `k` / scene-detect
  hyperparameters) that we ran in parallel. The shared logic lives in
  `pipeline.py` + `stages/` of each.
- Azure-OpenAI endpoint values in the configs are placeholders
  (`https://<AZURE_OPENAI_ENDPOINT>`) — set yours before any GPT-judged
  run.
