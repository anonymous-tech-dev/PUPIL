# mllm_evaluation/

Open-ended QA evaluation harness for the PUPIL benchmark. Runs any model
from a plug-in zoo on the final 1k benchmark, then scores each prediction
with an LLM-as-judge (GPT-5 family via Azure OpenAI) and aggregates into a
leaderboard.

## Layout

```
mllm_evaluation/
├── models/                 # Plug-in model wrappers (one .py per model)
├── envs/                   # Per-model pip requirements + sysinfo dumps
├── results/                # Per-model prediction + judge JSONs
├── ablation_results/       # Blind / transcript-only / re-prompt ablations
├── prompt_tuned_judge/     # Judge-prompt tuning experiments
│
├── config.py               # Dataset paths, model registry, defaults
├── evaluate.py             # Sequential prediction → judge runner
├── evaluate_parallel.py    # Parallel LLM-judge dispatcher (Azure)
├── merge_shards.py         # Merge sharded prediction files
├── aggregate_results.py    # Per-source / per-axis aggregation
├── generate_leaderboard.py # Build a Markdown leaderboard from judged results
├── print_leaderboard.py    # Pretty-print to stdout
└── convert_final1k_to_jsonl.py
```

## Model zoo

Each file in [models/](models/) is a thin adapter that exposes a single
`predict(question, video_path) -> str` interface. Currently included:

| Family | Files |
|---|---|
| Qwen2.5-VL | [qwen_25_vl.py](models/qwen_25_vl.py), [qwen_25_32_vl.py](models/qwen_25_32_vl.py), [qwen_25_vl_transcript.py](models/qwen_25_vl_transcript.py) |
| Qwen3-VL  | [qwen_3_vl.py](models/qwen_3_vl.py), [qwen_32_vl.py](models/qwen_32_vl.py), [qwen3_vl_finetuned.py](models/qwen3_vl_finetuned.py), [qwen3_vl_blind.py](models/qwen3_vl_blind.py), [qwen3_vl_matched.py](models/qwen3_vl_matched.py), [qwen3_vl_transcript.py](models/qwen3_vl_transcript.py) |
| InternVL  | [intern_3_vl.py](models/intern_3_vl.py), [intern_3_38_vl.py](models/intern_3_38_vl.py), [intern_3_78_vl.py](models/intern_3_78_vl.py), [intern_35_vl.py](models/intern_35_vl.py), [intern_35_vl_transcript.py](models/intern_35_vl_transcript.py) |
| GPT (Azure) | [gpt.py](models/gpt.py), [gpt54.py](models/gpt54.py), [gpt54_blind.py](models/gpt54_blind.py), [gpt54_transcript.py](models/gpt54_transcript.py) |
| Claude    | [claude_opus_46.py](models/claude_opus_46.py), [claude_sonnet_46.py](models/claude_sonnet_46.py) |
| Other     | [aria.py](models/aria.py), [llava_video.py](models/llava_video.py), [oryx_15_32b.py](models/oryx_15_32b.py), [tarsier2.py](models/tarsier2.py), [videollama_3.py](models/videollama_3.py), [videosalmonn_2.py](models/videosalmonn_2.py), [videosalmonn_2plus.py](models/videosalmonn_2plus.py) |

Add a new model by dropping a file in [models/](models/) that subclasses
[base.py](models/base.py) (or [transcript_base.py](models/transcript_base.py)
/ [blind_base.py](models/blind_base.py) for the ablation modes) and
registering it in [config.py](config.py).

## Running

### Predictions

```bash
# Run a model on the final-1k benchmark (writes results/<model>/<run>/*_results.json)
python evaluate.py --model qwen3_vl --run final_1k_benchmark
```

### Parallel LLM judge

After predictions are produced, score them with GPT-5 via Azure:

```bash
# Judge a single run; output goes to results_v2/<model>/<run>/ by default
python evaluate_parallel.py \
    --results-dir results/qwen3_vl/final_1k_benchmark \
    --max-workers 20

# Dry-run (count pending entries without calling Azure)
python evaluate_parallel.py --results-dir results/qwen3_vl/final_1k_benchmark --dry-run

# Overwrite existing verdicts
python evaluate_parallel.py --results-dir results/qwen3_vl/final_1k_benchmark --overwrite
```

The judge dispatcher is hot-resumable — it skips entries that already have
a `judge_verdict` set.

### Leaderboard

```bash
python aggregate_results.py            # per-source / per-axis aggregation
python generate_leaderboard.py         # build leaderboard.md
python print_leaderboard.py            # pretty-print to stdout
```

### Sharded runs

If predictions were generated in shards (e.g. `*_shard0of4_results.jsonl`),
merge them first:

```bash
python merge_shards.py --in-dir results/<model>/<run>/
```

## Setup

Each model family has its own pinned pip requirements in [envs/](envs/):

- [qwen_vl_reqs.txt](envs/qwen_vl_reqs.txt)
- [intern_reqs.txt](envs/intern_reqs.txt)
- [videollama_reqs.txt](envs/videollama_reqs.txt)
- [videosalmonn_reqs.txt](envs/videosalmonn_reqs.txt)

Use a separate virtualenv per family — the requirements are mutually
incompatible. The Azure judge needs:

```bash
pip install azure-identity openai
az login           # the judge uses AzureCliCredential
export AZURE_OPENAI_ENDPOINT=https://<your-azure-openai-endpoint>
```

## Notes

- Result JSONs were stripped from [results/](results/) and
  [ablation_results/](ablation_results/) for size; the per-model directory
  skeletons remain so that pointers in the launcher scripts still resolve.
- The judge endpoint is a placeholder
  (`https://<AZURE_OPENAI_ENDPOINT>`) in [config.py](config.py),
  [evaluate.py](evaluate.py), and [evaluate_parallel.py](evaluate_parallel.py)
  — override at runtime via `--endpoint` or by editing
  `DEFAULT_ENDPOINT`.
- The "blind" / "transcript" model variants implement two important
  ablations: **blind** disables visual input (text-only baseline),
  **transcript** appends the speech transcript to the prompt. Results
  for both live under [ablation_results/](ablation_results/).
