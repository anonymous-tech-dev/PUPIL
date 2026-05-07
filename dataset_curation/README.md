# dataset_curation/

End-to-end pipeline that turns raw long-form lecture videos into the PUPIL
benchmark of audio/visual/priority/time questions, plus the larger SFT
training set.

## Layout

```
dataset_curation/
├── transcription/        # Download videos, generate transcripts (Whisper / Azure STT)
├── generation/           # MMCTAgent-based ingestion + QA generation
│   ├── MMCTAgent/        # Vendored upstream toolkit (public third-party dep)
│   ├── ingestion_script*.py
│   ├── prompts.py
│   └── pipelines/
├── sft_data_curation/    # Build the larger SFT training set from raw queries
└── dataset/              # On-disk layout for videos, transcripts, queries
    ├── videos_db/
    ├── transcripts_db/
    └── queries_db/       # Per-axis queries (sof_audio / sof_visual / sof_priority / sof_time)
```

## Pipeline overview

| Stage | Folder | Entry point | What it does |
|---|---|---|---|
| 1. Source videos | [transcription/](transcription/) | [yt_download.py](transcription/yt_download.py), [gdrive_download.py](transcription/gdrive_download.py) | Pull raw lecture videos |
| 2. Transcribe | [transcription/](transcription/) | [gen_transcript.py](transcription/gen_transcript.py) | Generate `.srt` transcripts |
| 3. Ingest videos into MMCT | [generation/](generation/) | [ingestion_script_v2.py](generation/ingestion_script_v2.py) | Chunk + index videos using the MMCTAgent video pipeline |
| 4. Generate QA queries | [generation/](generation/) | [script.py](generation/script.py), [pipelines/](generation/pipelines/) | Produce per-axis question candidates from chunks |
| 5. Filter / consolidate | [generation/](generation/), [dataset/queries_db/utils/](dataset/queries_db/utils/) | various | Filter, dedupe, and combine per-axis JSON files into the final benchmark |
| 6. Build SFT pool | [sft_data_curation/](sft_data_curation/) | [main.py](sft_data_curation/main.py), [main_parallel.py](sft_data_curation/main_parallel.py) | Curate a large SFT training set out of the same source videos |

## Running

### Setup

1. Install [MMCTAgent](generation/MMCTAgent/) (vendored — see its
   [pyproject.toml](generation/MMCTAgent/pyproject.toml)). MMCTAgent is a
   publicly available third-party toolkit; we use its video ingestion +
   chunking pipeline.
2. Provide an Azure OpenAI deployment (`AZURE_OPENAI_ENDPOINT`) — used for
   chapter generation and QA candidate generation by MMCT.
3. Edit the absolute paths near the top of every `ingestion_script*.py`
   and any `combiner_*.py` in [dataset/queries_db/utils/](dataset/queries_db/utils/)
   so they point at your local checkout (the originals reference
   `/home/Pupil/...`).

### Typical run

```bash
# 1. Download videos + generate transcripts
python transcription/yt_download.py
python transcription/gen_transcript.py

# 2. Ingest into MMCT (chunks + per-chunk embeddings)
python generation/ingestion_script_v2.py        # processes [START_INDEX, STOP_INDEX)

# 3. Generate per-axis queries
python generation/script.py

# 4. Build the SFT pool (parallel across multiple GPT workers)
python sft_data_curation/main_parallel.py
```

Per-axis query files end up under
[dataset/queries_db/final_train/](dataset/queries_db/final_train/) (raw) and
the final consolidated benchmark lives under
[dataset/queries_db/final_1k/](dataset/queries_db/final_1k/).

## Notes

- The actual videos, transcripts, and query JSON/JSONL files are *not*
  included in this submission (size + licensing). Only the curation code
  is shipped. Reviewers can reproduce the dataset by pointing
  `transcription/` at the public NPTEL-style source URLs in the original
  curation scripts.
- The vendored [MMCTAgent/](generation/MMCTAgent/) folder is a public
  third-party toolkit included for convenience only.
