#!/usr/bin/env python3
"""
run_judge_parallel.py — Send every prompt in judge_prompts.jsonl to GPT-5
(Azure Azure) and write the verdict (YES / PARTIAL / NO) to judge_results.jsonl.

Resumable: any query_id already present in --out is skipped on restart.

Output schema (one JSON per line):
    {"query_id": "...", "axis": "...", "verdict": "YES|PARTIAL|NO|ERROR",
     "raw": "<model reply, trimmed>"}
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

# ----- Azure Azure client (same auth pattern as /tmp/judge_noise/rejudge_*.py) -
_credential = AzureCliCredential()
_token_provider = get_bearer_token_provider(_credential, "api://azure/.default")
_client = AzureOpenAI(
    azure_endpoint="https://<AZURE_OPENAI_ENDPOINT>",
    azure_ad_token_provider=_token_provider,
    api_version="2024-10-21",
)

VALID = {"YES", "PARTIAL", "NO"}
_FIRST_WORD = re.compile(r"[A-Za-z]+")


def parse_verdict(text: str) -> str:
    if not text:
        return "ERROR"
    m = _FIRST_WORD.search(text.strip())
    if not m:
        return "ERROR"
    w = m.group(0).upper()
    return w if w in VALID else "ERROR"


def call_one(model: str, prompt: str, max_retries: int = 4) -> tuple[str, str]:
    last_err = ""
    for i in range(max_retries):
        try:
            r = _client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                # gpt-5.x is a reasoning model: most of the budget is consumed
                # by hidden reasoning tokens before any text is emitted. Give
                # it ~512 so the actual one-word reply lands.
                max_completion_tokens=1024,
            )
            raw = (r.choices[0].message.content or "").strip()
            return parse_verdict(raw), raw
        except Exception as e:
            last_err = repr(e)
            time.sleep(2 ** i)
    return "ERROR", last_err[:300]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="inp",
        default="/workspace/Pupil/contrastive_experiments/sof_dpo/data/judge_prompts.jsonl",
    )
    ap.add_argument(
        "--out",
        default="/workspace/Pupil/contrastive_experiments/sof_dpo/data/judge_results.jsonl",
    )
    ap.add_argument("--model", default="gpt-5.1_2025-11-13")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support.
    done: set[str] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["query_id"])
                except Exception:
                    pass
        print(f"[resume] {len(done)} already done in {out_path}", flush=True)

    items: list[dict] = []
    with in_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r["query_id"] in done:
                continue
            items.append(r)
            if args.limit and len(items) >= args.limit:
                break

    print(
        f"[start] model={args.model} workers={args.workers} pending={len(items)} "
        f"(skip={len(done)}) -> {out_path}",
        flush=True,
    )

    if not items:
        print("[done] nothing to do.")
        return 0

    counts: dict[str, int] = {}
    t0 = time.time()
    # Append-only writes; one line per future as it completes.
    fout = out_path.open("a")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut2row = {
                ex.submit(call_one, args.model, r["prompt"]): r for r in items
            }
            for i, fut in enumerate(as_completed(fut2row), 1):
                r = fut2row[fut]
                verdict, raw = fut.result()
                counts[verdict] = counts.get(verdict, 0) + 1
                fout.write(
                    json.dumps(
                        {
                            "query_id": r["query_id"],
                            "axis": r.get("axis"),
                            "verdict": verdict,
                            "raw": raw,
                        }
                    )
                    + "\n"
                )
                fout.flush()
                if i % 50 == 0 or i == len(items):
                    rate = i / max(1e-6, time.time() - t0)
                    print(
                        f"[{i:5d}/{len(items)}]  {rate:5.1f} req/s  counts={counts}",
                        flush=True,
                    )
    finally:
        fout.close()

    print(f"[done] counts={counts}  out={out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
