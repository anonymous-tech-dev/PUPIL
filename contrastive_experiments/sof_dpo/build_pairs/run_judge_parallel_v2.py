#!/usr/bin/env python3
"""
run_judge_parallel_v2.py — Run the GPT-5 judge over the v2 judge_prompts.jsonl.

Differs from v1 (run_judge_parallel.py) in ONE thing: the v2 judge prompts
use the EXACT JSON-reply template from
mllm_evaluation/evaluate_parallel.py (verdict bool + reason string).  This
script parses that JSON and remaps the bool to the YES / NO vocabulary
that apply_judge_to_dpo.py expects, so the rest of the v1 pipeline keeps
working unchanged.

Mapping
-------
    {"verdict": true,  ...}  ->  "YES"   (rejected ≈ chosen → drop in DPO)
    {"verdict": false, ...}  ->  "NO"    (rejected ≠ chosen → keep in DPO)
    parse-failure / API-error -> "ERROR" (treated as keep)

Hot-resumable: skip any query_id already in --out.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

_credential = AzureCliCredential()
_token_provider = get_bearer_token_provider(_credential, "api://azure/.default")
_client = AzureOpenAI(
    azure_endpoint="https://<AZURE_OPENAI_ENDPOINT>",
    azure_ad_token_provider=_token_provider,
    api_version="2024-12-01-preview",
)


def call_one(model: str, prompt: str, max_retries: int = 4) -> tuple[str, str]:
    last_err = ""
    for i in range(max_retries):
        try:
            r = _client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_completion_tokens=1024,
                response_format={"type": "json_object"},
            )
            raw = (r.choices[0].message.content or "").strip()
            try:
                data = json.loads(raw)
                v = data.get("verdict", None)
                if isinstance(v, str):
                    v = v.strip().lower() == "true"
                if v is True:
                    return "YES", raw
                if v is False:
                    return "NO", raw
                # Some models return verdict as int 0/1.
                if isinstance(v, (int, float)):
                    return ("YES" if v else "NO"), raw
                return "ERROR", raw[:300]
            except json.JSONDecodeError:
                # Last-resort: look for a stray "true"/"false" in the reply.
                low = raw.lower()
                if re.search(r'"verdict"\s*:\s*true', low):
                    return "YES", raw
                if re.search(r'"verdict"\s*:\s*false', low):
                    return "NO", raw
                return "ERROR", raw[:300]
        except Exception as e:
            last_err = repr(e)
            time.sleep(2 ** i)
    return "ERROR", last_err[:300]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-5.1_2025-11-13")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    print(f"[start] model={args.model} workers={args.workers} pending={len(items)} "
          f"(skip={len(done)}) -> {out_path}", flush=True)
    if not items:
        print("[done] nothing to do.")
        return 0

    counts: dict[str, int] = {}
    t0 = time.time()
    fout = out_path.open("a")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut2row = {ex.submit(call_one, args.model, r["prompt"]): r for r in items}
            for i, fut in enumerate(as_completed(fut2row), 1):
                r = fut2row[fut]
                verdict, raw = fut.result()
                counts[verdict] = counts.get(verdict, 0) + 1
                fout.write(json.dumps({
                    "query_id": r["query_id"],
                    "axis": r.get("axis"),
                    "verdict": verdict,
                    "raw": raw,
                }) + "\n")
                fout.flush()
                if i % 50 == 0 or i == len(items):
                    rate = i / max(1e-6, time.time() - t0)
                    print(f"[{i:5d}/{len(items)}]  {rate:5.1f} req/s  counts={counts}",
                          flush=True)
    finally:
        fout.close()

    print(f"[done] counts={counts}  out={out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
