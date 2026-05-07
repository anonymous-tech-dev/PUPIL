"""
sof_dpo_frames_diagnostic_judge.py

Pairwise judge over the (neg_low, neg_high) outputs from
sof_dpo_frames_diagnostic_generate.py.  Uses GPT-5.4 via Azure Azure in
text-only mode (no frames sent).

For each pair we ask:
    "Both A and B are wrong answers to QUESTION (the reference is REF).
     Which one's wrongness is more clearly attributable to the answerer
     not having access to the spoken transcript / audio of the video,
     as opposed to having access to fewer / blurrier video frames?"

We randomise the A/B mapping per row to control for position bias.
Decision rule (reported at the end):
    If `high` wins ≤55% of decided pairs  -> 24 frames is fine.
    If `high` wins 55-70%                 -> bump to 32 or 48.
    If `high` wins  >70%                  -> 24 contaminates, use 64.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI


JUDGE_SYSTEM = (
    "You are a careful diagnostic grader for a video-QA failure analysis. "
    "Your only job is to decide which of two wrong answers shows a failure "
    "mode that is more clearly caused by missing AUDIO / spoken-transcript "
    "evidence (as opposed to missing fine-grained VISUAL evidence). "
    "Answer with exactly one token: A, B, or TIE."
)

JUDGE_USER_TEMPLATE = """QUESTION:
{question}

REFERENCE ANSWER (correct):
{ref}

CANDIDATE A:
{a}

CANDIDATE B:
{b}

Both A and B are wrong. Decide which one's wrongness is MORE CLEARLY \
attributable to the answerer lacking access to the SPOKEN TRANSCRIPT of the \
video, as opposed to lacking visual detail.

Hints:
- An answer like "I cannot tell from the video" or "the video does not show" \
that fails on a verbally-explained concept is an audio-failure.
- An answer that hallucinates plausible visual details but misses what the \
speaker said is an audio-failure.
- An answer that confuses fine spatial / pixel-level detail is a visual-failure.

Answer with exactly one token: A, B, or TIE."""


def build_client():
    cred = AzureCliCredential()
    tok = get_bearer_token_provider(cred, "api://azure/.default")
    azure_ai = AzureOpenAI(
        azure_endpoint="https://<AZURE_OPENAI_ENDPOINT>",
        azure_ad_token_provider=tok,
        api_version="2024-12-01-preview",
    )
    azure_openai = AzureOpenAI(
        azure_endpoint="https://<AZURE_OPENAI_ENDPOINT>",
        azure_ad_token_provider=tok,
        api_version="2024-12-01-preview",
    )
    return azure_openai, azure_ai


def call_judge(active, fallback, deployment, messages, max_retries=4,
               base_delay=4.0):
    last_err = None
    for client in (active, fallback):
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    max_completion_tokens=8,
                    reasoning_effort="minimal",
                )
                return resp.choices[0].message.content.strip(), client
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if "rate" in msg or "429" in msg or "throttl" in msg:
                    time.sleep(base_delay * (attempt + 1))
                else:
                    if attempt == max_retries - 1:
                        break
                    time.sleep(base_delay)
    raise RuntimeError(f"judge failed: {last_err!r}")


_LBL = re.compile(r"\b(A|B|TIE)\b", re.I)


def parse_choice(s: str) -> str | None:
    if not s:
        return None
    m = _LBL.search(s)
    return m.group(1).upper() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--deployment", default="gpt-5.4_2026-03-05")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.in_jsonl) if l.strip()]
    print(f"loaded {len(rows)} pairs from {args.in_jsonl}")

    azure_openai, azure_ai = build_client()
    active = azure_ai

    rng = random.Random(args.seed)
    out = open(args.out_jsonl, "w", buffering=1)
    counts = {"low": 0, "high": 0, "tie": 0, "err": 0}
    for k, r in enumerate(rows):
        # randomise A/B mapping
        if rng.random() < 0.5:
            a_label, b_label = "low", "high"
            a_text, b_text = r["neg_low"], r["neg_high"]
        else:
            a_label, b_label = "high", "low"
            a_text, b_text = r["neg_high"], r["neg_low"]

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                question=r["question"], ref=r["ground_truth"],
                a=a_text, b=b_text,
            )},
        ]
        try:
            raw, active = call_judge(active, azure_ai, args.deployment, messages)
        except Exception as e:
            print(f"  [err] {r['query_id']}: {e!r}", flush=True)
            counts["err"] += 1
            continue
        choice = parse_choice(raw)
        if choice is None:
            counts["err"] += 1
            decision = None
        elif choice == "TIE":
            counts["tie"] += 1
            decision = "tie"
        else:
            chosen = a_label if choice == "A" else b_label
            counts[chosen] += 1
            decision = f"more_audio_failure={chosen}"

        out.write(json.dumps({
            "query_id": r["query_id"],
            "a_is": a_label,
            "judge_raw": raw,
            "decision": decision,
        }) + "\n")
        if k % 5 == 0:
            print(f"  {k+1}/{len(rows)}  raw={raw!r}  -> {decision}",
                  flush=True)

    out.close()

    decided = counts["low"] + counts["high"]
    pct_high = counts["high"] / max(1, decided) * 100
    print(f"\n=== Frames-diagnostic verdict ===")
    print(f"  pairs total           : {len(rows)}")
    print(f"  judge errors          : {counts['err']}")
    print(f"  ties                  : {counts['tie']}")
    print(f"  high (64f) chosen     : {counts['high']}")
    print(f"  low  (24f) chosen     : {counts['low']}")
    print(f"  high-win % of decided : {pct_high:.1f}")
    print()
    if pct_high <= 55:
        print("  VERDICT: 24 frames is FINE.  Ship N_FRAMES_FULL=24.")
    elif pct_high <= 70:
        print("  VERDICT: BORDERLINE.  Bump to N_FRAMES_FULL=32 or 48 as a hedge.")
    else:
        print("  VERDICT: 24 CONTAMINATES the audio axis with frame-starvation.")
        print("           Use N_FRAMES_FULL=64 for the audio axis.")


if __name__ == "__main__":
    main()
