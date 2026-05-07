#!/usr/bin/env python3
"""Strip the Transcript:\"\"\"...\"\"\" block from every SFT prompt.

Input:  sof_sft_warmstart.json         (every prompt has transcript)
Output: sof_sft_warmstart.no_transcript.json
"""
import json, re, sys, os

SRC = "/workspace/Pupil/contrastive_experiments/sof_dpo/data/sof_sft_warmstart.json"
DST = "/workspace/Pupil/contrastive_experiments/sof_dpo/data/sof_sft_warmstart.no_transcript.json"

# Match: Transcript:\n"""\n<anything incl newlines>\n"""\n\n
# Use DOTALL and non-greedy. Also tolerate stray whitespace.
TRANSCRIPT_RE = re.compile(
    r'Transcript:\s*\n\s*"""\s*\n.*?\n\s*"""\s*\n+',
    re.DOTALL,
)

def strip_prompt(p: str) -> str:
    new, n = TRANSCRIPT_RE.subn("", p, count=1)
    return new, n

def main():
    data = json.load(open(SRC))
    n_total = len(data)
    n_stripped = 0
    n_unchanged = 0
    leak_words = ("transcript", "speaker says above", "as stated above", "above transcript")
    leaks = []

    for rec in data:
        for turn in rec.get("conversations", []):
            if turn.get("from") != "human":
                continue
            v = turn["value"]
            new, n = strip_prompt(v)
            if n:
                n_stripped += 1
                turn["value"] = new
            else:
                n_unchanged += 1
            low = turn["value"].lower()
            if any(w in low for w in leak_words):
                leaks.append((rec["id"], turn["value"][:300]))

    json.dump(data, open(DST, "w"), ensure_ascii=False, indent=2)
    print(f"records: {n_total}")
    print(f"prompts stripped: {n_stripped}")
    print(f"prompts unchanged (no transcript found): {n_unchanged}")
    print(f"prompts still mentioning transcript/speaker-above: {len(leaks)}")
    for i, (rid, snip) in enumerate(leaks[:10]):
        print(f"  [{i}] {rid}: {snip!r}")
    print(f"wrote: {DST}")

    # sanity: average prompt length before/after
    src = json.load(open(SRC))
    def avg_len(d):
        L = []
        for r in d:
            for t in r.get("conversations", []):
                if t.get("from") == "human":
                    L.append(len(t["value"]))
        return sum(L)/len(L), max(L), min(L)
    a0 = avg_len(src); a1 = avg_len(data)
    print(f"prompt char len  before: avg={a0[0]:.0f} max={a0[1]} min={a0[2]}")
    print(f"prompt char len   after: avg={a1[0]:.0f} max={a1[1]} min={a1[2]}")

    # show one example
    print("\n--- EXAMPLE AFTER STRIP (record 0) ---")
    print(data[0]["conversations"][0]["value"][:500])

if __name__ == "__main__":
    main()
