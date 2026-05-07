#!/usr/bin/env python3
"""Local LLM MCQ Judge - deterministic, free, no API calls.
Uses Qwen2.5-1.5B-Instruct with temperature=0 for reproducible scoring.
Drop-in replacement for gpt5_mcq_judge.py.
Usage: CUDA_VISIBLE_DEVICES=0 python tools/rule_based_mcq_judge.py --predictions_path outputs/.../predictions.json --num_samples -1
"""
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DEFAULT_CGBENCH = _HERE.parent.parent / "cgbench_setup" / "cgbench.json"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

def build_prompt(prediction, choices):
    cb = "\n".join(f"  {chr(65+i)}. {c}" for i, c in enumerate(choices))
    return (f"You are matching a model's free-form prediction to multiple-choice options.\n"
            f"Pick the single option letter (A, B, C, ...) whose meaning best matches the prediction.\n"
            f"Match on meaning: accept paraphrases, numeral/word substitutions (\"3\"=\"three\"), partial key-entity mentions.\n"
            f"If no option matches well, pick the closest one.\n\nChoices:\n{cb}\n\n"
            f"Model's prediction: {prediction}\n\nBest matching letter:")

def load_question_lookup(path):
    if not path: return {}
    with open(path) as f: data = json.load(f)
    out = {}
    for item in data:
        oid = str(item.get("metadata", {}).get("original_id", ""))
        for turn in item.get("conversations", []):
            if turn.get("from") in ("human", "user"):
                out[oid] = turn.get("value", "").replace("<video>", "").strip(); break
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_path", required=True)
    p.add_argument("--cgbench_path", default=str(DEFAULT_CGBENCH))
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_path", default=None)
    p.add_argument("--question_lookup", default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    with open(args.cgbench_path) as f: cg = {str(x["qid"]): x for x in json.load(f)}
    print(f"Loaded {len(cg)} CGBench MCQ items.")
    with open(args.predictions_path) as f: preds = json.load(f)
    print(f"Loaded {len(preds)} predictions.")
    ql = load_question_lookup(args.question_lookup)
    matchable = [(str(pp.get("metadata",{}).get("original_id","")), pp) for pp in preds if str(pp.get("metadata",{}).get("original_id","")) in cg]
    print(f"  {len(matchable)} matchable.")
    if 0 < args.num_samples < len(matchable):
        import random; random.seed(args.seed); matchable = random.sample(matchable, args.num_samples)

    tasks = []
    for oid, pp in matchable:
        item = cg[oid]
        tasks.append({"id": oid, "choices": item["choices"], "prediction": pp["prediction"],
                       "gold_letter": item["right_answer"].upper(), "gold_answer_text": item.get("answer",""),
                       "question": ql.get(oid, item.get("question","")),
                       "sub_category": pp.get("metadata",{}).get("sub_category", item.get("sub_category","?")),
                       "domain": pp.get("metadata",{}).get("domain", item.get("domain","?"))})

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval(); print("  Model loaded.")

    t0 = time.time(); all_chosen = []
    for bs in range(0, len(tasks), args.batch_size):
        batch = tasks[bs:bs+args.batch_size]
        prompts = [tokenizer.apply_chat_template([{"role":"user","content":build_prompt(t["prediction"],t["choices"])}], tokenize=False, add_generation_prompt=True) for t in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=3, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        for i, t in enumerate(batch):
            resp = tokenizer.decode(outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            chosen = "A"
            for ch in resp.upper():
                if ch.isalpha() and ord(ch)-65 < len(t["choices"]): chosen = ch; break
            all_chosen.append(chosen)
        done = min(bs+args.batch_size, len(tasks))
        if done % 100 < args.batch_size or done == len(tasks):
            print(f"  [{done}/{len(tasks)}] {done/(time.time()-t0):.1f} samples/s ({time.time()-t0:.1f}s)")

    results = []
    for task, chosen in zip(tasks, all_chosen):
        results.append({**task, "chosen_letter": chosen, "confidence": "deterministic",
                        "reason": f"local_llm={args.model}", "is_correct": chosen==task["gold_letter"], "error": None})
    n_total = len(results); n_correct = sum(r["is_correct"] for r in results)
    print(f"\n{'='*70}\n  Local LLM Judge\n{'-'*70}\n  Model: {args.model}\n  Judged: {n_total}/{n_total}")
    if n_total: print(f"  Correct: {n_correct}/{n_total} = {100*n_correct/n_total:.1f}%")
    by_sub = defaultdict(lambda:[0,0])
    for r in results: by_sub[r["sub_category"]][1]+=1; (by_sub[r["sub_category"]].__setitem__(0, by_sub[r["sub_category"]][0]+1) if r["is_correct"] else None)
    if len(by_sub) > 1:
        print(f"\n  Per-sub-category:")
        for s,(cor,tot) in sorted(by_sub.items(), key=lambda kv:-kv[1][0]/max(1,kv[1][1])):
            print(f"    {s:<28} {cor}/{tot} = {100*cor/tot:.1f}%")
    out_path = args.output_path or os.path.join(os.path.dirname(args.predictions_path), "rule_judge.json")
    with open(out_path, "w") as f:
        json.dump({"model": f"local-{args.model}", "predictions_file": args.predictions_path,
                    "num_total": n_total, "num_valid": n_total, "num_correct": n_correct,
                    "accuracy": n_correct/n_total if n_total else None, "errors": 0, "samples": results}, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved -> {out_path}")

if __name__ == "__main__": main()
