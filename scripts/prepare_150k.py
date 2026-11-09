#!/usr/bin/env python3
"""Prepare exactly 150k Canarim instruction/answer pairs + 3k held out.

This does NOT train a language model. Run on GitHub Actions with internet.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

SOURCE = "dominguesm/Canarim-Instruct-PTBR-Dataset"
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b")
URL = re.compile(r"https?://\S+", re.I)
REPEAT = re.compile(r"(.{5,40}?)\1{4,}", re.S)

def clean(s):
    return " ".join(unicodedata.normalize("NFC", str(s or "")).replace("\x00", "").split())

def normalize(row):
    prompt = clean(row.get("instruction") or row.get("question"))
    context = clean(row.get("input") or row.get("context"))
    answer = clean(row.get("output") or row.get("answer") or row.get("response"))
    if context:
        prompt += "\nContexto: " + context
    if not (4 <= len(prompt.encode()) <= 1000 and 8 <= len(answer.encode()) <= 5000):
        return None
    if EMAIL.search(prompt + " " + answer) or URL.search(prompt + " " + answer):
        return None
    if "\ufffd" in (prompt + answer) or REPEAT.search(answer):
        return None
    if sum(c.isalpha() for c in answer) < 5:
        return None
    return {"question": prompt, "answer": answer}

def prepare(rows, output, wanted, seed):
    output.mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in wanted}
    files = {name: (output / (name + ".jsonl.partial")).open("w", encoding="utf-8") for name in wanted}
    seen = set()
    examined = rejected = 0
    try:
        for row in rows:
            examined += 1
            item = normalize(row)
            if item is None:
                rejected += 1
                continue
            key = hashlib.sha256(item["question"].casefold().encode()).digest()
            if key in seen:
                rejected += 1
                continue
            seen.add(key)
            split = next((s for s in ("test", "validation", "train") if counts[s] < wanted[s]), None)
            if split is None:
                break
            item["id"] = hashlib.sha256((item["question"] + "\0" + item["answer"]).encode()).hexdigest()[:20]
            files[split].write(json.dumps(item, ensure_ascii=False) + "\n")
            counts[split] += 1
            count = sum(counts.values())
            if count % 15000 == 0:
                print("Collected", count, "examined", examined, "rejected", rejected, flush=True)
            if count == sum(wanted.values()):
                break
    finally:
        for f in files.values():
            f.close()
    complete = counts == wanted
    checksums = {}
    if complete:
        for name in wanted:
            source = output / (name + ".jsonl.partial")
            final = output / (name + ".jsonl")
            source.rename(final)
            checksums[final.name] = hashlib.sha256(final.read_bytes()).hexdigest()
    manifest = {
        "dataset": SOURCE,
        "license": "CC BY-NC 4.0 (current HF metadata; noncommercial)",
        "status": "complete" if complete else "incomplete",
        "requested": wanted, "counts": counts, "examined": examined,
        "rejected": rejected, "seed": seed, "sha256": checksums,
        "note": "Pairs of instructions/answers, not necessarily multi-turn conversations. No training occurred."
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if not complete:
        raise RuntimeError("Insufficient clean unique pairs: " + str(counts))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="corpus_150k_pt")
    p.add_argument("--train", type=int, default=150000)
    p.add_argument("--validation", type=int, default=1500)
    p.add_argument("--test", type=int, default=1500)
    p.add_argument("--seed", type=int, default=20261109)
    p.add_argument("--fixture", default="", help="Local JSONL for smoke testing only")
    a = p.parse_args()
    if min(a.train, a.validation, a.test) < 1:
        p.error("Splits must have at least one item")
    if a.fixture:
        with open(a.fixture, encoding="utf-8") as f:
            rows = (json.loads(line) for line in f)
            prepare(rows, Path(a.out), {"train": a.train, "validation": a.validation, "test": a.test}, a.seed)
    else:
        from datasets import load_dataset
        rows = load_dataset(SOURCE, split="train", streaming=True)
        rows = rows.shuffle(seed=a.seed, buffer_size=20000)
        prepare(rows, Path(a.out), {"train": a.train, "validation": a.validation, "test": a.test}, a.seed)

if __name__ == "__main__":
    main()
