"""Build the train and test files for the fine-tuning experiment.

Usage: python experiments/fine_tune_reader/make_data.py <output folder>
The test sets match the zero-shot test (logs seed 7, SQuAD validation[:200]); the training sets use a different log seed and the SQuAD train split.
"""
import json, random, sys, urllib.request
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.token_compression.compressor import Compressor

def squad(split, n):
    rows = []
    for off in range(0, n, 100):
        u = f"https://datasets-server.huggingface.co/rows?dataset=rajpurkar/squad&config=plain_text&split={split}&offset={off}&length=100"
        for r in json.load(urllib.request.urlopen(u, timeout=60))["rows"]:
            r = r["row"]; rows.append(dict(task="squad", c=r["context"], q=r["question"], a=r["answers"]["text"]))
    return rows

def logs(seed, n):
    random.seed(seed); out = []
    for _ in range(n):
        rows = [(random.randint(10**7, 10**8), random.choice([200, 200, 200, 404, 500]), random.randint(5, 90), random.randint(1, 500)) for _ in range(40)]
        text = "\n".join(f"2026-11-12 10:{i % 60:02d}:00 INFO request_id={r[0]} method=GET path=/api/v1/users/{r[3]} status={r[1]} duration_ms={r[2]} user_agent=Mozilla/5.0" for i, r in enumerate(rows))
        t = random.choice(rows); out.append(dict(task="logs", c=text, q=f"What were the status and duration_ms of request_id={t[0]}?", a=[str(t[1]), str(t[2])]))
    return out

def comp(x): x["cc"] = Compressor().compress(x["c"]); return x

if __name__ == "__main__":
    # test sets: same as the first Kaggle run (logs seed 7, squad validation[:200]); train sets use different seeds / the train split
    sets = {"test": logs(7, 100) + squad("validation", 200), "train": logs(11, 600) + squad("train", 600)}
    with Pool(4) as p:
        for name, items in sets.items():
            done = p.map(comp, items, chunksize=8)
            with open(rf"{sys.argv[1]}/{name}.jsonl", "w", encoding="utf-8") as f:
                for x in done: f.write(json.dumps(x, ensure_ascii=False) + "\n")
            print(name, len(done), flush=True)
