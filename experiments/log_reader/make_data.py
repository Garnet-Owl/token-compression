"""Build synthetic log-question data for the template-compression experiment.

Usage: python experiments/log_reader/make_data.py <output folder> [train_n] [test_n]

Each example has a 30-line log (four kinds of line), a question, and the answer.
`ctx` is the original log and `ctx_comp` is the template-compressed log. Every
example is checked to round-trip exactly before it is written.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.structured_compression.compressor import StructuredCompressor  # noqa: E402

LINES = 30


def make_log(rng: random.Random):
    """Return (log text, structured facts about the lines)."""
    seconds = rng.randint(0, 3000)
    rows, facts = [], []
    for _ in range(LINES):
        seconds += rng.randint(1, 9)
        stamp = f"2026-11-12 10:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
        kind = rng.choices(["request", "slow", "error", "cache"], weights=[50, 15, 15, 20])[0]
        if kind == "request":
            rid, status, ms = rng.randint(10**7, 10**8 - 1), rng.choice([200, 200, 201, 404, 500]), rng.randint(5, 900)
            line = (f"{stamp} INFO request_id={rid} method={rng.choice(['GET', 'POST'])} "
                    f"path=/api/v1/{rng.choice(['users', 'orders', 'items'])}/{rng.randint(1, 500)} status={status} duration_ms={ms}")
            facts.append(dict(kind=kind, rid=rid, status=status, ms=ms))
        elif kind == "slow":
            line = f"{stamp} WARN slow_query table={rng.choice(['users', 'orders'])} duration_ms={rng.randint(500, 3000)} rows={rng.randint(1, 9000)}"
            facts.append(dict(kind=kind))
        elif kind == "error":
            line = f"{stamp} ERROR upstream_timeout service={rng.choice(['auth', 'billing', 'search'])} after_ms={rng.randint(1000, 30000)} request_id={rng.randint(10**7, 10**8 - 1)}"
            facts.append(dict(kind=kind))
        else:
            line = f"{stamp} DEBUG cache_hit key=user:{rng.randint(1, 500)} ttl={rng.randint(10, 600)}"
            facts.append(dict(kind=kind))
        rows.append(line)
    return "\n".join(rows), facts


def make_question(rng: random.Random, facts):
    requests = [f for f in facts if f["kind"] == "request"]
    options = ["count"] + (["lookup", "slowest"] if requests else [])
    qtype = rng.choice(options)
    if qtype == "lookup":
        t = rng.choice(requests)
        return qtype, f"What were the status code and duration_ms of request_id={t['rid']}?", f"{t['status']} {t['ms']}"
    if qtype == "slowest":
        return qtype, "What is the highest duration_ms among the INFO request lines?", str(max(r["ms"] for r in requests))
    return qtype, "How many ERROR lines are in the log?", str(sum(f["kind"] == "error" for f in facts))


def build(seed: int, n: int):
    rng, compressor, out = random.Random(seed), StructuredCompressor(), []
    for _ in range(n):
        text, facts = make_log(rng)
        packed = compressor.compress(text)
        assert compressor.decompress(packed) == text, "round trip failed"
        qtype, question, answer = make_question(rng, facts)
        out.append(dict(ctx=text, ctx_comp=packed, q=question, a=answer, qtype=qtype))
    return out


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    folder.mkdir(parents=True, exist_ok=True)
    train_n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    test_n = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    for name, seed, n in (("train", 11, train_n), ("test", 7, test_n)):
        data = build(seed, n)
        with open(folder / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for x in data:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(name, len(data))
