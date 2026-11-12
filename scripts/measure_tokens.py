"""Count real LLM tokens before and after compression.

Run from the repo root:  uv run python -m scripts.measure_tokens
Needs: uv add tiktoken
"""

import json
import random
import time
from pathlib import Path

import tiktoken

from app.token_compression.compressor import Compressor

ENCODINGS = {name: tiktoken.get_encoding(name) for name in ("cl100k_base", "o200k_base")}
ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CHARS = 9000


def synthetic_logs() -> str:
    random.seed(1)
    return "\n".join(
        f"2026-11-12 10:{i % 60:02d}:{random.randint(0, 59):02d} INFO "
        f"request_id={random.randint(10**7, 10**8)} method=GET "
        f"path=/api/v1/users/{random.randint(1, 500)} status=200 "
        f"duration_ms={random.randint(5, 90)} user_agent=Mozilla/5.0"
        for i in range(120)
    )


def synthetic_json() -> str:
    random.seed(1)
    records = [
        {
            "id": i,
            "name": f"item{i}",
            "status": "active",
            "category": random.choice(["books", "toys", "tools"]),
            "price": round(random.random() * 100, 2),
            "in_stock": True,
        }
        for i in range(60)
    ]
    return json.dumps(records, indent=1)


def corpora() -> dict[str, str]:
    spec = (ROOT / "Mathematical Spec.md").read_text(encoding="utf-8")
    code = "\n".join(p.read_text(encoding="utf-8") for p in sorted((ROOT / "app").rglob("*.py")))
    return {
        "prose (this repo's docs)": spec[:SAMPLE_CHARS],
        "Python code (this repo)": code[:SAMPLE_CHARS],
        "server logs (synthetic)": synthetic_logs(),
        "JSON records (synthetic)": synthetic_json(),
    }


def main() -> None:
    print(f"{'corpus':28} {'chars':>6} {'words saved':>12} {'cl100k':>8} {'o200k':>8}")
    for name, text in corpora().items():
        compressor = Compressor()
        start = time.time()
        compressed = compressor.compress(text)
        stats = compressor.get_stats()
        cells = []
        for encoding in ENCODINGS.values():
            before, after = len(encoding.encode(text)), len(encoding.encode(compressed))
            cells.append(f"{(1 - after / before) * 100:+.1f}%")
        print(
            f"{name:28} {len(text):>6} {stats['token_reduction_pct']:>11}% "
            f"{cells[0]:>8} {cells[1]:>8}  ({time.time() - start:.0f}s)"
        )


if __name__ == "__main__":
    main()
