from typing import List
from dataclasses import dataclass
from pathlib import Path
import json
from app.token_compression.service import CompressionService
from app.shared.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingPair:
    """A single training example with original and compressed text."""

    original: str
    compressed: str
    compression_ratio: float
    token_reduction_pct: float


class PairGenerator:
    """Generates training pairs from raw text samples."""

    def __init__(self):
        self.compression_service = CompressionService()

    def generate_pair(self, text: str) -> TrainingPair:
        compressed = self.compression_service.compress(text)
        stats = self.compression_service.get_stats()

        return TrainingPair(
            original=text.strip(),
            compressed=compressed,
            compression_ratio=stats["compression_ratio"],
            token_reduction_pct=stats["token_reduction_pct"],
        )

    def generate_pairs(self, texts: List[str]) -> List[TrainingPair]:
        pairs = []
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            pair = self.generate_pair(text)
            pairs.append(pair)
            logger.info(f"Pair {i + 1}: ratio={pair.compression_ratio}x, token_reduction={pair.token_reduction_pct}%")
        return pairs

    def save_pairs(self, pairs: List[TrainingPair], output_path: Path):
        data = [
            {
                "original": p.original,
                "compressed": p.compressed,
                "compression_ratio": p.compression_ratio,
                "token_reduction_pct": p.token_reduction_pct,
            }
            for p in pairs
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(pairs)} pairs to {output_path}")

    def load_raw_texts(self, input_path: Path) -> List[str]:
        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()
        texts = content.split("---")
        return [t.strip() for t in texts if t.strip()]


if __name__ == "__main__":
    from app.training_data_generation.prompts import GENERATION_PROMPTS

    def main():
        generator = PairGenerator()

        sample_texts = [
            """The database connection pool manages all database connections.
            When a service needs a database connection, it requests one from
            the database connection pool. The database connection pool returns
            an available database connection. After the service is done with
            the database connection, it returns the database connection to
            the database connection pool for reuse.""",
            """The quick brown fox jumps over the lazy dog. The quick brown fox
            jumps over the lazy dog again. And again, the quick brown fox
            jumps over the lazy dog. The lazy dog never moves because the
            lazy dog is lazy.""",
        ]

        logger.info(f"Generating {len(sample_texts)} training pairs...")
        pairs = generator.generate_pairs(sample_texts)

        for i, pair in enumerate(pairs):
            logger.info(f"\n--- Pair {i + 1} ---")
            logger.info(f"Original: {pair.original[:80]}...")
            logger.info(f"Compressed: {pair.compressed[:80]}...")
            logger.info(f"Ratio: {pair.compression_ratio}x")

        output_path = Path("data/training_pairs.json")
        generator.save_pairs(pairs, output_path)

        logger.info("\n=== Prompts for generating more data ===")
        for i, prompt in enumerate(GENERATION_PROMPTS[:3]):
            logger.info(f"\nPrompt {i + 1}: {prompt[:100]}...")

    main()
