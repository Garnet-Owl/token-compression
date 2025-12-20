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

    def main():
        generator = PairGenerator()
        input_path = Path("data/raw_texts.txt")
        output_path = Path("data/training_pairs.json")

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return

        logger.info(f"Loading raw texts from {input_path}...")
        sample_texts = generator.load_raw_texts(input_path)
        logger.info(f"Loaded {len(sample_texts)} samples.")

        logger.info("Generating training pairs...")
        pairs = generator.generate_pairs(sample_texts)

        generator.save_pairs(pairs, output_path)
        logger.info("Generation complete.")

    main()
