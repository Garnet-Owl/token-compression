from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import multiprocessing as mp
from multiprocessing import Pool
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


def compress_text_worker(text: str) -> Optional[dict]:
    """Worker function for parallel compression."""
    if not text.strip():
        return None

    compression_service = CompressionService()
    compressed = compression_service.compress(text)
    stats = compression_service.get_stats()

    return {
        "original": text.strip(),
        "compressed": compressed,
        "compression_ratio": stats["compression_ratio"],
        "token_reduction_pct": stats["token_reduction_pct"],
    }


class PairGenerator:
    """Generates training pairs from raw text samples."""

    def __init__(self, batch_size: int = 100):
        self.compression_service = CompressionService()
        self.batch_size = batch_size

    def generate_pair(self, text: str) -> TrainingPair:
        compressed = self.compression_service.compress(text)
        stats = self.compression_service.get_stats()

        return TrainingPair(
            original=text.strip(),
            compressed=compressed,
            compression_ratio=stats["compression_ratio"],
            token_reduction_pct=stats["token_reduction_pct"],
        )

    def generate_pairs_parallel(self, texts: List[str], output_path: Path, num_workers: int = None):
        if num_workers is None:
            num_workers = mp.cpu_count()

        logger.info(f"Starting parallel compression with {num_workers} workers")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("[\n")

        total_processed = 0
        first_batch = True

        with Pool(processes=num_workers) as pool:
            batch = []
            for idx, result in enumerate(pool.imap(compress_text_worker, texts, chunksize=10)):
                if result is None:
                    continue

                batch.append(result)
                total_processed += 1

                if total_processed % 100 == 0:
                    logger.info(f"Processed {total_processed}/{len(texts)} samples")

                if len(batch) >= self.batch_size:
                    self._append_batch(batch, output_path, first_batch)
                    first_batch = False
                    batch = []

            if batch:
                self._append_batch(batch, output_path, first_batch)

        with open(output_path, "a", encoding="utf-8") as f:
            f.write("\n]")

        logger.info(f"Saved {total_processed} pairs to {output_path}")

    @staticmethod
    def _append_batch(batch: List[dict], output_path: Path, is_first: bool):
        with open(output_path, "a", encoding="utf-8") as f:
            for i, pair in enumerate(batch):
                if not is_first or i > 0:
                    f.write(",\n")
                json.dump(pair, f, ensure_ascii=False)

    @staticmethod
    def load_raw_texts(input_path: Path) -> List[str]:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [line.strip() for line in lines if line.strip()]


if __name__ == "__main__":

    def main():
        generator = PairGenerator(batch_size=100)
        input_path = Path("data/raw_texts.txt")
        output_path = Path("data/training_pairs.json")

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return

        logger.info(f"Loading raw texts from {input_path}...")
        sample_texts = generator.load_raw_texts(input_path)
        logger.info(f"Loaded {len(sample_texts)} samples.")

        num_cpus = mp.cpu_count()
        logger.info(f"System has {num_cpus} CPU cores available")

        logger.info("Generating training pairs with multiprocessing...")
        generator.generate_pairs_parallel(sample_texts, output_path, num_workers=num_cpus)
        logger.info("Generation complete.")

    main()
