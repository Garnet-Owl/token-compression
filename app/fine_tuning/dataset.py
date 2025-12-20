import json
from pathlib import Path
from typing import List, Dict, Tuple
from datasets import Dataset
from app.shared.logger import get_logger

logger = get_logger(__name__)

PROMPT_TEMPLATE = """Decompress the following text by expanding all reference tokens:

Compressed: {compressed}

Original:"""


def load_training_pairs(data_path: Path) -> List[Dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_for_training(pairs: List[Dict]) -> List[Dict]:
    formatted = []
    for pair in pairs:
        text = f"{PROMPT_TEMPLATE.format(compressed=pair['compressed'])} {pair['original']}"
        formatted.append({"text": text})
    return formatted


def create_dataset(data_path: Path, test_mode: bool = False, test_samples: int = 5) -> Tuple[Dataset, Dataset]:
    pairs = load_training_pairs(data_path)
    logger.info(f"Loaded {len(pairs)} training pairs")

    if test_mode:
        pairs = pairs[:test_samples]
        logger.info(f"Test mode: using {len(pairs)} samples")

    formatted = format_for_training(pairs)
    full_dataset = Dataset.from_list(formatted)

    split = full_dataset.train_test_split(test_size=0.1, seed=42)
    logger.info(f"Train: {len(split['train'])}, Val: {len(split['test'])}")
    return split["train"], split["test"]
