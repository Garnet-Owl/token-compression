import argparse
from pathlib import Path
from app.fine_tuning.config import FineTuningConfig
from app.fine_tuning.dataset import create_dataset
from app.fine_tuning.trainer import DecompressionTrainer
from app.shared.logger import get_logger

logger = get_logger(__name__)


class FineTuningService:
    """Orchestrates the fine-tuning pipeline."""

    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.trainer = DecompressionTrainer(config)

    def run(self):
        logger.info("Starting fine-tuning pipeline")
        train_ds, eval_ds = create_dataset(
            self.config.data_path,
            test_mode=self.config.test_mode,
            test_samples=self.config.test_samples,
        )
        self.trainer.train(train_ds, eval_ds)
        logger.info("Fine-tuning pipeline complete")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for decompression")
    parser.add_argument("--test-mode", action="store_true", help="Run with subset")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    config = FineTuningConfig(
        test_mode=args.test_mode,
        data_path=Path("data/training_pairs.json"),
    )
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size

    service = FineTuningService(config)
    service.run()


if __name__ == "__main__":
    main()
