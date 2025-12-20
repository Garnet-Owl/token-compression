import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.fine_tuning.dataset import load_training_pairs, PROMPT_TEMPLATE
from app.shared.logger import get_logger

logger = get_logger(__name__)


class DecompressionEvaluator:
    """Evaluates fine-tuned model on decompression tasks."""

    def __init__(self, adapter_path: Path = Path("models/lora_adapter")):
        self.adapter_path = adapter_path
        self.tokenizer = None
        self.model = None

    def load_model(self):
        logger.info(f"Loading adapter from {self.adapter_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            trust_remote_code=True,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        base_model.resize_token_embeddings(len(self.tokenizer))
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model.eval()
        logger.info("Model loaded with fine-tuned adapter")

    def decompress(self, compressed_text: str, max_new_tokens: int = 256) -> str:
        prompt = PROMPT_TEMPLATE.format(compressed=compressed_text)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )

        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_output[len(prompt) :].strip()
        return response

    def evaluate_samples(self, data_path: Path, num_samples: int = 5):
        if self.model is None:
            self.load_model()

        pairs = load_training_pairs(data_path)[:num_samples]
        results = []

        for i, pair in enumerate(pairs):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Sample {i + 1}/{num_samples}")
            logger.info(f"{'=' * 60}")

            compressed = pair["compressed"]
            original = pair["original"]
            predicted = self.decompress(compressed)

            logger.info(f"COMPRESSED: {compressed[:150]}...")
            logger.info(f"EXPECTED:   {original[:150]}...")
            logger.info(f"PREDICTED:  {predicted[:150]}...")

            exact_match = predicted.strip() == original.strip()
            results.append(
                {
                    "compressed": compressed,
                    "original": original,
                    "predicted": predicted,
                    "exact_match": exact_match,
                }
            )

            logger.info(f"Match: {'✓ YES' if exact_match else '✗ NO'}")

        accuracy = sum(r["exact_match"] for r in results) / len(results)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"FINAL ACCURACY: {accuracy * 100:.1f}% ({sum(r['exact_match'] for r in results)}/{len(results)})")
        return results


if __name__ == "__main__":
    evaluator = DecompressionEvaluator()
    evaluator.evaluate_samples(Path("data/training_pairs.json"), num_samples=5)
