import torch
from typing import Optional
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.main_runner.config import Config
from app.token_compression.service import CompressionService
from app.shared.logger import get_logger

logger = get_logger(__name__)


class QAService:
    """Q&A service using the fine-tuned LLM with compressed context."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.compression = CompressionService()
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        logger.info(f"Loading model from: {self.config.adapter_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.config.adapter_path), trust_remote_code=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            attn_implementation="eager",
        )

        base_model.resize_token_embeddings(len(self.tokenizer))
        self.model = PeftModel.from_pretrained(base_model, str(self.config.adapter_path))
        self.model.eval()

        if device == "cpu":
            self.model = self.model.to(device)

        self._loaded = True
        logger.info(f"Model ready on {device}")

    def ask(self, context: str, question: str) -> dict:
        if not self._loaded:
            self.load()

        compressed = self.compression.compress(context)
        stats = self.compression.get_stats()

        logger.info(f"Question: {question}")
        logger.info(f"Original: {len(context)} chars → Compressed: {len(compressed)} chars")
        logger.info(f"Compression: {stats['compression_ratio']}x ({stats['token_reduction_pct']}% tokens saved)")
        logger.info(f"Compressed text: {compressed[:200]}...")

        prompt = f"Context: {compressed}\n\nQuestion: {question}\n\nAnswer:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt) :].strip()

        return {
            "question": question,
            "answer": answer,
            "compression_ratio": stats["compression_ratio"],
            "token_reduction_pct": stats["token_reduction_pct"],
        }
