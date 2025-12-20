from typing import Optional
from app.llm_integration.model import CompressedLLM
from app.token_compression.service import CompressionService
from app.token_compression.config import CompressionConfig
from app.shared.logger import get_logger

logger = get_logger(__name__)


class LLMService:
    """Service for interacting with the compressed LLM."""

    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        self.llm = CompressedLLM(config=self.config)
        self.compressor_service = CompressionService(config=self.config)

    def generate_from_compressed(self, text: str) -> str:
        compressed = self.compressor_service.compress(text)
        return self.llm.generate(compressed)

    def analyze_tokenization(self, text: str) -> dict:
        compressed = self.compressor_service.compress(text)
        token_ids = self.llm.get_token_ids(compressed)
        decoded = self.llm.tokenizer.decode(token_ids)

        return {
            "original": text,
            "compressed": compressed,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "decoded_check": decoded,
        }


if __name__ == "__main__":

    def main():
        logger.info("Initializing LLM integration...")
        service = LLMService()

        test_text = "The quick brown fox jumps over the lazy dog."
        logger.info(f"Analyzing: {test_text}")

        analysis = service.analyze_tokenization(test_text)
        logger.info(f"Compressed: {analysis['compressed']}")
        logger.info(f"Token Count: {analysis['token_count']}")
        logger.info(f"Token IDs: {analysis['token_ids']}")

        logger.info("Attempting generation...")
        try:
            output = service.generate_from_compressed(test_text)
            logger.info(f"Output: {output}")
        except Exception as e:
            logger.error(f"Generation failed: {e}")

    main()
