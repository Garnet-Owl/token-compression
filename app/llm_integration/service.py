from typing import List, Optional
from app.llm_integration.model import CompressedLLM
from app.token_compression.service import CompressionService
from app.token_compression.config import CompressionConfig


class LLMService:
    """
    Service for interacting with the compressed LLM.
    """

    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        self.llm = CompressedLLM(config=self.config)
        self.compressor_service = CompressionService(config=self.config)

    def generate_from_compressed(self, text: str) -> str:
        """Compress text and then generate from it."""
        compressed = self.compressor_service.compress(text)
        return self.llm.generate(compressed)

    def analyze_tokenization(self, text: str) -> dict:
        """Analyze how the text is compressed and tokenized."""
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
        print("Initializing LLM integration...")
        service = LLMService()

        test_text = "The quick brown fox jumps over the lazy dog."
        print(f"\nAnalyzing: {test_text}")

        analysis = service.analyze_tokenization(test_text)
        print(f"Compressed: {analysis['compressed']}")
        print(f"Token Count: {analysis['token_count']}")
        print(f"Token IDs: {analysis['token_ids']}")

        print("\nAttempting generation...")
        try:
            output = service.generate_from_compressed(test_text)
            print(f"Output: {output}")
        except Exception as e:
            print(f"Generation failed: {e}")

    main()
