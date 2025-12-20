from typing import Optional, Dict
from app.token_compression.config import CompressionConfig
from app.token_compression.compressor import Compressor
from app.token_compression.decompressor import Decompressor
from app.shared.logger import get_logger

logger = get_logger(__name__)


class CompressionService:
    """Facade for compression and decompression services."""

    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        self.compressor = Compressor(self.config)
        self.decompressor = Decompressor()

    def compress(self, text: str) -> str:
        return self.compressor.compress(text)

    def decompress(self, compressed_text: str) -> str:
        return self.decompressor.decompress(compressed_text)

    def get_stats(self) -> Dict:
        return self.compressor.get_stats()

    @staticmethod
    def verify_roundtrip(original: str, decompressed: str) -> bool:
        orig_norm = " ".join(original.split())
        decomp_norm = " ".join(decompressed.split())
        return orig_norm == decomp_norm


if __name__ == "__main__":

    def main():
        tests = [
            (
                "Highly Repetitive",
                """
            The quick brown fox jumps over the lazy dog. The quick brown fox
            jumps over the lazy dog again. And again, the quick brown fox
            jumps over the lazy dog. The lazy dog never moves because the
            lazy dog is lazy. The quick brown fox keeps jumping over the
            lazy dog because that is what the quick brown fox does.
            """,
            ),
            (
                "Technical Text",
                """
            The database connection pool manages database connections. When a
            service needs a database connection, it requests one from the
            database connection pool. The database connection pool returns an
            available database connection. After the service is done with the
            database connection, it returns the database connection to the
            database connection pool.
            """,
            ),
            (
                "Natural Language",
                """
            I was thinking about what we discussed yesterday. The proposal
            seems reasonable, however I think we should consider the budget
            implications more carefully. Without proper planning, projects
            like this could go over budget. What do you think about scheduling
            another meeting to discuss the budget in more detail?
            """,
            ),
        ]

        service = CompressionService()

        logger.info("=" * 70)
        logger.info("LLM Text Compression System - Service Demo")
        logger.info("=" * 70)

        all_passed = True

        for name, text in tests:
            logger.info(f"{'─' * 70}")
            logger.info(f"Test: {name}")

            compressed = service.compress(text)
            decompressed = service.decompress(compressed)
            stats = service.get_stats()
            passed = service.verify_roundtrip(text, decompressed)
            all_passed &= passed

            logger.info(f"Original: {stats['original_chars']} chars, ~{stats['original_tokens']} tokens")
            logger.info(f"Compressed: {stats['compressed_chars']} chars, ~{stats['compressed_tokens']} tokens")
            logger.info(f"Ratio: {stats['compression_ratio']}x, Saved: {stats['space_saved_pct']}%")
            logger.info(f"Token reduction: {stats['token_reduction_pct']}%")
            logger.info(f"Dict subs: {stats['dict_subs']}, LZ77 refs: {stats['lz77_refs']}")
            logger.info(f"Roundtrip: {'✓ PASS' if passed else '✗ FAIL'}")

        logger.info("=" * 70)
        logger.info(f"All tests passed: {'✓ YES' if all_passed else '✗ NO'}")

    main()
