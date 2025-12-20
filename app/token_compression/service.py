from typing import Optional, Dict
from app.token_compression.config import CompressionConfig
from app.token_compression.compressor import Compressor
from app.token_compression.decompressor import Decompressor


class CompressionService:
    """
    Facade for compression and decompression services.
    """

    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        self.compressor = Compressor(self.config)
        self.decompressor = Decompressor()

    def compress(self, text: str) -> str:
        """Compress text."""
        return self.compressor.compress(text)

    def decompress(self, compressed_text: str) -> str:
        """Decompress text."""
        return self.decompressor.decompress(compressed_text)

    def get_stats(self) -> Dict:
        """Get latest compression stats."""
        return self.compressor.get_stats()

    @staticmethod
    def verify_roundtrip(original: str, decompressed: str) -> bool:
        """Verify lossless roundtrip."""
        # Normalize for comparison
        orig_norm = " ".join(original.split())
        decomp_norm = " ".join(decompressed.split())
        return orig_norm == decomp_norm


if __name__ == "__main__":

    def main():
        """Demonstrate the compression system."""
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

        # Use local instance
        service = CompressionService()

        print("=" * 70)
        print("LLM Text Compression System - Service Demo")
        print("=" * 70)

        all_passed = True

        for name, text in tests:
            print(f"\n{'─' * 70}")
            print(f"Test: {name}")
            print("─" * 70)

            compressed = service.compress(text)
            decompressed = service.decompress(compressed)
            stats = service.get_stats()
            passed = service.verify_roundtrip(text, decompressed)
            all_passed &= passed

            print(
                f"\nOriginal ({stats['original_chars']} chars, ~{stats['original_tokens']} tokens):"
            )
            print(text.strip()[:100] + "...")

            print(
                f"\nCompressed ({stats['compressed_chars']} chars, ~{stats['compressed_tokens']} tokens):"
            )
            print(compressed[:100] + "...")

            print(f"\nResults:")
            print(f"  • Compression ratio: {stats['compression_ratio']}x")
            print(f"  • Space saved: {stats['space_saved_pct']}%")
            print(f"  • Token reduction: {stats['token_reduction_pct']}%")
            print(f"  • Dictionary substitutions: {stats['dict_subs']}")
            print(f"  • LZ77 references: {stats['lz77_refs']}")
            print(f"  • Roundtrip verified: {'✓ PASS' if passed else '✗ FAIL'}")

        print(f"\n{'=' * 70}")
        print(f"All tests passed: {'✓ YES' if all_passed else '✗ NO'}")
        print("=" * 70)

    main()
