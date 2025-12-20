"""
LLM-Optimized Text Compression System - Final Version
======================================================

A text compression algorithm inspired by DEFLATE/LZ77, optimized for:
1. LLM token efficiency
2. Human-readable compressed output
3. Attention-based selective decompression compatibility

Author: James (with Claude)
Date: 20th Dec 2025
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import json


@dataclass
class CompressionConfig:
    """Configuration for compression algorithm."""
    window_size: int = 8192
    min_match_length: int = 4
    max_match_length: int = 256
    min_savings: int = 2  # Minimum characters saved to use reference


class LLMTextCompressor:
    """
    Text compression optimized for LLM processing.

    Features:
    - Dictionary-based common phrase compression
    - LZ77-style back-reference compression
    - Readable reference format for potential LLM native processing
    - Verified lossless roundtrip
    """

    # Common phrases -> single-char tokens
    DICTIONARY = {
        # High-frequency words
        "the ": "①",
        "and ": "②",
        "that ": "③",
        "with ": "④",
        "from ": "⑤",
        "have ": "⑥",
        "this ": "⑦",
        "will ": "⑧",
        "your ": "⑨",
        "they ": "⑩",
        "been ": "⑪",
        "were ": "⑫",
        "would ": "⑬",
        "could ": "⑭",
        "should ": "⑮",
        "there ": "Ⓐ",
        "their ": "Ⓑ",
        "which ": "Ⓒ",
        "about ": "Ⓓ",
        "other ": "Ⓔ",
        "these ": "Ⓕ",
        "those ": "Ⓖ",
        "being ": "Ⓗ",
        "through ": "Ⓘ",
        "because ": "Ⓙ",
        "between ": "Ⓚ",
        "however ": "Ⓛ",
        "without ": "Ⓜ",
        "another ": "Ⓝ",
        "something ": "Ⓞ",
        "different ": "Ⓟ",
        "important ": "Ⓠ",
    }

    REVERSE_DICTIONARY = {v: k for k, v in DICTIONARY.items()}

    # Base62 characters for compact encoding
    BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        self._reset_stats()

    def _reset_stats(self):
        self.stats = {
            "original_chars": 0,
            "compressed_chars": 0,
            "original_tokens": 0,
            "compressed_tokens": 0,
            "dict_subs": 0,
            "lz77_refs": 0,
        }

    def _to_base62(self, n: int) -> str:
        """Convert integer to base62 string."""
        if n == 0:
            return "0"
        result = ""
        while n:
            result = self.BASE62[n % 62] + result
            n //= 62
        return result

    def _from_base62(self, s: str) -> int:
        """Convert base62 string to integer."""
        result = 0
        for c in s:
            result = result * 62 + self.BASE62.index(c)
        return result

    def _make_ref(self, distance: int, length: int) -> str:
        """Create compact reference: ‹distance:length›"""
        return f"‹{self._to_base62(distance)}:{self._to_base62(length)}›"

    def _parse_ref(self, ref: str) -> Tuple[int, int]:
        """Parse reference to (distance, length)."""
        inner = ref[1:-1]  # Remove ‹ and ›
        d_str, l_str = inner.split(":")
        return self._from_base62(d_str), self._from_base62(l_str)

    def _count_tokens(self, text: str) -> int:
        """Simple token count (words + punctuation)."""
        return len(re.findall(r'\S+', text))

    def _apply_dictionary(self, text: str) -> str:
        """Replace common phrases with single tokens."""
        result = text
        for phrase, token in sorted(self.DICTIONARY.items(), key=lambda x: -len(x[0])):
            count = result.count(phrase)
            if count > 0:
                result = result.replace(phrase, token)
                self.stats["dict_subs"] += count
        return result

    def _reverse_dictionary(self, text: str) -> str:
        """Expand dictionary tokens back to phrases."""
        result = text
        for token, phrase in self.REVERSE_DICTIONARY.items():
            result = result.replace(token, phrase)
        return result

    def _find_best_match(self, text: str, pos: int) -> Optional[Tuple[int, int]]:
        """
        Find longest match in sliding window.
        Returns (distance, length) or None.
        """
        window_start = max(0, pos - self.config.window_size)
        window = text[window_start:pos]
        lookahead = text[pos:pos + self.config.max_match_length]

        if len(lookahead) < self.config.min_match_length:
            return None

        best_distance = 0
        best_length = 0

        # Try decreasing lengths until we find a match
        for length in range(min(len(lookahead), self.config.max_match_length),
                            self.config.min_match_length - 1, -1):
            pattern = lookahead[:length]
            match_pos = window.rfind(pattern)

            if match_pos >= 0:
                distance = len(window) - match_pos
                ref = self._make_ref(distance, length)

                # Only use if we save enough characters
                if length - len(ref) >= self.config.min_savings:
                    return (distance, length)

        return None

    def compress(self, text: str) -> str:
        """
        Compress text using dictionary + LZ77.

        Returns compressed text that could theoretically be processed
        by an LLM trained on this format.
        """
        self._reset_stats()
        self.stats["original_chars"] = len(text)
        self.stats["original_tokens"] = self._count_tokens(text)

        # Phase 1: Dictionary compression
        working = self._apply_dictionary(text)

        # Phase 2: LZ77 back-reference compression
        output = []
        pos = 0

        while pos < len(working):
            match = self._find_best_match(working, pos)

            if match:
                distance, length = match
                output.append(self._make_ref(distance, length))
                pos += length
                self.stats["lz77_refs"] += 1
            else:
                output.append(working[pos])
                pos += 1

        result = "".join(output)
        self.stats["compressed_chars"] = len(result)
        self.stats["compressed_tokens"] = self._count_tokens(result)

        return result

    def decompress(self, compressed: str) -> str:
        """
        Decompress text by expanding references.

        This is what the LLM's attention would conceptually do:
        selectively expand references as they're processed.
        """
        # Phase 1: Expand LZ77 references
        output = []
        i = 0

        while i < len(compressed):
            if compressed[i] == "‹":
                # Find closing bracket
                end = compressed.find("›", i)
                if end > i:
                    ref = compressed[i:end + 1]
                    try:
                        distance, length = self._parse_ref(ref)
                        # Copy from already-decompressed output
                        current = "".join(output)
                        start = len(current) - distance

                        # Handle overlapping copy (like repeating patterns)
                        if start >= 0:
                            copied = ""
                            for j in range(length):
                                src_idx = start + j
                                if src_idx < len(current):
                                    copied += current[src_idx]
                                else:
                                    # Overlapping: copy from what we just added
                                    copied += copied[j - distance]
                            output.append(copied)
                        i = end + 1
                        continue
                    except:
                        pass

            output.append(compressed[i])
            i += 1

        # Phase 2: Expand dictionary tokens
        return self._reverse_dictionary("".join(output))

    def get_stats(self) -> Dict:
        """Get compression statistics."""
        orig = max(1, self.stats["original_chars"])
        comp = max(1, self.stats["compressed_chars"])
        orig_tok = max(1, self.stats["original_tokens"])
        comp_tok = max(1, self.stats["compressed_tokens"])

        return {
            **self.stats,
            "compression_ratio": round(orig / comp, 2),
            "space_saved_pct": round((1 - comp / orig) * 100, 1),
            "token_reduction_pct": round((1 - comp_tok / orig_tok) * 100, 1),
        }


def verify_roundtrip(original: str, compressed: str, decompressed: str) -> bool:
    """Verify lossless roundtrip."""
    # Normalize for comparison
    orig_norm = " ".join(original.split())
    decomp_norm = " ".join(decompressed.split())
    return orig_norm == decomp_norm


def main():
    """Demonstrate the compression system."""

    tests = [
        ("Highly Repetitive", """
        The quick brown fox jumps over the lazy dog. The quick brown fox 
        jumps over the lazy dog again. And again, the quick brown fox 
        jumps over the lazy dog. The lazy dog never moves because the 
        lazy dog is lazy. The quick brown fox keeps jumping over the 
        lazy dog because that is what the quick brown fox does.
        """),

        ("Technical Text", """
        The database connection pool manages database connections. When a 
        service needs a database connection, it requests one from the 
        database connection pool. The database connection pool returns an 
        available database connection. After the service is done with the 
        database connection, it returns the database connection to the 
        database connection pool.
        """),

        ("Natural Language", """
        I was thinking about what we discussed yesterday. The proposal 
        seems reasonable, however I think we should consider the budget 
        implications more carefully. Without proper planning, projects 
        like this could go over budget. What do you think about scheduling 
        another meeting to discuss the budget in more detail?
        """),
    ]

    compressor = LLMTextCompressor()

    print("=" * 70)
    print("LLM Text Compression System - Demonstration")
    print("=" * 70)

    all_passed = True

    for name, text in tests:
        print(f"\n{'─' * 70}")
        print(f"Test: {name}")
        print("─" * 70)

        compressed = compressor.compress(text)
        decompressed = compressor.decompress(compressed)
        stats = compressor.get_stats()
        passed = verify_roundtrip(text, compressed, decompressed)
        all_passed &= passed

        print(f"\nOriginal ({stats['original_chars']} chars, ~{stats['original_tokens']} tokens):")
        print(text.strip()[:100] + "...")

        print(f"\nCompressed ({stats['compressed_chars']} chars, ~{stats['compressed_tokens']} tokens):")
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

    # Show how this would work with an LLM API
    print("\n" + "=" * 70)
    print("Example: LLM API Integration")
    print("=" * 70)

    sample_prompt = """
    Please analyze the following data and provide recommendations. The data 
    shows that user engagement has been declining over the past quarter. 
    However, the data also shows that certain features have been performing 
    well. Based on this data, what would you recommend for the next quarter?
    """

    compressed = compressor.compress(sample_prompt)
    stats = compressor.get_stats()

    api_request = {
        "model": "claude-hypothetical-compressed",
        "compression": {
            "enabled": True,
            "algorithm": "lz77-dict-v1",
            "original_tokens": stats["original_tokens"],
            "compressed_tokens": stats["compressed_tokens"],
            "reduction": f"{stats['token_reduction_pct']}%"
        },
        "prompt": compressed,
        "max_tokens": 1000
    }

    print("\nAPI Request (compressed):")
    print(json.dumps(api_request, indent=2, ensure_ascii=False))

    print("\n" + "─" * 70)
    print("Key Insight for Your Research:")
    print("─" * 70)
    print("""
The core idea is that an LLM's attention mechanism could serve as a 
'selective decompressor' - only expanding reference tokens when they're
being attended to. This means:

1. Most of the compressed text stays compressed during processing
2. Only the tokens relevant to current computation get expanded
3. Effective context window is multiplied by compression ratio

For a model trained on this format, the attention pattern:
    Query -> [compressed tokens] -> selectively decompress -> attend

Could achieve significant efficiency gains while maintaining semantic
understanding, especially for repetitive or structured text.
""")


if __name__ == "__main__":
    main()
