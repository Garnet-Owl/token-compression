import re
from typing import Optional, Tuple, Dict
from app.token_compression.config import CompressionConfig
from app.token_compression.dictionary import DICTIONARY, to_base62


class Compressor:
    """
    Text compression logic optimized for LLM processing.
    """

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

    @staticmethod
    def _make_ref(distance: int, length: int) -> str:
        """Create compact reference: ‹distance:length›"""
        return f"‹{to_base62(distance)}:{to_base62(length)}›"

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Simple token count (words + punctuation)."""
        return len(re.findall(r"\S+", text))

    def _apply_dictionary(self, text: str) -> str:
        """Replace common phrases with single tokens."""
        result = text
        for phrase, token in sorted(DICTIONARY.items(), key=lambda x: -len(x[0])):
            count = result.count(phrase)
            if count > 0:
                result = result.replace(phrase, token)
                self.stats["dict_subs"] += count
        return result

    def _find_best_match(self, text: str, pos: int) -> Optional[Tuple[int, int]]:
        """
        Find longest match in sliding window.
        Returns (distance, length) or None.
        """
        window_start = max(0, pos - self.config.window_size)
        window = text[window_start:pos]
        lookahead = text[pos : pos + self.config.max_match_length]

        if len(lookahead) < self.config.min_match_length:
            return None

        # Try decreasing lengths until we find a match
        for length in range(
            min(len(lookahead), self.config.max_match_length),
            self.config.min_match_length - 1,
            -1,
        ):
            pattern = lookahead[:length]
            match_pos = window.rfind(pattern)

            if match_pos >= 0:
                distance = len(window) - match_pos
                ref = self._make_ref(distance, length)

                # Only use if we save enough characters
                if length - len(ref) >= self.config.min_savings:
                    return distance, length

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
