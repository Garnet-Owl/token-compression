import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Set
from app.token_compression.config import CompressionConfig
from app.token_compression.dictionary import DICTIONARY
from app.llm_integration.attention import replace_attention_layers


class CompressedLLM:
    """Wrapper for LLM that handles compressed text input with custom attention."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        config: Optional[CompressionConfig] = None,
        enable_decompression_detection: bool = True,
    ):
        self.config = config or CompressionConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.float16,
            attn_implementation="eager",
        ).to(device)
        self._reference_token_ids: Set[int] = set()
        self._setup_special_tokens()

        if enable_decompression_detection:
            self._setup_custom_attention()

    def _setup_special_tokens(self):
        lz77_tokens = ["‹", "›"]
        dict_tokens = list(DICTIONARY.values())

        all_tokens = lz77_tokens + dict_tokens
        new_tokens = [
            token for token in all_tokens if self.tokenizer.convert_tokens_to_ids(token) == self.tokenizer.unk_token_id
        ]

        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))

        self._build_reference_token_ids()

    def _build_reference_token_ids(self):
        lz77_tokens = ["‹", "›"]
        dict_tokens = list(DICTIONARY.values())
        reference_markers = lz77_tokens + dict_tokens

        for marker in reference_markers:
            token_id = self.tokenizer.convert_tokens_to_ids(marker)
            if token_id != self.tokenizer.unk_token_id:
                self._reference_token_ids.add(token_id)

    def _setup_custom_attention(self):
        replace_attention_layers(self.model, self._reference_token_ids)

    def generate(self, compressed_prompt: str, max_new_tokens: int = 50) -> str:
        inputs = self.tokenizer(compressed_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_token_ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)
