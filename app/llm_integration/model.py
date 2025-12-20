import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Set
from app.token_compression.config import CompressionConfig
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
        special_tokens = ["‹", "›"]
        common_dict_tokens = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]

        new_tokens = []
        for token in special_tokens + common_dict_tokens:
            if self.tokenizer.convert_tokens_to_ids(token) == self.tokenizer.unk_token_id:
                new_tokens.append(token)

        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))

        self._build_reference_token_ids()

    def _build_reference_token_ids(self):
        reference_markers = ["‹", "›"] + [
            "①",
            "②",
            "③",
            "④",
            "⑤",
            "⑥",
            "⑦",
            "⑧",
            "⑨",
            "⑩",
        ]
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
