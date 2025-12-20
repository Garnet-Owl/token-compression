import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List
from app.token_compression.config import CompressionConfig


class CompressedLLM:
    """
    Wrapper for LLM that handles compressed text input.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        config: Optional[CompressionConfig] = None,
    ):
        self.config = config or CompressionConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.float16,
        ).to(device)
        self._setup_special_tokens()

    def _setup_special_tokens(self):
        """Ensure tokenizer handles our compression tokens correctly."""
        # Add dictionary tokens and reference markers if not present
        special_tokens = ["‹", "›"]
        # Add a few common dictionary tokens to ensure they are single tokens
        common_dict_tokens = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]

        # Check if we need to add tokens
        new_tokens = []
        for token in special_tokens + common_dict_tokens:
            if (
                self.tokenizer.convert_tokens_to_ids(token)
                == self.tokenizer.unk_token_id
            ):
                new_tokens.append(token)

        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))

    def generate(self, compressed_prompt: str, max_new_tokens: int = 50) -> str:
        """
        Generate text from a compressed prompt.
        For now, this just passes the compressed text directly to the model
        to see how it handles the tokens.
        """
        inputs = self.tokenizer(compressed_prompt, return_tensors="pt").to(
            self.model.device
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_token_ids(self, text: str) -> List[int]:
        """Helper to inspect tokenization."""
        return self.tokenizer.encode(text)
