from app.token_compression.llm_model import CompressedLLM
from app.token_compression.service import CompressionService


def main():
    print("Loading Qwen model...")
    llm = CompressedLLM()
    service = CompressionService()

    text = "The quick brown fox jumps over the lazy dog."
    compressed = service.compress(text)

    print(f"\nOriginal: {text}")
    print(f"Compressed: {compressed}")

    # Inspect tokenization
    tokens = llm.get_token_ids(compressed)
    print(f"\nToken IDs: {tokens}")
    print(f"Token count: {len(tokens)}")

    decoded = llm.tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")

    print("\nAttempting generation (un-fine-tuned)...")
    try:
        output = llm.generate(compressed, max_new_tokens=20)
        print(f"Output: {output}")
    except Exception as e:
        print(f"Generation failed: {e}")


if __name__ == "__main__":
    main()
