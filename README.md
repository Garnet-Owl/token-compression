# Token Compression for LLMs: an experiment in prompt compression

Can you cut down what you send to an LLM by compressing the text first, the way ZIP does?

I tried it. This repo combines a dictionary of common phrases with LZ77 back-references (the idea behind ZIP), then fine-tunes a small model with LoRA to read the compressed text. It started after I watched a video about ZIP compression and wondered if the same trick could save tokens and API cost.

It's an experiment, and I'm keeping it public with the real results, including where it doesn't work.

## What I found

I counted real LLM tokens with OpenAI's `tiktoken` (`cl100k_base` and `o200k_base`), before and after compression. A plus means fewer tokens, a minus means more.

| Text | cl100k | o200k |
| --- | --- | --- |
| Server logs (synthetic, repetitive) | +36% | +46% |
| JSON records (synthetic, repetitive) | +48% | +56% |
| Prose (this repo's maths spec) | -56% | -40% |
| Python code (this repo) | -74% | -50% |

- **Repetitive data compresses well.** Logs and JSON with repeated keys and values need about a third to a half fewer tokens.
- **Prose and code get worse.** Modern tokenizers already pack normal English and code well. My replacement symbols and back-references, like `①` and `‹45:19›`, can cost more tokens than the words they replace.
- **My first numbers were misleading.** The old "30-50% token reduction" counted whitespace-separated words, not tokens from a real tokenizer. That's why it looked better than it is.

The logs and JSON are generated, not real traffic, so treat those two rows as a best case. Run `uv run python -m scripts.measure_tokens` to reproduce all four.

## How it works

```
Input text -> dictionary substitution -> LZ77 back-references -> compressed text
                                                                      |
                                                          fine-tuned LLM reads it
                                                                      |
                                                                  answer
```

1. **Dictionary.** Common phrases such as "the " and "and " become single symbols (`①`, `②`).
2. **LZ77.** Text that repeats an earlier part becomes a reference, `‹distance:length›`.
3. **Fine-tuned model.** Qwen2.5-0.5B with LoRA adapters (about 0.4% of the weights trained) learns to read the compressed format.

Example: `"database connection pool ... database connection"` becomes `"database connection pool ... ‹45:19›"`.

The maths is written up in [Mathematical Spec.md](Mathematical%20Spec.md): the LZ77 matching, a compression-aware attention idea, and a lazy decompression argument.

## Limits

- The fine-tune used only 100 (original, compressed) pairs for 50 epochs. That's small, so it's probably overfitted, and I haven't tested it on held-out data.
- I haven't measured answer quality after compression, so I can't claim the meaning is preserved.
- I haven't compared it with existing prompt-compression methods such as LLMLingua.

## What I'd test next

- Answer accuracy on real QA data (SQuAD, HotpotQA, LongBench) at different compression levels.
- Comparison with LLMLingua-2 and plain stopword removal.
- Whether a current model can read the compressed format with no fine-tuning, just a legend in the prompt, counting the legend's tokens too.
- Real logs, JSON and tool outputs, where the best results are.
- A bigger training set with a proper validation split.

## Try it

You need Python 3.11 or later and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/Garnet-Owl/token-compression.git
cd token-compression
uv sync
```

Start the API:

```bash
uv run python -m app.main_runner.api
```

Then open http://localhost:8000/docs, or send a request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"context": "The database connection pool manages all connections...", "question": "What does the pool manage?"}'
```

Or from Python:

```python
from app.main_runner.qa_service import QAService

service = QAService()
result = service.ask(
    context="The database connection pool manages all connections...",
    question="What does the pool manage?",
)
print(result["answer"])
```

The fine-tuning notebook is `notebooks/token_compression_colab_finetune.ipynb`. I trained on a Colab GPU.

## Project layout

```
app/
  token_compression/   dictionary + LZ77 compressor and decompressor
  fine_tuning/         LoRA training pipeline
  main_runner/         FastAPI server and Q&A service
scripts/
  measure_tokens.py    token counts before and after, with real tokenizers
notebooks/             fine-tuning notebook
```

## Contributing

Issues and pull requests are welcome, especially new experiments or better baselines.

## Author

I'm James Wanjiku, an AI/ML engineer in Nairobi. [LinkedIn](https://www.linkedin.com/in/james-wanjiku/) | [Portfolio](https://garnet-owl.github.io/)

## License

MIT. See [LICENSE](LICENSE).
