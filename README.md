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

## Can a model read it? (GPU tests)

I ran two short tests on Kaggle's free GPU. Scripts and raw results are in [`experiments/`](experiments/).

### Test 1: no training, legend in the prompt

Qwen3.5-4B, a short-answer prompt, and a legend that explains the compressed format.

| Task | Context | Answer score | Context tokens |
| --- | --- | --- | --- |
| Server logs, 100 questions (synthetic) | Original | 96% | 2,509 |
| | Stopwords removed | 93% | 2,509 |
| | Compressed + legend | **4%** | 1,883 |
| SQuAD, 200 questions (prose) | Original | 0.94 F1 | 113 |
| | Stopwords removed | 0.93 F1 | 90 |
| | Compressed + legend | 0.85 F1 | **752** |

- **Even a current model can't decode the format on its own.** On the logs it was right 4% of the time, against 96% with the original text.
- **The legend is expensive.** It adds about 640 tokens, so a short paragraph becomes 6.7x longer.
- **For prose, plain stopword removal is the better deal:** 21% fewer tokens for a 0.01 drop in F1.

An earlier run with an older model and a looser prompt is also in `experiments/`. Its SQuAD scores were mostly noise from answer length, which is why I reran it.

### Test 2: fine-tune a small model to read it (no legend)

Qwen3-1.7B with LoRA, trained on compressed contexts so it doesn't need the legend. Qwen3.5-2B ran out of memory on the free GPU, so I used Qwen3-1.7B. Training hit my 11-minute cap after 477 examples (I had planned 1,200).

| Task | Model and context | Answer score | Context tokens |
| --- | --- | --- | --- |
| Server logs | Base model, original text | 42% | 2,509 |
| | Fine-tuned, compressed text | **4%** | 1,292 |
| SQuAD | Base model, original text | 0.81 F1 | 114 |
| | Fine-tuned, compressed text | 0.83 F1 | **160** |

- **No legend means the logs shrink by about 49%,** but after 477 examples the model still can't read them (4%). That's too little training to say it can't be learned. It's a first look, not a verdict.
- **On prose, compression costs tokens.** The compressed text is 41% longer in tokens than the original.
- **The SQuAD score held up (0.83 against 0.81),** but that shows the model can answer from text. It doesn't show it understands the symbols.

Each test is a single run, about 18 and 16 GPU minutes.

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

- The original notebook fine-tuned on only 100 pairs for 50 epochs, which is probably overfitted. My newer test used 477 pairs for one epoch and a held-out test set, which is better but still small.
- Answer quality is tested on a small model with short training (above), so I can't claim the meaning is preserved.
- I haven't compared it with existing prompt-compression methods such as LLMLingua.

## What I'd test next

- Train on the full 1,200 examples for more than one epoch, and save the model's raw answers on the logs to see what it actually outputs.
- Swap the rare symbols (`①`, `‹45:19›`) for common words the tokenizer already knows, so the compressed text costs fewer tokens.
- Comparison with LLMLingua-2.
- Real logs, JSON and tool outputs instead of generated ones.
- A bigger model, once it fits the free GPU.

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
