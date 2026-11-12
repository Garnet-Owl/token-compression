# Token Compression for LLMs: compress logs to cut prompt tokens

Compress repetitive logs into templates and short rows, then ask an LLM questions about them. A fine-tuned Qwen3-1.7B answers **73%** of test questions correctly on the compressed log, against **77%** on the original, with **25% fewer prompt tokens**.

<p align="center">
  <img src="docs/token-compression-hero.png" alt="Many text tokens compressed into a smaller stream of symbols flowing into a language model" width="760">
</p>

[Results](#results) | [How it works](#how-it-works) | [Quick start](#quick-start) | [Limitations](#limitations) | [Model on Hugging Face](https://huggingface.co/Garnet-Owl/token-compression-log-reader)

This is research code. It tests whether prompt compression can reduce LLM token usage and API cost, and it reports where it works and where it doesn't.

## Why compress prompts

Prompts that carry logs, JSON or tool output are long and repetitive, and you pay for every token. Compression only helps if the model can still answer from the compressed text. So each method here is measured two ways: tokens counted with real tokenizers (`tiktoken`), and answer accuracy on held-out questions.

## Results

### Template compression with a fine-tuned reader

The best result comes from a template compressor for logs plus a small fine-tuned model. Test set: 150 new 30-line logs, with three kinds of question.

| Model and log | Correct | Prompt tokens |
| --- | --- | --- |
| Base Qwen3-1.7B, original log | 33% | 1,459 |
| Base Qwen3-1.7B, compressed log with a short legend | 16% | 1,147 |
| Fine-tuned, original log | 77% | 1,459 |
| Fine-tuned, compressed log, no legend | **73%** | **1,090** |

| Question | Fine-tuned, original | Fine-tuned, compressed |
| --- | --- | --- |
| Look up a request's status and duration | 100% | 100% |
| Highest duration among requests | 90% | 90% |
| Count the ERROR lines | 41% | 31% |

- The fine-tuned model reads the compressed log almost as well as the original. With 150 questions, the 4-point gap is about 5 questions, so it's within noise.
- Fine-tuning is what makes the format readable. Without it, the base model scores 16%, even with a legend.
- Counting is the weak spot. Both versions overcount ERROR lines, and the compressed one does worse.

### What doesn't work: dictionary + LZ77

The first compressor combines a phrase dictionary with LZ77 back-references (the idea behind ZIP). Token counts before and after, using `tiktoken` (a plus means fewer tokens, a minus means more):

| Text | `cl100k_base` | `o200k_base` |
| --- | --- | --- |
| Server logs (synthetic) | +36% | +46% |
| JSON records (synthetic) | +48% | +56% |
| Prose (this repo's maths spec) | -56% | -40% |
| Python code (this repo) | -74% | -50% |

It shrinks repetitive data but makes prose and code longer in tokens, because symbols like `①` and `‹45:19›` cost more tokens than the words they replace. Counting whitespace-separated words overstates the saving, so these tables use real tokenizers.

It's also unreadable without training. Qwen3.5-4B, given a legend that explains the format:

| Task | Context | Score | Context tokens |
| --- | --- | --- | --- |
| Server logs, 100 questions | Original | 96% | 2,509 |
| | Compressed + legend | **4%** | 1,883 |
| SQuAD, 200 questions | Original | 0.94 F1 | 113 |
| | Stopwords removed | 0.93 F1 | 90 |
| | Compressed + legend | 0.85 F1 | **752** |

The legend adds about 640 tokens, so short prompts get longer. For prose, plain stopword removal is the better trade: 21% fewer tokens for a 0.01 drop in F1.

Why it fails: the back-references point by character distance, and a model can't count characters. That's why the log compressor below uses a different design.

## How it works

### Template compression for logs

Each line shape is written once as a template, and every line becomes a short row of values, like a table. It's lossless: decompressing gives back the exact original text. It saves about 27% of tokens on a 30-line log, and about 32% on 60 lines. The ceiling is set by random numbers and IDs, which can't be compressed.

```
@1 2026-11-12 {} INFO request_id={} method={} path={} status={} duration_ms={}
@2 2026-11-12 {} ERROR upstream_timeout service={} after_ms={} request_id={}
---
1|10:22:09|16480894|GET|/api/v1/orders/299|200|845
2|10:22:35|search|2953|93082061
```

A value that is the same in every line of a template is written into the template. A line that doesn't fit is kept as raw text, marked with `!`. Code: [`app/structured_compression/`](app/structured_compression/compressor.py).

### Dictionary + LZ77

Common phrases become single symbols (`①`, `②`), and repeated text becomes a reference, `‹distance:length›`. For example, `"database connection pool ... database connection"` becomes `"database connection pool ... ‹45:19›"`. The maths is in [Mathematical Spec.md](Mathematical%20Spec.md). Code: [`app/token_compression/`](app/token_compression/).

### The fine-tuned reader

- **Model:** Qwen3-1.7B with LoRA adapters (rank 32, all linear layers), about 35M trainable parameters. It runs on a CPU for inference.
- **Training data:** 1,470 examples, about 80% compressed logs without a legend and 20% original logs, so one model works on both. The loss is on answer tokens only.
- **Efficiency:** examples are sorted by length and packed into batches with a fixed token budget, gradient checkpointing keeps memory near 7 GB, and the output layer runs only on answer positions.
- **Compute:** one pass on a single 16 GB T4 GPU, about 44 minutes. fp16, cosine learning-rate schedule.

Scripts, the model's raw answers and the full results are in [`experiments/log_reader/`](experiments/log_reader/).

## Fine-tuned model

The LoRA adapter is on Hugging Face: **[Garnet-Owl/token-compression-log-reader](https://huggingface.co/Garnet-Owl/token-compression-log-reader)**. See the model card for a loading example.

## Quick start

You need Python 3.11 or later and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/Garnet-Owl/token-compression.git
cd token-compression
uv sync
```

Compress a log and restore it:

```python
from app.structured_compression.compressor import StructuredCompressor

compressor = StructuredCompressor()
packed = compressor.compress(log_text)
assert compressor.decompress(packed) == log_text
```

Count the tokens saved by the dictionary + LZ77 compressor (this reproduces the token table above):

```bash
uv run python -m scripts.measure_tokens
```

Run the tests:

```bash
uv run pytest
```

To rebuild the log test set and train your own reader, follow [`experiments/log_reader/README.md`](experiments/log_reader/README.md).

### Question-answering API

The repo also has the original pipeline: a FastAPI service that compresses a context with dictionary + LZ77 and asks a fine-tuned model a question about it.

```bash
uv run python -m app.main_runner.api
```

Open http://localhost:8000/docs, or send a request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"context": "The database connection pool manages all connections...", "question": "What does the pool manage?"}'
```

## Limitations

- The logs are generated from one random seed. Real logs are messier, and results will differ.
- The test set is 150 questions and each number is a single run, so differences of a few points aren't meaningful.
- The reader was trained on this log format. Other formats need their own training.
- Counting questions are weak.
- The original dictionary + LZ77 pipeline was trained on a small set of 100 pairs and isn't evaluated on held-out data.
- There's no comparison yet with other prompt-compression methods such as LLMLingua.

## Roadmap

- Test on real logs and JSON from a running service.
- Improve counting, for example by adding a running count to the compressed rows.
- Train for more than one pass.
- Try a smaller model for CPU-only use.
- Compare against LLMLingua-2.

## Project layout

```
app/
  structured_compression/   template compressor for logs
  token_compression/        dictionary + LZ77 compressor
  fine_tuning/              LoRA training pipeline for the original pipeline
  main_runner/              FastAPI server and question-answering service
experiments/
  log_reader/               fine-tuned reader for compressed logs (main result)
  zero_shot_qa/             legend-only test of the dictionary + LZ77 format
scripts/measure_tokens.py   token counts before and after, with real tokenizers
tests/                      unit tests for the template compressor
```

## Contributing

Issues and pull requests are welcome, especially new experiments and baselines.

## Author

James Wanjiku, AI/ML engineer in Nairobi. [LinkedIn](https://www.linkedin.com/in/james-wanjiku/) | [Portfolio](https://garnet-owl.github.io/)

## License

MIT. See [LICENSE](LICENSE).
