# Log reader: fine-tune a model to read template-compressed logs

The first compressor (dictionary + LZ77) was unreadable to a model, so this version compresses repeated log lines into templates and short rows (see `app/structured_compression/`). The test asks whether a small model can still answer questions about the compressed log.

1. `python experiments/log_reader/make_data.py <folder> 3000 150` writes `train.jsonl` and `test.jsonl`. Every example is checked to decompress back to the exact original.
2. Upload the folder as a Kaggle dataset and run `train.py` as a GPU kernel (one T4).

`train.py` fine-tunes Qwen3-1.7B with LoRA. Examples are sorted by length and packed into batches with a fixed token budget, gradient checkpointing keeps memory near 7 GB, and the output layer runs only on answer positions. About 80% of the training examples are compressed logs and 20% are original logs, so one model can be tested on both.

- `results.json`: accuracy per condition and per question type, token counts, training numbers.
- `predictions.jsonl`: the model's raw answer to every test question.
