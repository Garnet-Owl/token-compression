# Fine-tune a reader

Fine-tunes a small Qwen with LoRA on compressed contexts, so it needs no legend, then compares it with the base model on the original text.

1. `python experiments/fine_tune_reader/make_data.py <folder>` builds `train.jsonl` (600 log questions and 600 SQuAD questions) and `test.jsonl` (100 log and 200 SQuAD questions, the same as the zero-shot test). It uses 4 CPU cores and takes a few minutes.
2. Upload the folder as a Kaggle dataset and run `train.py` as a GPU kernel. It tries Qwen3.5-2B first and falls back to Qwen3-1.7B. Training stops after 11 minutes and the whole run after 26.
3. `results.json` has the output of the run described in the main README.
