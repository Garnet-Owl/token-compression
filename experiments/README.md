# Experiments

Each folder is one test. Both ran on Kaggle's free GPU.

- [`zero_shot_qa/`](zero_shot_qa/): can a model answer questions from compressed text with no training, using a legend in the prompt? `run_qwen35.py` is the main run. `run_qwen25.py` is the first, weaker run.
- [`log_reader/`](log_reader/): the main result. Fine-tune a small model to read template-compressed logs.
- [`fine_tune_reader/`](fine_tune_reader/): does fine-tuning a small model remove the need for the legend? Build the data with `make_data.py`, then train and test with `train.py`.

The kernel scripts expect Kaggle paths (`/kaggle/input/...`) and internet access.
