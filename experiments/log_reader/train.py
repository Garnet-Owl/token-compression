"""Fine-tune Qwen3-1.7B with LoRA to answer questions about template-compressed logs.

Runs as a Kaggle GPU kernel (one T4, 16 GB). Set SMOKE=1 for a 3-minute check.

Design
- Model: Qwen3-1.7B in fp16, LoRA (rank 32) on every linear layer, fp32 adapter weights.
- Data: 80% compressed logs (no legend) and 20% original logs, so one model can be
  tested on both formats. Loss is on the answer tokens only.
- Batching: examples are sorted by length and packed into micro-batches with a fixed
  token budget, then the micro-batches are shuffled. Three micro-batches make one update.
- Memory: gradient checkpointing, and the output layer is applied only at answer
  positions, so the 150k-word vocabulary never produces logits for the whole prompt.
- Schedule: cosine learning rate that follows whichever is further along, steps or
  the time limit, so the run always ends with a fully decayed rate.
"""

import glob
import json
import math
import os
import random
import re
import subprocess
import sys
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "peft"], check=False)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

T0 = time.time()
SMOKE = os.environ.get("SMOKE") == "1"
MODEL = "Qwen/Qwen3-1.7B"
N_COMP, N_ORIG, N_TEST = (40, 10, 12) if SMOKE else (1500, 400, 150)
TRAIN_S = 150 if SMOKE else 38 * 60
TOTAL_S = 6 * 60 if SMOKE else 62 * 60
TOKENS_PER_BATCH = 5000
ACCUM = 3
LR, WARMUP = 2e-4, 10

data_dir = os.path.dirname(glob.glob("/kaggle/input/**/train.jsonl", recursive=True)[0])
train = [json.loads(line) for line in open(f"{data_dir}/train.jsonl", encoding="utf-8")]
test = [json.loads(line) for line in open(f"{data_dir}/test.jsonl", encoding="utf-8")][:N_TEST]
random.seed(0)
random.shuffle(train)

INSTRUCTION = ("Reply with only the answer. For a request_id question give the status code and duration_ms. "
               "For a count or a highest value give just the number.")
LEGEND = ("The log below is stored as templates and rows. A line starting with @ defines a template, and {} marks a value "
          "that changes. A row 'id|a|b' fills the {} slots of template @id in order. A row starting with ! is a raw line.\n\n")


def build_prompt(x, kind):
    if kind == "original":
        body = f"Log:\n{x['ctx']}"
    elif kind == "compressed_legend":
        body = f"{LEGEND}Log:\n{x['ctx_comp']}"
    else:
        body = f"Log:\n{x['ctx_comp']}"
    return f"{body}\n\nQuestion: {x['q']}\n{INSTRUCTION}"


tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda()
EOS = tok.convert_tokens_to_ids("<|im_end|>")


def chat(user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


# ---------- training examples ----------
examples = []
for x in train[:N_COMP]:
    examples.append(("compressed", x))
for x in train[N_COMP:N_COMP + N_ORIG]:
    examples.append(("original", x))
encoded = []
for kind, x in examples:
    prompt = tok(chat(build_prompt(x, kind)), add_special_tokens=False).input_ids
    answer = tok(x["a"], add_special_tokens=False).input_ids + [EOS]
    encoded.append((prompt, answer, kind))


def make_batches(items, rng):
    """Sort by length, pack under a token budget (padded size), then shuffle the batches."""
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]) + len(items[i][1]))
    batches, current, longest = [], [], 0
    for i in order:
        length = len(items[i][0]) + len(items[i][1])
        if current and max(longest, length) * (len(current) + 1) > TOKENS_PER_BATCH:
            batches.append(current)
            current, longest = [], 0
        current.append(i)
        longest = max(longest, length)
    if current:
        batches.append(current)
    rng.shuffle(batches)
    return batches


def collate(items, idx):
    longest = max(len(items[i][0]) + len(items[i][1]) for i in idx)
    ids = torch.full((len(idx), longest), EOS, dtype=torch.long)
    mask = torch.zeros((len(idx), longest), dtype=torch.long)
    labels = torch.full((len(idx), longest), -100, dtype=torch.long)
    for row, i in enumerate(idx):
        prompt, answer, _ = items[i]
        seq = prompt + answer
        ids[row, :len(seq)] = torch.tensor(seq)
        mask[row, :len(seq)] = 1
        labels[row, len(prompt):len(seq)] = torch.tensor(answer)
    return ids.cuda(), mask.cuda(), labels.cuda()


# ---------- model ----------
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.train()  # gradient checkpointing and dropout only apply in train mode
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"trainable parameters: {sum(p.numel() for p in trainable) / 1e6:.1f}M", flush=True)
optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
scaler = torch.amp.GradScaler()
inner = model.base_model.model  # the Qwen model with LoRA layers


def answer_loss(ids, mask, labels):
    with torch.autocast("cuda", dtype=torch.float16):
        hidden = inner.model(input_ids=ids, attention_mask=mask).last_hidden_state
        shifted = labels[:, 1:]
        keep = shifted != -100
        logits = inner.lm_head(hidden[:, :-1][keep])
    return torch.nn.functional.cross_entropy(logits.float(), shifted[keep])


rng = random.Random(1)
batches_per_epoch = None
train_start, updates, micro, seen, losses = time.time(), 0, 0, 0, []
epoch, stop = 0, False
planned_updates = None
while not stop:
    batches = make_batches(encoded, rng)
    batches_per_epoch = len(batches)
    if planned_updates is None:
        planned_updates = 2 * batches_per_epoch // ACCUM  # two epochs
    for batch in batches:
        elapsed = time.time() - train_start
        progress = max(updates / max(1, planned_updates), elapsed / TRAIN_S)
        if progress >= 1.0:
            stop = True
            break
        lr = LR * min(1.0, (updates + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss = answer_loss(*collate(encoded, batch))
        scaler.scale(loss / ACCUM).backward()
        losses.append(loss.item())
        seen += len(batch)
        micro += 1
        if micro % ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            if updates % 10 == 0:
                print(f"update {updates}/{planned_updates}  examples {seen}  loss {sum(losses[-20:]) / len(losses[-20:]):.3f}  "
                      f"lr {lr:.2e}  {time.time() - train_start:.0f}s  mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    epoch += 1
model.eval()
print(f"trained: {updates} updates, {seen} examples, {epoch} epochs, {time.time() - train_start:.0f}s", flush=True)


# ---------- evaluation ----------
def correct(pred, x):
    nums = re.findall(r"\d+", pred)
    gold = re.findall(r"\d+", x["a"])
    if x["qtype"] == "lookup":
        return nums[:2] == gold
    return nums[:1] == gold


def generate(kind, tuned, bs=8):
    outs = []
    for i in range(0, len(test), bs):
        if time.time() - T0 > TOTAL_S:
            outs += [None] * (len(test) - i)
            break
        prompts = [chat(build_prompt(x, kind)) for x in test[i:i + bs]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            if tuned:
                out = model.generate(**enc, max_new_tokens=10, do_sample=False)
            else:
                with model.disable_adapter():
                    out = model.generate(**enc, max_new_tokens=10, do_sample=False)
        outs += tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


conditions = [
    ("base model, original log", "original", False),
    ("base model, compressed log + short legend", "compressed_legend", False),
    ("fine-tuned, original log", "original", True),
    ("fine-tuned, compressed log (no legend)", "compressed", True),
]
results, predictions = {}, []
for label, kind, tuned in conditions:
    preds = generate(kind, tuned)
    pairs = [(p, x) for p, x in zip(preds, test) if p is not None]
    hits = [correct(p, x) for p, x in pairs]
    by_type = {}
    for (p, x), h in zip(pairs, hits):
        by_type.setdefault(x["qtype"], []).append(h)
    tokens = sum(len(tok(build_prompt(x, kind)).input_ids) for x in test) / len(test)
    results[label] = dict(answered=len(pairs), accuracy=round(sum(hits) / max(1, len(hits)), 3),
                          by_question={k: round(sum(v) / len(v), 3) for k, v in by_type.items()}, mean_prompt_tokens=round(tokens))
    print(label, results[label], flush=True)
    for (p, x), h in zip(pairs, hits):
        predictions.append(dict(condition=label, qtype=x["qtype"], question=x["q"], gold=x["a"], pred=p, correct=h))

results["_meta"] = dict(model=MODEL, train_examples_seen=seen, updates=updates, epochs=epoch, train_seconds=round(time.time() - train_start),
                        gpu_seconds=round(time.time() - T0), final_loss=round(sum(losses[-30:]) / max(1, len(losses[-30:])), 3),
                        peak_memory_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1), smoke=SMOKE)
os.makedirs("/kaggle/working", exist_ok=True)
json.dump(results, open("/kaggle/working/results.json", "w"), indent=1)
with open("/kaggle/working/predictions.jsonl", "w", encoding="utf-8") as f:
    for row in predictions:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
if not SMOKE:
    model.save_pretrained("/kaggle/working/adapter")
print(json.dumps(results, indent=1))
