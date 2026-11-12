"""Does a small LLM still answer correctly when the context is compressed? Short GPU run, hard time limit."""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import json, random, re, string, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate"], check=False)
from collections import Counter

T0 = time.time()
BUDGET_S = 18 * 60  # stop generating after 20 minutes in total

subprocess.run(["git", "clone", "-q", "https://github.com/Garnet-Owl/token-compression", "tc"], check=True)
sys.path.insert(0, "tc")
from app.token_compression.compressor import Compressor
from app.token_compression.dictionary import DICTIONARY

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
except Exception as e:
    print("CausalLM load failed, trying image-text class:", e, flush=True)
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()

STOP = set("a an the of to in on at is are was were be been and or but for with from by as that this it its".split())
LEGEND = (
    "The context is compressed. Decode it in two steps.\n"
    "1. A reference like \u2039d:l\u203a means: copy l characters from d characters back in the text so far. "
    "d and l are base-62 numbers (digits 0-9, then a-z, then A-Z).\n"
    "2. These symbols stand for phrases: "
    + "; ".join(f"{sym} = {phrase!r}" for phrase, sym in DICTIONARY.items())
    + "\n\nCompressed context:\n"
)

def strip_stopwords(t): return " ".join(w for w in t.split() if w.lower().strip(string.punctuation) not in STOP)
def compress(t): return Compressor().compress(t)

def norm(s):
    s = s.lower(); s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    common = Counter(p) & Counter(g); n = sum(common.values())
    if not p or not g or n == 0: return 0.0
    pr, rc = n / len(p), n / len(g); return 2 * pr * rc / (pr + rc)

def ask(items, build, bs=8):
    outs = []
    for i in range(0, len(items), bs):
        if time.time() - T0 > BUDGET_S: outs += [""] * (len(items) - i); break
        batch = items[i:i + bs]
        prompts = [tok.apply_chat_template([{"role": "user", "content": build(x)}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for x in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad(): gen = model.generate(**enc, max_new_tokens=16, do_sample=False)
        outs += tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs

def ntok(t): return len(tok(t).input_ids)

# --- task 1: SQuAD (prose) ---
squad = [dict(c=r["context"], q=r["question"], a=r["answers"]["text"]) for r in load_dataset("rajpurkar/squad", split="validation[:200]")]
# --- task 2: log lookup (repetitive, synthetic) ---
random.seed(7); logs = []
for _ in range(100):
    rows = [(random.randint(10**7, 10**8), random.choice([200, 200, 200, 404, 500]), random.randint(5, 90), random.randint(1, 500)) for _ in range(40)]
    text = "\n".join(f"2026-11-12 10:{i % 60:02d}:00 INFO request_id={r[0]} method=GET path=/api/v1/users/{r[3]} status={r[1]} duration_ms={r[2]} user_agent=Mozilla/5.0" for i, r in enumerate(rows))
    t = random.choice(rows); logs.append(dict(c=text, q=f"What were the status and duration_ms of request_id={t[0]}?", a=[str(t[1]), str(t[2])]))

methods = {"original": lambda c: c, "stopwords removed": strip_stopwords, "compressed + legend": lambda c: LEGEND + compress(c)}
results = {}
for task, data in (("squad", squad), ("logs", logs)):
    for name, fn in methods.items():
        build = lambda x, fn=fn: f"{fn(x['c'])}\n\nQuestion: {x['q']}\nAnswer with only the answer text, a few words at most, and no explanation." if task == "squad" else f"Log:\n{fn(x['c'])}\n\nQuestion: {x['q']}\nReply with only the status code and the duration_ms number, nothing else."
        preds = ask(data, build, bs=2 if task == 'logs' else 8)
        done = [(p, x) for p, x in zip(preds, data) if p != ""]
        if task == "squad":
            score = sum(max(f1(p, g) for g in x["a"]) for p, x in done) / max(1, len(done))
            em = sum(max(norm(p) == norm(g) for g in x["a"]) for p, x in done) / max(1, len(done))
        else:
            score = em = sum(all(a in p for a in x["a"]) for p, x in done) / max(1, len(done))
        toks = sum(ntok(fn(x["c"])) for x in data) / len(data)
        results[f"{task}/{name}"] = dict(answered=len(done), score=round(score, 3), exact=round(em, 3), mean_context_tokens=round(toks, 1))
        print(task, name, results[f"{task}/{name}"], flush=True)

results["_meta"] = dict(model=MODEL, gpu_seconds=round(time.time() - T0))
json.dump(results, open("/kaggle/working/results.json", "w"), indent=1)
print(json.dumps(results, indent=1))
