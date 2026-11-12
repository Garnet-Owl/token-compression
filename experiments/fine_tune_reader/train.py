"""Fine-tune a small Qwen with LoRA to read compressed contexts (no legend), then compare with the base model on original text.
Hard limits: training stops at 11 minutes, the whole run at 26 minutes."""
import json, random, re, string, subprocess, sys, time
from collections import Counter
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "peft"], check=False)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
T0 = time.time(); TRAIN_S = 11 * 60; TOTAL_S = 26 * 60
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

import glob, os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
D = os.path.dirname(glob.glob("/kaggle/input/**/train.jsonl", recursive=True)[0]) + "/"
print("data dir:", D, flush=True)
train = [json.loads(l) for l in open(D + "train.jsonl", encoding="utf-8")]
test = [json.loads(l) for l in open(D + "test.jsonl", encoding="utf-8")]
random.seed(0); random.shuffle(train)


def prompt_text(task, ctx, q):
    if task == "squad":
        return f"{ctx}\n\nQuestion: {q}\nAnswer with only the answer text, a few words at most, and no explanation."
    return f"Log:\n{ctx}\n\nQuestion: {q}\nReply with only the status code and the duration_ms number, nothing else."


def answer_text(x):
    return x["a"][0] if x["task"] == "squad" else " ".join(x["a"])


def load(name):
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    try:
        m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).cuda()
    except Exception as e:
        print("CausalLM load failed:", e, flush=True)
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(name, torch_dtype=torch.float16).cuda()
    return tok, m


def chat(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def encode(tok, x, field):
    p = tok(chat(tok, prompt_text(x["task"], x[field], x["q"])), add_special_tokens=False).input_ids
    a = tok(answer_text(x), add_special_tokens=False).input_ids + [tok.convert_tokens_to_ids("<|im_end|>")]
    return p, a


def build(name):
    tok, m = load(name)
    m.gradient_checkpointing_enable()
    m.enable_input_require_grads()
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))
    for p in m.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
    scaler = torch.amp.GradScaler()
    return tok, m, opt, scaler


def step(tok, m, opt, scaler, x, accum_every, i):
    p, a = encode(tok, x, "cc")
    if len(p) + len(a) > 2600:
        return None
    ids = torch.tensor([p + a]).cuda()
    labels = torch.tensor([[-100] * len(p) + a]).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = m(input_ids=ids, labels=labels).loss / accum_every
    scaler.scale(loss).backward()
    if (i + 1) % accum_every == 0:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
    return loss.item() * accum_every, len(p) + len(a)


chosen = None
for name in ("Qwen/Qwen3.5-2B", "Qwen/Qwen3-1.7B"):
    try:
        tok, m, opt, scaler = build(name)
        t = time.time(); toks = 0
        for i in range(4):
            r = step(tok, m, opt, scaler, train[i], 8, i)
            if r:
                toks += r[1]
        torch.cuda.synchronize(); dt = time.time() - t
        total_tokens = sum(len(x["cc"]) / 3.2 + 60 for x in train)  # rough estimate
        projected = total_tokens / max(1, toks / dt)
        print(f"{name}: {toks/dt:.0f} tok/s in probe, projected full epoch {projected/60:.1f} min", flush=True)
        if projected <= TRAIN_S * 1.3 or name.endswith("1.7B"):
            chosen = name
            break
        del m, opt
        torch.cuda.empty_cache()
    except Exception as e:
        print(name, "failed:", repr(e)[:300], flush=True)
        try:
            del m
        except Exception:
            pass
        torch.cuda.empty_cache()
assert chosen, "no model worked"
print("using", chosen, flush=True)
del m
torch.cuda.empty_cache()
tok, m, opt, scaler = build(chosen)

m.train(); done = 0; losses = []; ttrain = time.time()
for i, x in enumerate(train):
    if time.time() - ttrain > TRAIN_S:
        break
    r = step(tok, m, opt, scaler, x, 8, i)
    if r:
        done += 1
        losses.append(r[0])
    if (i + 1) % 100 == 0:
        print(f"step {i+1}, mean loss last 100: {sum(losses[-100:])/len(losses[-100:]):.3f}", flush=True)
m.eval()
print("trained on", done, "examples in", round(time.time() - ttrain), "s", flush=True)


def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if not p or not g or n == 0:
        return 0.0
    pr, rc = n / len(p), n / len(g)
    return 2 * pr * rc / (pr + rc)


def gen(items, field, tuned, bs=8):
    outs = []
    for i in range(0, len(items), bs):
        if time.time() - T0 > TOTAL_S:
            outs += [None] * (len(items) - i)
            break
        prompts = [chat(tok, prompt_text(x["task"], x[field], x["q"])) for x in items[i:i + bs]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            if tuned:
                g = m.generate(**enc, max_new_tokens=14, do_sample=False)
            else:
                with m.disable_adapter():
                    g = m.generate(**enc, max_new_tokens=14, do_sample=False)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


results = {}
for task in ("logs", "squad"):
    items = [x for x in test if x["task"] == task]
    for label, field, tuned in (("base model, original text", "c", False), ("fine-tuned, compressed text (no legend)", "cc", True)):
        preds = gen(items, field, tuned)
        pairs = [(p, x) for p, x in zip(preds, items) if p is not None]
        if task == "squad":
            score = sum(max(f1(p, g) for g in x["a"]) for p, x in pairs) / max(1, len(pairs))
            em = sum(max(norm(p) == norm(g) for g in x["a"]) for p, x in pairs) / max(1, len(pairs))
        else:
            score = em = sum(all(a in p for a in x["a"]) for p, x in pairs) / max(1, len(pairs))
        mt = sum(len(tok(x[field]).input_ids) for x in items) / len(items)
        results[f"{task}/{label}"] = dict(answered=len(pairs), score=round(score, 3), exact=round(em, 3), mean_context_tokens=round(mt, 1))
        print(task, label, results[f"{task}/{label}"], flush=True)
results["_meta"] = dict(model=chosen, trained_examples=done, gpu_seconds=round(time.time() - T0), last_loss=round(sum(losses[-50:]) / max(1, len(losses[-50:])), 3))
json.dump(results, open("/kaggle/working/results.json", "w"), indent=1)
print(json.dumps(results, indent=1))
