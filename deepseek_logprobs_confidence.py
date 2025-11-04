# deepseek_logprobs_confidence.py
# pip install openai tiktoken numpy

import os, asyncio, math, numpy as np
from typing import List, Dict, Tuple
from openai import OpenAI, AsyncOpenAI
import tiktoken

# DeepSeek配置
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")  # DeepSeek模型
TOP_K = 5  # top_logprobs 允许 0-5
TEMPERATURE = 0  # 建议取 0 做稳定评分

# 在这里直接设置你的 DeepSeek API 密钥
DEEPSEEK_API_KEY = "sk-f76caab6ef614fa9b6c368616a998022"  # ← 在这里填入你的密钥

# 初始化DeepSeek客户端
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,  # 直接使用上面设置的密钥
    base_url="https://api.deepseek.com/v1"  # DeepSeek API端点
)

aclient = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

# ---------------- 以下代码完全不变 ----------------
def exp(x: float) -> float:
    return math.exp(x)

def safe_softmax_from_topk(logprobs: List[float]) -> List[float]:
    m = max(logprobs)
    exps = [math.exp(li - m) for li in logprobs]
    s = sum(exps) if sum(exps) > 0 else 1.0
    return [e / s for e in exps]

def gen_with_logprobs(prompt: str, max_tokens: int = 64) -> Dict:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=TOP_K
    )
    choice = resp.choices[0]
    tokens, token_logprobs, topk = [], [], []
    for piece in choice.logprobs.content:
        tokens.append(piece.token)
        token_logprobs.append(piece.logprob)
        tk = [(x.token, x.logprob) for x in piece.top_logprobs] if piece.top_logprobs else []
        topk.append(tk)

    last_lp = token_logprobs[-1] if token_logprobs else float("-inf")
    last_p = math.exp(last_lp) if last_lp > -1e9 else 0.0
    joint_logprob = sum(token_logprobs) if token_logprobs else float("-inf")
    avg_logprob = joint_logprob / max(1, len(token_logprobs))
    ppl = math.exp(-avg_logprob) if avg_logprob != float("-inf") else float("inf")

    return {
        "text": choice.message.content,
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "topk": topk,
        "last_token_conf": last_p,
        "joint_logprob": joint_logprob,
        "avg_logprob": avg_logprob,
        "perplexity": ppl
    }

def ensure_single_token_labels(labels: List[str], model: str = MODEL) -> Tuple[List[str], List[bool]]:
    enc = tiktoken.encoding_for_model("gpt-4o") if "gpt-4o" in model else tiktoken.get_encoding("cl100k_base")
    single, flags = [], []
    for lab in labels:
        toks = enc.encode(lab)
        single.append(lab)
        flags.append(len(toks) == 1)
    return single, flags

def classify_with_logprobs(texts: List[str], labels: List[str]) -> List[Dict]:
    system = "You are a strict classifier. Output ONLY one label, exactly matching one of the given labels."
    label_str = ", ".join(labels)
    results = []
    single_labels, mask_single = ensure_single_token_labels(labels, MODEL)

    for x in texts:
        user = f"Labels: [{label_str}]\nText:\n{x}\nAnswer with exactly one label from the list (no punctuation)."
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0,
            max_tokens=4,
            logprobs=True,
            top_logprobs=TOP_K
        )
        content = r.choices[0].message.content.strip()
        pieces = r.choices[0].logprobs.content
        if len(pieces) == 0:
            results.append({"label": content, "conf": 0.0, "dist": {}, "raw": r.model_dump()})
            continue

        topk0 = [(p.token, p.logprob) for p in pieces[0].top_logprobs] if pieces[0].top_logprobs else []
        label_scores = []
        enc = tiktoken.get_encoding("cl100k_base")
        for lab in labels:
            lp = None
            for tk, lpi in topk0:
                if tk == lab:
                    lp = lpi
                    break
            if lp is None and len(pieces) >= 1:
                seq = enc.encode(lab)
                if len(seq) <= len(pieces):
                    s = 0.0
                    ok = True
                    for pos, _ in enumerate(seq):
                        ttk = pieces[pos].top_logprobs or []
                        found = False
                        for cand in ttk:
                            if cand.token == enc.decode([seq[pos]]):
                                s += cand.logprob
                                found = True
                                break
                        if not found:
                            ok = False
                            break
                    if ok:
                        lp = s
            label_scores.append(lp if lp is not None else float("-inf"))

        valid = [li for li in label_scores if li != float("-inf")]
        if len(valid) == 0:
            conf, dist = 0.0, {}
        else:
            probs = safe_softmax_from_topk(label_scores)
            dist = {lab: float(p) for lab, p in zip(labels, probs)}
            conf = dist.get(content, 0.0)
        results.append({"label": content, "conf": float(conf), "dist": dist, "raw": r.model_dump()})
    return results

def expected_calibration_error(preds: List[str], gold: List[str], confs: List[float], n_bins: int = 10) -> float:
    preds = np.array(preds, dtype=object)
    gold = np.array(gold, dtype=object)
    confs = np.array(confs, dtype=float)
    acc = (preds == gold).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        l, r = bins[i], bins[i + 1]
        mask = (confs > l) & (confs <= r) if i > 0 else (confs >= l) & (confs <= r)
        if mask.any():
            ece += mask.mean() * abs(confs[mask].mean() - acc[mask].mean())
    return float(ece)

async def batch_gen(prompts: List[str], max_tokens: int = 64) -> List[Dict]:
    async def one(p):
        r = await aclient.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": p}],
            temperature=TEMPERATURE,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=TOP_K
        )
        c = r.choices[0]
        toks = [t.token for t in c.logprobs.content]
        lps = [t.logprob for t in c.logprobs.content]
        last_p = math.exp(lps[-1]) if lps else 0.0
        avg_lp = sum(lps) / max(1, len(lps)) if lps else float("-inf")
        ppl = math.exp(-avg_lp) if avg_lp != float("-inf") else float("inf")
        return {"text": c.message.content, "last_token_conf": last_p, "avg_logprob": avg_lp, "perplexity": ppl}
    return await asyncio.gather(*[one(p) for p in prompts])

# ---------------- 演示 ----------------
if __name__ == "__main__":
    out = gen_with_logprobs("Answer in one short sentence: What is the capital of France?")
    print("=== Generation ===")
    print("text:", out["text"])
    print("last_token_conf:", round(out["last_token_conf"], 4))
    print("avg_logprob:", round(out["avg_logprob"], 4), "perplexity:", round(out["perplexity"], 3))

    labels = ["sports", "politics", "tech", "art"]
    texts = [
        "NVIDIA launches new GPU for AI training.",
        "The senate passed a new education reform bill.",
        "Striker scores a hat-trick in the final.",
    ]
    cls = classify_with_logprobs(texts, labels)
    print("\n=== Classification ===")
    for i, r in enumerate(cls):
        print(f"sample {i} -> label={r['label']}  conf={round(r['conf'], 4)}  dist={ {k: round(v, 3) for k, v in r['dist'].items()} }")

    gold = ["tech", "politics", "sports"]
    ece = expected_calibration_error([r["label"] for r in cls], gold, [r["conf"] for r in cls], n_bins=10)
    print("\nECE:", round(ece, 4))

    res = asyncio.run(batch_gen(["1+1=?", "Give a color name.", "A prime after 10?"]))
    print("\n=== Async batch ===")
    for r in res:
        print(r)