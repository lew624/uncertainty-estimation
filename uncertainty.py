# uncertainty.py
import re, json, math, random
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# 1) 基础：从 logits 计算不确定性分数
# ---------------------------

def temperature_scale(logits: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    # logits: [B, C], T: 标量张量（>0）
    return logits / T.clamp_min(1e-6)

def msp_confidence(probs: torch.Tensor) -> torch.Tensor:
    # Maximum Softmax Probability 作为置信度
    return probs.max(dim=1).values  # [B]

def entropy_uncertainty(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # 香农熵，值越大越不确定
    ent = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=1)
    # 归一化到 [0,1]（除以 log(C)）
    C = probs.shape[1]
    return ent / math.log(C)

def margin_confidence(probs: torch.Tensor) -> torch.Tensor:
    # top1-top2 的概率差，越大越笃定
    top2 = torch.topk(probs, k=2, dim=1).values
    return (top2[:, 0] - top2[:, 1])  # [B]

def energy_score(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    # Energy-based score: -T * logsumexp(logits/T)
    return -T * torch.logsumexp(logits / T, dim=1)

@torch.no_grad()
def compute_uncertainties_from_logits(
    logits: torch.Tensor, T: float = 1.0
) -> Dict[str, torch.Tensor]:
    # 返回多种度量，便于后续融合
    scaled = temperature_scale(logits, torch.tensor(T, device=logits.device))
    probs = F.softmax(scaled, dim=1)
    return {
        "probs": probs,                                      # [B, C]
        "msp": msp_confidence(probs),                        # [B]
        "entropy": entropy_uncertainty(probs),               # [B] 越大越不确定
        "margin": margin_confidence(probs),                  # [B]
        "energy": energy_score(scaled, T=1.0),               # [B]
        "pred": probs.argmax(dim=1),                         # [B]
    }

# ---------------------------
# 2) 温度缩放（用验证集最小化 NLL）
# ---------------------------

class _Temperature(nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.logT = nn.Parameter(torch.tensor(math.log(init_T)))

    def forward(self):
        return torch.exp(self.logT).clamp_min(1e-6)

def optimize_temperature(logits_val: torch.Tensor, y_val: torch.Tensor, steps: int = 500, lr: float = 0.05) -> float:
    # logits_val: [N, C], y_val: [N] (long)
    device = logits_val.device
    Tm = _Temperature(1.0).to(device)
    opt = torch.optim.LBFGS(Tm.parameters(), lr=lr, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        T = Tm()
        scaled = logits_val / T
        loss = F.cross_entropy(scaled, y_val)
        loss.backward()
        return loss

    opt.step(closure)
    return float(Tm().item())

# ---------------------------
# 3) ECE (Expected Calibration Error)
# ---------------------------

@torch.no_grad()
def expected_calibration_error(probs: torch.Tensor, y_true: torch.Tensor, n_bins: int = 15) -> float:
    # probs: [N, C], y_true: [N]
    conf = probs.max(dim=1).values
    preds = probs.argmax(dim=1)
    acc = (preds == y_true).float()
    bins = torch.linspace(0, 1, steps=n_bins+1, device=probs.device)
    ece = torch.tensor(0.0, device=probs.device)
    N = probs.shape[0]
    for i in range(n_bins):
        l, r = bins[i], bins[i+1]
        mask = (conf > l) & (conf <= r) if i > 0 else (conf >= l) & (conf <= r)
        if mask.any():
            bin_conf = conf[mask].mean()
            bin_acc = acc[mask].mean()
            ece += (mask.float().mean()) * (bin_conf - bin_acc).abs()
    return float(ece.item())

# ---------------------------
# 4) 解析“语言自报置信度”
#    支持 0~1 小数或百分比字符串；也支持标准化 JSON 的 float
# ---------------------------

@torch.no_grad()
def parse_verbal_confidences(verbal_jsons) -> torch.Tensor:
    """
    verbal_jsons: 长度为 B 的列表，每个元素是：
      - 已结构化的 dict/JSON，含键 'confidence'（0~1小数或百分比数字）
      - 或原始字符串，包含诸如 '0.73' 或 '73%' 等
    返回 [B] 张量，范围 [0,1]，无法解析时回落到 0.5
    """
    out = []
    for v in verbal_jsons:
        p = None
        if isinstance(v, dict) and "confidence" in v:
            p = v["confidence"]
        elif isinstance(v, str):
            # 先找小数 0.x 或 1.0
            m = re.search(r"(?<!\d)(0?\.\d+|1(?:\.0+)?)", v)
            if m:
                p = float(m.group(1))
            else:
                # 再找百分数
                m2 = re.search(r"(\d+(?:\.\d+)?)\s*%", v)
                if m2:
                    p = float(m2.group(1)) / 100.0
        elif isinstance(v, (int, float)):
            p = float(v)

        if p is None or math.isnan(p):
            p = 0.5
        p = max(0.0, min(1.0, float(p)))
        out.append(p)
    return torch.tensor(out)

# ---------------------------
# 5) 融合：用逻辑回归把 logits 派生特征 + verbal confidence 做校准
#    目标：预测样本“是否正确”的概率（可作为最终置信度）
# ---------------------------

class FusionCalibrator(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.w = nn.Linear(in_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输出 [B] 概率
        return torch.sigmoid(self.w(x)).squeeze(-1)

def _features_for_fusion(
    logits: torch.Tensor, T: float, verbal_conf: torch.Tensor
) -> torch.Tensor:
    # 组装特征：msp, margin, entropy(取1-entropy为“确定性”), energy, verbal_conf
    with torch.no_grad():
        metrics = compute_uncertainties_from_logits(logits, T)
        probs = metrics["probs"]
        msp = metrics["msp"]
        margin = metrics["margin"]
        certainty = 1.0 - metrics["entropy"]
        energy = metrics["energy"]
    feats = torch.stack([msp, margin, certainty, energy, verbal_conf.to(logits.device)], dim=1)
    return feats  # [B, 5]

def fit_fusion_calibrator(
    logits_val: torch.Tensor, y_val: torch.Tensor, verbal_val: torch.Tensor,
    T: float, epochs: int = 800, lr: float = 0.05, weight_decay: float = 0.0
) -> FusionCalibrator:
    device = logits_val.device
    with torch.no_grad():
        preds = logits_val.argmax(dim=1)
        y_corr = (preds == y_val).float()  # [N] 二分类标签：预测是否正确
    X = _features_for_fusion(logits_val, T, verbal_val)  # [N, 5]
    model = FusionCalibrator(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        p = model(X)
        loss = F.binary_cross_entropy(p, y_corr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model

@torch.no_grad()
def fused_confidence(
    logits: torch.Tensor, verbal_conf: torch.Tensor, T: float, fusion: FusionCalibrator
) -> torch.Tensor:
    X = _features_for_fusion(logits, T, verbal_conf)
    return fusion(X)  # [B]，作为最终校准后的置信度

# ---------------------------
# 6) 多 GPU 并行推理（DataParallel 简易版）
#    你的模型只需实现 forward -> [B, C] logits
# ---------------------------

class DummyClassifier(nn.Module):
    def __init__(self, in_dim: int, n_class: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_class),
        )

    def forward(self, x):
        return self.net(x)

def build_model(in_dim=128, n_class=5):
    model = DummyClassifier(in_dim, n_class)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model.cuda() if torch.cuda.is_available() else model

@torch.no_grad()
def dummy_model_forward(model, x: torch.Tensor) -> torch.Tensor:
    # 替换成你的真实前向；确保输出 [B, C] logits
    model.eval()
    return model(x)

# ---------------------------
# 7) 演示：整合流程
# ---------------------------

def demo():
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B_train, B_val, B_test = 2048, 1024, 1024
    D, C = 128, 7

    # 随机数据与标签（演示用；实际替换为你的数据与模型输出）
    X_val = torch.randn(B_val, D, device=device)
    y_val = torch.randint(0, C, (B_val,), device=device)
    X_test = torch.randn(B_test, D, device=device)
    y_test = torch.randint(0, C, (B_test,), device=device)

    # 构造模型并前向
    model = build_model(D, C)
    logits_val = dummy_model_forward(model, X_val)           # [N_val, C]
    logits_test = dummy_model_forward(model, X_test)         # [N_test, C]

    # 假设我们从 LLM 拿到了结构化 verbal confidence（演示：用随机数模拟）
    # 实际应用中请用上面的 Prompt 让模型只输出 {"confidence": 0.xx} 然后用 parse_verbal_confidences 解析
    verbal_jsons_val = [{"confidence": float(random.random())} for _ in range(B_val)]
    verbal_jsons_test = [{"confidence": float(random.random())} for _ in range(B_test)]
    v_val = parse_verbal_confidences(verbal_jsons_val).to(device)
    v_test = parse_verbal_confidences(verbal_jsons_test).to(device)

    # 1) 温度缩放（先在验证集拟合 T）
    T = optimize_temperature(logits_val, y_val, steps=200, lr=0.1)
    print(f"[Temperature] T = {T:.3f}")

    # 2) 评估温度缩放前后 ECE（仅作参考）
    with torch.no_grad():
        probs_val_raw = F.softmax(logits_val, dim=1)
        probs_val_cal = F.softmax(logits_val / T, dim=1)
        ece_raw = expected_calibration_error(probs_val_raw, y_val)
        ece_cal = expected_calibration_error(probs_val_cal, y_val)
    print(f"[ECE] before={ece_raw:.4f}, after(T)={ece_cal:.4f}")

    # 3) 训练融合校准器：把 logits 的多个度量 + verbal_conf 一起学成“预测是否正确”的概率
    fusion = fit_fusion_calibrator(logits_val, y_val, v_val, T, epochs=400, lr=0.05)

    # 4) 在测试集上输出多种不确定性 & 融合后的最终置信度
    with torch.no_grad():
        metrics = compute_uncertainties_from_logits(logits_test, T)
        probs_test = metrics["probs"]
        msp = metrics["msp"]
        margin = metrics["margin"]
        entropy = metrics["entropy"]
        energy = metrics["energy"]
        pred = metrics["pred"]
        fused = fused_confidence(logits_test, v_test, T, fusion)  # 我们主推的最终置信度

        # 粗略看下和真实正确性的相关性
        correct = (pred == y_test).float()
        corr_msp = float(torch.corrcoef(torch.stack([msp, correct]))[0,1])
        corr_fused = float(torch.corrcoef(torch.stack([fused, correct]))[0,1])
        print(f"[Corr with correctness] MSP={corr_msp:.3f}, FUSED={corr_fused:.3f}")

        # 示例：把高不确定样本（低 fused）筛出来
        idx_uncertain = torch.topk(-fused, k=10).indices
        print("[Top-10 uncertain indices]", idx_uncertain[:10].tolist())
        print("Sample fused confidences:", fused[idx_uncertain[:5]].tolist())

if __name__ == "__main__":
    demo()
