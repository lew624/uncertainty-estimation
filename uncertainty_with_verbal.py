import torch, math
from uncertainty import (
    compute_uncertainties_from_logits,
    parse_verbal_confidences,
    _features_for_fusion,
    fused_confidence,
    FusionCalibrator
)

# 1. 构造一条 logits（可以反推任意满足 softmax 后 [0.90,0.07,0.03] 的向量）
#    简单写法：取 ln 即可
probs_ref = torch.tensor([0.90, 0.07, 0.03])
logits    = torch.log(probs_ref).unsqueeze(0)          # shape [1,3]

# 2. 口头自报置信度（LLM 直接说 90%）
verbal_json = [{"confidence": 0.90}]
verbal_conf = parse_verbal_confidences(verbal_json)    # shape [1]

# 3. 温度缩放：这里只有一条样本，直接 T=1 即可
T = 1.0

# 4. 计算各种不确定性指标
metrics = compute_uncertainties_from_logits(logits, T)
print("prob        :", metrics["probs"].tolist())
print("MSP         :", metrics["msp"].item())
print("Margin      :", metrics["margin"].item())
print("Entropy     :", metrics["entropy"].item())   # 越大越不确定
print("Energy      :", metrics["energy"].item())
print("pred class  :", metrics["pred"].item())      # 0 对应 A

# 5. 融合校准（演示用，我们随机造一个已训练好的 fusion 模型）
#    实际使用时用 fit_fusion_calibrator 在验证集训练
feats = _features_for_fusion(logits, T, verbal_conf) # [1,5]
print("fusion feats:", feats.tolist())

# 这里假装我们已经训练好一个融合器，权重全 1、偏置 0，方便看数值
fusion = FusionCalibrator(5)
with torch.no_grad():
    fusion.w.weight.fill_(1.0)
    fusion.w.bias.fill_(0.0)
    fused = fused_confidence(logits, verbal_conf, T, fusion)
print("fused conf  :", fused.item())