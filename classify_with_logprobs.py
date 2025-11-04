from deepseek_logprobs_confidence import classify_with_logprobs

labels = ["A", "B", "C"]
texts = ["10 --> 90%  3. --> 7%  2. --> 3%  请选出最大概率对应的选项字母（A/B/C）"]

result = classify_with_logprobs(texts, labels)[0]
print("模型选择：", result["label"])
print("置信度：", round(result["conf"], 4))
print("分布：", {k: round(v, 3) for k, v in result["dist"].items()})