from transformers import BertTokenizer, BertModel
import torch
from torch.nn.functional import cosine_similarity

# 加载预训练的BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained(r"D:\wk\bert-base-uncased")
model = BertModel.from_pretrained(r"D:\wk\bert-base-uncased")

# 准备要比较的句子
sentence1 = "你叫什么名字!"
sentence2 = "你叫什么名字!"

# 使用tokenizer将句子转换为BERT的输入格式
inputs1 = tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True)
inputs2 = tokenizer(sentence2, return_tensors="pt", padding=True, truncation=True)

# 获取句子的BERT表示
with torch.no_grad():
    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)

# 获取[CLS] token的嵌入，它通常用作整个句子的表示
print(type(outputs1))

print(type(outputs1.last_hidden_state[:, 0, :]))
print(type(outputs1.last_hidden_state))
last_hidden_states1 = outputs1.last_hidden_state[:, 0, :]  # 取第一个token（通常是[CLS]）的嵌入
last_hidden_states2 = outputs2.last_hidden_state[:, 0, :]  # 同样地，取第二个句子的[CLS] token的嵌入

# 计算两个嵌入之间的余弦相似度
cosine_sim = cosine_similarity(last_hidden_states1.unsqueeze(0), last_hidden_states2.unsqueeze(0))
# 输出相似度分数
print(f"The cosine similarity between the two sentences is: {cosine_sim.mean().item()}")