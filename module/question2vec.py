# 提问向量化

question = "你叫什么名字"

from transformers import BertModel, BertTokenizer
# 加载模型和分词器
tokenizer = BertTokenizer.from_pretrained(r"D:\wk\bert-base-uncased")
inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
# print(inputs)

