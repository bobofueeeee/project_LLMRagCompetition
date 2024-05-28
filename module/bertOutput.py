

def model_output(question,context):

    return answer





question = "你叫什么名字"
context = "我叫付博"


from transformers import BertModel, BertTokenizer
import torch
tokenizer = BertTokenizer.from_pretrained(r"D:\wk\bert-base-uncased")
model = BertModel.from_pretrained(r"D:\wk\bert-base-uncased")

# question = mostSimilary['question']
# context = mostSimilary['answer']

input_dict = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
print(input_dict["input_ids"])
print(input_dict["input_ids"][0, :])

# 提取答案文本
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(input_dict["input_ids"][0, :]))

# 输出答案
print(f"Question: {question}")
print(f"Answer: {answer}")