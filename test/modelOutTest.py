from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased-finetuned-squad')  # 使用基于SQuAD数据集的预训练模型
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased-finetuned-squad')

# 定义问题和上下文
question = "What is the capital of France?"
context = "Paris is the capital and most populous city of France."

# 对问题和上下文进行编码
input_dict = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")

# 预测答案的起始位置和结束位置
with torch.no_grad():
    outputs = model(**input_dict)

# 获取预测结果
answer_start_scores = outputs.start_logits
answer_end_scores = outputs.end_logits

# 获取最佳答案的起始和结束位置索引
answer_start = torch.argmax(answer_start_scores)
answer_end = torch.argmax(answer_end_scores[0, answer_start:]) + answer_start

# 提取答案文本
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(input_dict["input_ids"][0, answer_start:answer_end + 1]))

# 输出答案
print(f"Question: {question}")
print(f"Answer: {answer}")