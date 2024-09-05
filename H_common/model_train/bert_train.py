import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig

# 设定随机种子以便复现结果
torch.manual_seed(42)

# 输入文本
input_text = "你好，我叫付博"

# 准备tokenizer和模型
tokenizer = BertTokenizer.from_pretrained(r'D:\wk\model\bert-base-chinese')
model = BertForMaskedLM.from_pretrained(r'D:\wk\model\bert-base-chinese')

# 使用tokenizer对文本进行编码，包括加入特殊token [CLS] 和 [SEP]
tokenized_text = tokenizer.tokenize(input_text)
tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']

# 找到"我叫付博"的起始和结束位置
start_idx = tokenized_text.index("我")
end_idx = tokenized_text.index("博") + 1

# 将"我叫付博"部分mask掉
masked_text = tokenized_text[:]
for i in range(start_idx, end_idx):
    masked_text[i] = '[MASK]'

# 将token转换为input_ids
input_ids = tokenizer.convert_tokens_to_ids(masked_text)
tokens_tensor = torch.tensor([input_ids])

# 创建labels，labels中包含被mask掉的部分的真实token id
labels = input_ids[:]
for i in range(start_idx, end_idx):
    labels[i] = tokenizer.convert_tokens_to_ids(tokenized_text[i])

labels_tensor = torch.tensor([labels])

# 准备模型输入
inputs = {
    'input_ids': tokens_tensor,
    'labels': labels_tensor
}

# 模型训练设置
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# 模型训练
model.train()
for epoch in range(10):  # 进行10个epoch的训练
    optimizer.zero_grad()
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 测试推理
model.eval()
with torch.no_grad():
    outputs = model(tokens_tensor)

# 打印输出的top k预测结果
masked_index = masked_text.index("[MASK]") + 3
predicted_index = torch.argmax(outputs.logits[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(f'Predicted token: {predicted_token}')
