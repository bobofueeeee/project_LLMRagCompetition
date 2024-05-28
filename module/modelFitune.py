from transformers import BertTokenizer, BertForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from your_dataset_loader import load_and_cache_examples  # 假设你有一个加载数据集的函数

# 加载预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# 加载并缓存数据
train_dataset = load_and_cache_examples('train_data.json', tokenizer)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=16)

# 定义优化器和调度器
optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
epochs = 3
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * epochs)

# 微调模型
model.train()
for epoch in range(epochs):
    for batch in train_dataloader:
        # 假设batch包含input_ids, attention_mask, start_positions, end_positions
        inputs = {'input_ids': batch['input_ids'].to(device),
                  'attention_mask': batch['attention_mask'].to(device)}
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs.loss  # 模型已经内置了计算损失的逻辑
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # 保存微调后的模型
model.save_pretrained('finetuned_bert_qa')