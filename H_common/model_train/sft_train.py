from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 加载预训练模型和分词器
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 假设是二分类任务

# 加载数据集（这里以GLUE中的MRPC为例，它是一个二分类任务）
dataset = load_dataset('glue', 'mrpc')


# 准备数据，将其转换为适合模型训练的格式
def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 划分训练集和验证集
train_dataset = tokenized_datasets["train"].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
eval_dataset = tokenized_datasets["validation"].set_format(type='torch',
                                                           columns=['input_ids', 'attention_mask', 'label'])

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',  # 输出目录
    num_train_epochs=3,  # 训练轮次
    per_device_train_batch_size=16,  # 每个设备的批处理大小
    per_device_eval_batch_size=64,  # 每个设备的评估批处理大小
    warmup_steps=500,  # 预热步数
    weight_decay=0.01,  # 权重衰减
    logging_dir='./logs',  # 日志目录
    logging_steps=10,
)

# 初始化Trainer
trainer = Trainer(
    model=model,  # 模型
    args=training_args,  # 训练参数
    train_dataset=train_dataset,  # 训练数据集
    eval_dataset=eval_dataset  # 评估数据集
)

# 开始训练
trainer.train()