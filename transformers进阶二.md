地址：https://juejin.cn/post/7249286405023825977

#### 前言

聊聊transformers库——[基础与入门](https://juejin.cn/post/7248918622692769851)

聊聊transformers库——进阶-[模型微调和保存](https://juejin.cn/post/7249233583146450981)

transformers库进阶之——[使用自定义数据集来训练和预测](https://juejin.cn/spost/7249286405023825977)

#### transformers库进阶之——使用自定义数据集来训练和预测

在实际应用中，我们通常需要处理自定义的数据集。

为了方便地使用transformers库处理这些数据，我们可以继承`Dataset`类来实现自定义的数据集类。

以下是一个简单的例子：

```python
python 代码解读复制代码import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

# 自定义数据集类
class CustomTextDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=None):
        self.tokenizer = tokenizer  # 传入tokenizer对象
        self.texts = []
        self.labels = []
        self.max_length = max_length  # 设置文本的最大长度

        # 从文件中读取文本和标签
        with open(os.path.join(data_dir, "texts.txt"), "r", encoding="utf-8") as f:
            self.texts = [line.strip() for line in f.readlines()]

        with open(os.path.join(data_dir, "labels.txt"), "r", encoding="utf-8") as f:
            self.labels = [int(line.strip()) for line in f.readlines()]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # 使用tokenizer对文本进行编码，并设置max_length和padding参数以执行填充操作
        encoded_input = self.tokenizer.encode_plus(
            text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        # 返回编码结果和对应的标签
        return {key: tensor.squeeze(0) for key, tensor in encoded_input.items()}, torch.tensor(label)
```

这个例子中，我们假设数据集存储在一个文件夹中，包含两个文件：`texts.txt`和`labels.txt`。

`texts.txt`中的每一行是一个文本，`labels.txt`中的每一行是对应的标签。

我们使用tokenizer对文本进行编码，并将编码结果与标签一起返回。

`texts.txt` 里的内容示例：

```html
html 代码解读复制代码I love this movie!
This movie is terrible.
The plot is great.
The acting is bad.
```

`labels.txt` 里的内容示例：

```
 代码解读复制代码1
0
1
0
```

接下来可以使用这个自定义数据集来进行训练。

示例代码如下：

```python
python 代码解读复制代码# 指定数据集路径，需要根据实际情况设置正确的目录，里面包括上述提到的 texts.txt 和 labels.txt
data_dir = "/your/path/of/AA_data"

# 加载预训练的BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 创建自定义数据集对象
dataset = CustomTextDataset(data_dir, tokenizer)
print(f"len of dataset: {len(dataset)}")
for key in dataset[0][0].keys():
    print(key, dataset[0][0][key].size())

# 创建数据加载器，用于批量处理数据
train_dataloader = DataLoader(dataset, batch_size=8)

# 设置优化器
optimizer = Adam(model.parameters(), lr=1e-5)

# 设置损失函数
loss_fn = CrossEntropyLoss()

# 微调模型
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for batch_inputs, batch_labels in train_dataloader:
        optimizer.zero_grad()  # 重置优化器的梯度
        input_ids, attention_mask = batch_inputs["input_ids"], batch_inputs["attention_mask"]
        labels = batch_labels
        outputs = model(input_ids, attention_mask=attention_mask)  # 将输入数据传递给模型
        loss = loss_fn(outputs.logits, labels)  # 计算损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
        total_loss += loss.item()
    print(f"Epoch: {epoch}, Loss: {total_loss}")
```

对模型进行微调后，可以用来做预测：

```python
python 代码解读复制代码# 测试模型
test_texts = ["I can't wait to watch this movie again!", "The movie was not what I expected."]
encoded_inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():  # 禁用梯度计算
    outputs = model(**encoded_inputs)  # 将输入数据传递给模型
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)  # 计算预测结果

print("Predictions:", predictions)
```

为方便大家理解，下面再对整个代码进行详细的解释。

首先，我们定义了一个名为`CustomTextDataset`的自定义数据集类，它继承自`torch.utils.data.Dataset`。这个类用于从文件中读取文本和标签数据，并使用tokenizer对文本进行编码。

- 在`__init__`方法中，我们传入了数据集路径、tokenizer对象和文本的最大长度。然后，我们从文件中读取文本和标签数据，并将它们存储在类的属性中。
- 在`__len__`方法中，我们返回数据集中文本的数量。
- 在`__getitem__`方法中，我们根据给定的索引获取文本和标签。然后，我们使用tokenizer对文本进行编码，并设置`max_length`和`padding`参数以执行填充操作。最后，我们将编码结果和对应的标签返回。

接下来，我们指定了数据集路径，并加载了预训练的BERT模型和tokenizer。然后，我们创建了一个自定义数据集对象，并输出了数据集的长度和第一个元素。

之后，我们创建了一个数据加载器，用于批量处理数据。数据加载器将自动处理数据的抽取、打乱和分批等操作。

接着，我们设置了优化器和损失函数。优化器用于更新模型的参数，而损失函数用于计算模型的预测结果与真实标签之间的差异。

在进行模型微调时，我们遍历了数据加载器中的每个批次，并执行了以下操作：

1. 使用`optimizer.zero_grad()`重置优化器的梯度。
2. 从`batch_inputs`中获取`input_ids`和`attention_mask`。
3. 将输入数据传递给模型，得到输出结果。
4. 使用损失函数计算模型输出与真实标签之间的差异。
5. 使用`loss.backward()`进行反向传播计算梯度。
6. 使用`optimizer.step()`更新模型参数。
7. 将损失值累加到`total_loss`中。

在每个epoch结束时��我们输出了累计的损失值。

最后，我们使用微调后的模型对测试文本进行情感分析。

我们首先使用tokenizer对测试文本进行编码，然后将编码结果传递给模型。

模型输出的logits表示每个类别的预测分数。

我们使用`torch.argmax`函数计算具有最高分数的类别，并将其作为预测结果。

最后，我们输出了预测结果，如下：

> Predictions: tensor([1, 0])

这个示例展示了如何使用transformers库和自定义数据集类对模型进行微调和测试。

在实际应用中，你可以根据自己的任务和数据集调整相关参数和设置，实现定制化的训练过程。

