地址：https://juejin.cn/post/7249233583146450981

#### 前言

聊聊transformers库——[基础与入门](https://juejin.cn/post/7248918622692769851)

聊聊transformers库——进阶-[模型微调和保存](https://juejin.cn/post/7249233583146450981)

transformers库进阶之——[使用自定义数据集来训练和预测](https://juejin.cn/spost/7249286405023825977)

#### transformers库进阶之——对模型进行微调与保存

微调（Fine-tuning）是一种迁移学习技术，通过在预训练模型的基础上进行少量训练，使模型适应新任务或新数据集。

在本节中，我们将介绍如何使用transformers库对模型进行微调，并将微调后的模型保存到本地。

以文本分类任务为例，我们将使用BERT模型进行情感分析。

###### 首先，加载预训练的BERT模型和tokenizer

```python
python 代码解读复制代码# 导入所需库
from transformers import BertTokenizer, BertForSequenceClassification

# 从预训练模型中加载 BERT 分词器和分类器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
```

这段代码的目的是从预训练模型中加载 BERT 分词器和分类器。

我们使用 transformers 库，它是一个用于自然语言处理（NLP）任务的库，包含了许多预训练模型，如 BERT。

下面是详细的代码解释：

1. 导入所需库：我们需要从 transformers 库中导入 BertTokenizer（用于文本处理）和 BertForSequenceClassification（用于文本分类）。这些类将帮助我们处理输入文本并对其进行分类。
2. 加载预训练模型：
   - `tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")`：我们使用 `from_pretrained` 方法加载预训练的 BERT 分词器。我们将使用 "bert-base-uncased" 模型，表示使用小写字母的基本 BERT 模型。分词器将帮助我们将原始文本转换为模型可以理解的格式。
   - `model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)`：我们同样使用 `from_pretrained` 方法加载预训练的 BERT 分类器。我们指定 `num_labels=2`，表示我们的分类任务有两个类别（正面和负面）。这个分类器将用于对输入文本进行情感分类。

###### 准备训练数据

```python
python 代码解读复制代码# 导入 PyTorch 库
import torch
from torch.utils.data import DataLoader, TensorDataset

# 创建训练文本和标签
train_texts = ["I love this movie!", "This movie is terrible.", "The plot is great.", "The acting is bad."]
train_labels = [1, 0, 1, 0]

# 使用 BERT 分词器对文本进行编码，并��结果存储为 PyTorch 张量
encoded_inputs = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")

# 创建训练数据集
train_dataset = TensorDataset(encoded_inputs["input_ids"], encoded_inputs["attention_mask"], torch.tensor(train_labels))

# 创建数据加载器，用于将训练数据分批输入模型
train_dataloader = DataLoader(train_dataset, batch_size=2)
```

这段代码的目的是使用 PyTorch 库创建训练数据集并将其分批输入模型。

我们首先导入 PyTorch 库，然后创建训练文本和标签。

接着，我们使用 BERT 分词器对文本进行编码，并将结果存储为 PyTorch 张量。

最后，我们创建训练数据集和数据加载器。

以下是详细的代码解释：

1. 导入 PyTorch 库：我们需要 PyTorch 库来处理数据和训练模型。我们导入了 DataLoader 和 TensorDataset，它们帮助我们将数据分批输入模型。
2. 创建训练文本和标签：我们定义了一组训练文本和相应的标签。标签为 1 表示正面，0 表示负面。
3. 使用 BERT 分词器对文本进行编码：我们使用分词器将文本转换为模型可以理解的格式。`padding=True` 表示我们要对输入进行填充以使它们具有相同的长度，`truncation=True` 表示如果输入太长，我们将对其进行截断。`return_tensors="pt"` 表示我们希望以 PyTorch 张量的形式返回编码后的结果。
4. 创建训练数据集：我们使用 TensorDataset 将编码后的输入数据和标签组合成一个数据集。TensorDataset 接受一系列张量作为输入，并将它们组合在一起。在这里，我们将输入 ID、注意力掩码和标签张量组合在一起。
5. 创建数据加载器：我们使用 DataLoader 将训练数据集分批输入模型。`batch_size=2` 表示每次输入模型的文本数量为 2。数据加载器会自动处理数据的分批和洗牌。

###### 设置优化器和损失函数：

```python
python 代码解读复制代码# 导入优化器和损失函数
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

# 初始化优化器和损失函数
optimizer = Adam(model.parameters(), lr=1e-5)
loss_fn = CrossEntropyLoss()
```

这段代码的目的是初始化一个优化器和一个损失函数，这两个组件在训练神经网络时非常重要。

1. 首先，从`torch.optim`模块中导入`Adam`优化器。优化器的作用是在训练过程中更新模型的参数（权重和偏置），以便最小化损失函数。`Adam`是一种非常流行的优化器，因为它通常在许多任务中表现良好，并且具有自适应学习率。
2. 然后，从`torch.nn`模块中导入`CrossEntropyLoss`损失函数。损失函数用于衡量模型在训练数据上的表现。`CrossEntropyLoss`是一种常用于分类任务的损失函数，它衡量模型预测的概率分布与真实标签之间的差距。在训练过程中，我们的目标是最小化这个损失值，以便提高模型在训练数据上的准确性。
3. 接下来，我们使用`Adam`优化器并将其实例化。在这个过程中，我们需要传递模型的参数（`model.parameters()`）以及学习率（`lr=1e-5`）。模型的参数是神经网络中的权重和偏置，这些参数在训练过程中需要不断更新以优化模型的性能。学习率是一个超参数，它决定了在优化过程中参数更新的幅度。较大的学习率可能导致参数更新过快，而较小的学习率可能导致训练过程缓慢。在这个例子中，学习率被设置为1e-5（0.00001），这是一个相对较小的值，通常可以帮助模型更稳定地收敛。
4. 最后，我们实例化`CrossEntropyLoss`损失函数，将其赋值给变量`loss_fn`。在训练神经网络时，我们将使用这个损失函数来计算模型在每个批次的训练数据上的损失值。优化器会根据这个损失值来更新模型的参数。

###### 进行模型的微调

```python
python 代码解读复制代码# 设置训练轮数
epochs = 3

# 进行多轮训练
for epoch in range(epochs):
    total_loss = 0
    # 对每个批次进行训练
    for batch in train_dataloader:
        # 清空优化器的梯度
        optimizer.zero_grad()
        # 从批次中提取输入数据和标签
        input_ids, attention_mask, labels = batch
        # 将数据输入模型并获取输出
        outputs = model(input_ids, attention_mask=attention_mask)
        # 计算损失
        loss = loss_fn(outputs.logits, labels)
        # 反向传播损失
        loss.backward()
        # 更新模型参数
        optimizer.step()
        # 累计损失
        total_loss += loss.item()
        # 打印每轮训练的损失
        print(f"Epoch: {epoch}, Loss: {total_loss}")
```

这段代码的主要目的是对训练数据进行多轮迭代，每轮迭代中，模型会根据损失值更新参数以提高性能。以下是对代码的详细解释：

1. 首先，我们设置训练轮数（`epochs`）为3。这意味着我们将对整个训练数据集进行3次完整的迭代。每轮迭代后，模型的性能通常会有所提高。
2. 接下来，我们使用一个`for`循环来迭代`epochs`。在每个`epoch`中，我们将对训练数据集进行一次完整的遍历。
3. 在每个`epoch`中，我们初始化一个变量`total_loss`来累计损失值。这将用于跟踪每轮训练的总损失。
4. 接着，我们使用另一个`for`循环来遍历`train_dataloader`中的每个批次。`train_dataloader`是一个用于加载训练数据的迭代器，它将数据分成批次以便进行小批量梯度下降。
5. 在每个批次的训练中，我们首先使用`optimizer.zero_grad()`清空优化器的梯度。这是因为在每次迭代中，梯度会累积，所以我们需要在每个批次开始时清空梯度。
6. 然后，我们从当前批次中提取输入数据（`input_ids`）、注意力掩码（`attention_mask`）和标签（`labels`）。这些数据将用于训练模型。
7. 接下来，我们将输入数据和注意力掩码传递给模型，并获取输出结果。`outputs`是一个包含模型预测结果的对象。
8. 使用损失函数`loss_fn`计算模型输出的`logits`（预测概率分布）与真实`labels`之间的损失。损失值表示模型在当前批次上的表现。
9. 为了优化模型参数，我们对损失值进行反向传播。这是通过调用`loss.backward()`实现的。这一步会计算损失值关于模型参数的梯度。
10. 使用`optimizer.step()`更新模型参数。这一步根据梯度值和学习率调整模型的权重和偏置。
11. 累计当前批次的损失值到`total_loss`。
12. 在每轮训练结束后，我们打印出当前`epoch`和累计的`total_loss`。这有助于我们监控训练过程中模型的性能。

###### 使用模型和tokenizer保存到本地

```python
python 代码解读复制代码# 保存微调后的模型和分词器
model.save_pretrained("my_finetuned_model")
tokenizer.save_pretrained("my_finetuned_model")
```

这样，你就可以在其他项目中加载微调后的模型和tokenizer，例如：

```python
python 代码解读复制代码# 从保存的文件中加载微调后的模型和分词器
model = BertForSequenceClassification.from_pretrained("my_finetuned_model")
tokenizer = BertTokenizer.from_pretrained("my_finetuned_model")
```

上面的代码的目的是保存训练好的神经网络模型（在这里是一个微调过的BERT模型）和分词器，以便将来可以重新加载它们并在其他任务中使用。

1. 首先，我们使用`model.save_pretrained("my_finetuned_model")`将训练好的模型保存到本地文件系统。这里的`"my_finetuned_model"`是一个文件夹名称，用于存储模型的权重和配置文件。`save_pretrained`方法会将模型的权重和配置信息保存为二进制文件，以便稍后重新加载。
2. 接下来，我们使用`tokenizer.save_pretrained("my_finetuned_model")`将分词器（用于将文本转换为模型可以处理的形式的工具）保存到同一个文件夹。分词器的词汇表和配置信息也将保存为文件，以便将来重新加载。
3. 在保存了模型和分词器之后，我们可以使用`BertForSequenceClassification.from_pretrained("my_finetuned_model")`从刚刚保存的文件夹中重新加载模型。`from_pretrained`方法会读取权重和配置文件，并使用它们实例化一个新的模型对象。这个新的模型对象将具有与之前训练好的模型相同的权重和配置。
4. 类似地，我们使用`BertTokenizer.from_pretrained("my_finetuned_model")`从文件夹中重新加载分词器。这将返回一个新的分词器对象，与之前使用的分词器具有相同的词汇表和配置。

###### 总结

总结一下，上述代码的主要目的是使用预训练的 BERT 模型对给定的文本进行情感分类。

我们首先加载预训练模型和分词器，然后使用它们处理训练数据。

接着，我们使用 PyTorch 库设置优化器、损失函数和训练轮数，并进行多轮训练。

最后，我们保存和加载微调后的模型和分词器，以便在以后的任务中使用。

下面是详细的代码解释：

1. 导入所需库：我们需要从 transformers 库中导入 BertTokenizer（用于文本处理）和 BertForSequenceClassification（用于文本分类）。
2. 加载预训练模型：我们使用 `from_pretrained` 方法加载预训练的 BERT 分词器和分类器。我们将使用 "bert-base-uncased" 模型，表示使用小写字母的基本 BERT 模型。我们还指定 `num_labels=2`，表示我们的分类任务有两个类别（正面和负面）。
3. 导入 PyTorch 库：我们需要 PyTorch 库来处理数据和训练模型。我们导入了 DataLoader 和 TensorDataset，它们帮助我们将数据分批输入模型。
4. 创建训练文本和标签：我们定义了一组训练文本和相应的标签。标签为 1 表示正面，0 表示负面。
5. 使用 BERT 分词器对文本进行编码：我们使用分词器将文本转换为模型可以理解的格式。`padding=True` 表示我们要对输入进行填充以使它们具有相同的长度，`truncation=True` 表示如果输入太长，我们将对其进行截断。`return_tensors="pt"` 表示我们希望以 PyTorch 张量的形式返回编码后的结果。
6. 创建训练数据集：我们使用 TensorDataset 将编码后的输入数据和标签组合成一个数据集。
7. 创建数据加载器：我们使用 DataLoader 将训练数据集分批输入模型。`batch_size=2` 表示每次输入模型的文本数量为 2。
8. 导入优化器和损失函数：我们需要一个优化器（如 Adam）来更新模型参数，以及一个损失函数（如交叉熵损失）来衡量模型在训练过程中的表现。
9. 初始化优化器和损失函数：我们使用模型参数和学习率初始化 Adam 优化器，并初始化交叉熵损失函数。
10. 设置训练轮数：我们将对整个训练数据集进行 3 轮训练。
11. 进行多轮训练：对于每个训练轮，我们遍历数据加载器中的每个批次，将输入数据和标签输入模型，计算损失，执行反向传播，并更新模型参数。我们还计算每轮训练的总损失。
12. 保存微调后的模型和分词器：在训练完成后，我们将微调后的模型和分词器保存到文件中。
13. 从保存的文件中加载微调后的模型和分词器：为了在以后的任务中使用微调后的模型和分词器，我们可以从保存的文件中加载它们。