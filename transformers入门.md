地址：https://juejin.cn/post/7248918622692769851

#### 写在前面

聊聊transformers库——[基础与入门](https://juejin.cn/post/7248918622692769851)

聊聊transformers库——进阶-[模型微调和保存](https://juejin.cn/post/7249233583146450981)

transformers库进阶之——[使用自定义数据集来训练和预测](https://juejin.cn/spost/7249286405023825977)

#### 一、引言

###### 1、介绍transformers库的背景和重要性

在过去的几年里，自然语言处理（NLP）领域取得了巨大的进步。

这些进步主要归功于深度学习技术的发展，尤其是基于Transformer结构的预训练模型。

这些模型，如BERT、GPT-2、RoBERTa等，已经在各种NLP任务中取得了显著的成果，为我们提供了更高质量的文本处理能力。

为了让研究人员和开发者更方便地使用这些先进的模型，Hugging Face团队开发了一个名为transformers的开源库。

transformers库为我们提供了丰富的预训练模型、简洁的API接口和高效的性能，使得我们可以轻松地将这些模型应用到实际项目中。

因此，了解和掌握transformers库对于NLP从业者来说具有重要的意义。

###### 2、简要概述transformers库的功能和优势

transformers库的主要功能包括：

- 提供丰富的预训练模型，涵盖了目前主流的NLP任务，如文本分类、生成、摘要、问答等。
- 提供简洁的API接口，使得用户可以快速上手，无需关注模型的底层实现细节。
- 支持多种深度学习框架，如PyTorch和TensorFlow，方便用户根据自己的需求进行选择。
- 提供高效的性能，支持多GPU和分布式训练，满足大规模数据处理的需求。

transformers库的主要优势包括：

- 开源且持续更新，用户可以随时获取到最新的模型和功能。
- 社区活跃，有大量的教程和案例供用户参考，方便用户学习和交流。
- 模块化设计，用户可以根据需求灵活组合模型的各个部分，实现定制化的应用。

#### 二、基础知识

###### 1、自然语言处理（NLP）简介

自然语言处理（Natural Language Processing，简称NLP）是计算机科学、人工智能和语言学领域的交叉学科，主要研究如何让计算机能够理解、生成和处理人类语言。

NLP的主要任务包括文本分类、命名实体识别、情感分析、机器翻译、文本摘要、问答系统等。

随着科技的发展，NLP技术已经广泛应用于搜索引擎、智能客服、社交媒体分析等领域，对人们的生活产生了深远的影响。

###### 2、深度学习在NLP中的应用

深度学习是一种特殊的机器学习方法，通过构建多层神经网络模型来学习数据的表征和特征。

近年来，深度学习在NLP领域取得了显著的成果，推动了各种任务的性能不断提高。

深度学习在NLP中的应用主要包括以下几个方面：

- 词嵌入（Word Embedding）：将词汇映射到连续的向量空间，使得词汇之间的相似性可以通过向量的距离来度量。常见的词嵌入方法有Word2Vec、GloVe等。
- 循环神经网络（Recurrent Neural Network，简称RNN）：一种处理序列数据的神经网络结构，可以捕捉文本中的长距离依赖关系。常见的RNN变种有LSTM（长短时记忆网络）和GRU（门控循环单元）。
- Transformer模型：一种基于自注意力（Self-Attention）机制的神经网络结构，可以并行处理序列数据，具有更高的计算效率。Transformer模型已经成为NLP领域的主流方法。

###### 3、Transformer模型的原理与结构

Transformer模型最早由Vaswani等人在2017年的论文《Attention is All You Need》中提出。

该模型摒弃了传统的RNN结构，完全基于自注意力机制来处理序列数据。

Transformer模型的主要组成部分有：

- 自注意力机制（Self-Attention）：计算序列中每个词汇与其他词汇之间的关联程度，用于捕捉上下文信息。
- 多头注意力（Multi-Head Attention）：将自注意力机制并行化，可以同时学习多种不同的上下文关系。
- 位置编码（Positional Encoding）：将词汇在序列中的位置信息编码成向量，以弥补自注意力机制无法捕捉位置信息的缺陷。
- 前馈神经网络（Feed-Forward Neural Network）：用于提取词汇的局部特征。
- 残差连接（Residual Connection）和层归一化（Layer Normalization）：用于稳定模型的训练过程。

###### 4、常见的预训练模型（如BERT、GPT-2、RoBERTa等）

预训练模型是一种将大量无标签数据通过无监督学习方法进行预训练，然后在具体任务上进行微调的技术。

预训练模型可以有效地利用海量数据的信息，提高模型的泛化能力。

目前，基于Transformer结构的预训练模型已经成为NLP领域的主流方法。以下是一些常见的预训练模型：

- BERT（Bidirectional Encoder Representations from Transformers）：由Google在2018年提出的预训练模型。BERT采用双向Transformer编码器结构，通过掩码语言模型（Masked Language Model，MLM）和下一个句子预测（Next Sentence Prediction，NSP）两种任务进行预训练。BERT在多种NLP任务上取得了显著的成果，成为了许多后续模型的基础。
- GPT-2（Generative Pre-trained Transformer 2）：由OpenAI在2019年提出的预训练模型。GPT-2采用单向Transformer解码器结构，通过自回归（Autoregressive）语言模型进行预训练。GPT-2在文本生成任务上表现出色，引发了业界对生成模型潜在风险的关注。
- RoBERTa（Robustly Optimized BERT Pretraining Approach）：由Facebook AI在2019年提出的预训练模型。RoBERTa对BERT进行了改进，主要包括去除NSP任务、增加训练数据、调整训练策略等。RoBERTa在多种NLP任务上进一步提高了性能。
- T5（Text-to-Text Transfer Transformer）：由Google在2019年提出的预训练模型。T5将所有NLP任务统一为文本到文本的生成任务，采用端到端的Transformer结构进行预训练和微调。T5在多种任务上表现优��，展示了文本生成方法的强大潜力。
- ELECTRA（Efficiently Learning an Encoder that Classifies Token Replacements Accurately）：由Google在2020年提出的预训练模型。ELECTRA采用一种新颖的预训练任务——替换令牌检测（Replaced Token Detection），在计算效率和性能上取得了显著的改进。

这些预训练模型为NLP领域的研究和应用提供了强大的基础，transformers库则为我们提供了方便的工具，使得我们可以轻松地使用这些模型来解决实际问题。

在接下来的部分中，我们将介绍如何使用transformers库进行模型的加载、训练和微调等操作。

#### 三、环境配置与安装

###### 1、Python环境准备

在开始使用transformers库之前，我们需要确保已经安装了合适版本的Python环境。

建议使用Python 3.6及以上版本，以确保与transformers库的兼容性。

你可以通过以下命令检查当前的Python版本：

```shell
shell

 代码解读
复制代码python --version
```

如果你还没有安装Python，可以访问[Python官网](https://link.juejin.cn?target=https%3A%2F%2Fwww.python.org%2Fdownloads%2F)下载并安装相应版本。此外，我们还建议使用虚拟环境（如[venv](https://link.juejin.cn?target=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fvenv.html)或[conda](https://link.juejin.cn?target=https%3A%2F%2Fdocs.conda.io%2Fen%2Flatest%2F)）来管理项目的依赖库，以避免不同项目之间的库版本冲突。

###### 2、transformers库的安装

安装transformers库非常简单，只需使用pip命令进行安装即可：

```shell
shell 代码解读复制代码# 安装库
pip install transformers

# 查看版本信息
pip show transformers
```

这将会安装最新版本的transformers库以及其依赖库。

如果你需要安装特定版本的transformers库，可以在安装命令中指定版本号，例如：

```shell
shell

 代码解读
复制代码pip install transformers==4.0.0
```

安装完成后，你可以通过以下Python代码来验证transformers库是否安装成功：

```python
python 代码解读复制代码# 验证transformers安装
import transformers
print("transformers version:", transformers.__version__)
```

如果输出了transformers库的版本号，则表示安装成功。

###### 3、其他相关库的安装（如PyTorch、TensorFlow等）

transformers库支持多种深度学习框架，如PyTorch和TensorFlow。

根据你的需求和喜好，你可以选择安装其中一个或多个框架。

以下是安装这些框架的命令：

- 安装PyTorch：

  访问[PyTorch官网](https://link.juejin.cn?target=https%3A%2F%2Fpytorch.org%2Fget-started%2Flocally%2F)，根据你的操作系统和CUDA版本选择合适的安装命令。

  例如，对于Linux系统和CUDA 11版本，可以使用以下命令进行安装：

```shell
shell

 代码解读
复制代码pip3 install torch torchvision torchaudio
```

- 安装TensorFlow：

​	使用pip命令安装TensorFlow即可。

```bash
bash

 代码解读
复制代码pip install tensorflow
```

安装完成后，你可以通过以下Python代码来验证相应框架是否安装成功：

```python
python 代码解读复制代码# 验证PyTorch安装，以及GPU是否可用
import torch
print("torch version:", torch.__version__)
print("cuda is available:", torch.cuda.is_available())
print("cuDNN is available:", torch.backends.cudnn.enabled)
print("GPU numbers:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0))
print("GPU capability:", torch.cuda.get_device_capability(0))
print("GPU memory:", torch.cuda.get_device_properties(0).total_memory)
print("GPU compute capability:", torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor)

# 验证TensorFlow安装
import tensorflow as tf
print("tensorflow version:", tf.__version__)
print("GPU is available:", tf.test.is_gpu_available())
print("GPU name:", tf.test.gpu_device_name())
print("GPU memory:", tf.config.experimental.list_physical_devices('GPU'))
print("GPU number:", len(tf.config.experimental.list_physical_devices('GPU')))
```

如果输出了相应框架的版本号，则表示安装成功。同时还可以看到其它一些关键信息。

现在你已经完成了环境配置和库安装，接下来我们将介绍如何使用transformers库进行模型的加载、训练和微调等操作。

#### 四、transformers库入门

###### 1、加载预训练模型

transformers库提供了丰富的预训练模型，如BERT、GPT-2、RoBERTa等。要加载这些模型，你只需使用相应的模型类和预训练权重名称即可。

例如，加载一个预训练的BERT模型，并查看一些关键的参数信息，可以使用以下代码：

```python
python 代码解读复制代码from transformers import BertModel

# 加载预训练模型，会自动下载
model = BertModel.from_pretrained("bert-base-uncased")
print(model.config)
```

这段代码的目的是使用Transformers库加载一个预训练的BERT模型，并打印模型的配置信息。

接下来，我将详细解释每一行代码的含义。

1. `from transformers import BertModel`：这一行从Transformers库中导入`BertModel`类。Transformers库是Hugging Face公司开发的一个用于处理自然语言处理（NLP）任务的库。`BertModel`类是用于表示BERT模型的基本架构的类。
2. `model = BertModel.from_pretrained("bert-base-uncased")`：这一行加载一个预训练的BERT模型。`from_pretrained()`方法是一个类方法，它根据给定的预训练模型名称（在这里是"bert-base-uncased"）自动下载并加载相应的模型权重。"bert-base-uncased"是BERT模型的一个变体，它使用小写字母进行训练，具有较小的模型大小和计算复杂度。加载完成后，`model`变量将包含一个可以用于各种NLP任务的预训练BERT模型。
3. `print(model.config)`：这一行打印模型的配置信息。`model.config`是一个包含模型配置（例如模型架构、隐藏层大小、注意力头数等）的对象。通过打印这些信息，你可以了解模型的详细配置。

你也可以在[transformers库的模型列表](https://link.juejin.cn?target=https%3A%2F%2Fhuggingface.co%2Fmodels)中找到更多的预训练模型。

类似地，你可以加载其他类型的预训练模型，如GPT-2、RoBERTa等。例如：

```python
python 代码解读复制代码from transformers import GPT2Model, RobertaModel

gpt2_model = GPT2Model.from_pretrained("gpt2")
print(gpt2_model.config)

roberta_model = RobertaModel.from_pretrained("roberta-base")
print(roberta_model.config)
```

###### 2、Tokenizer的使用

在使用预训练模型处理文本之前，我们需要将文本转换为模型可以理解的格式。这就需要使用tokenizer对文本进行分词、编码等操作。transformers库为每种预训练模型提供了相应的tokenizer类，使用方法非常简单。

例如，使用BERT的tokenizer进行文本编码，可以使用以下代码：

```python
python 代码解读复制代码from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "here is some text to encode"
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)
```

这段代码的目的是使用Transformers库中的BERT分词器对输入文本进行编码。编码后的文本可以作为BERT模型的输入。接下来，我将详细解释每一行代码的含义。

1. `from transformers import BertTokenizer`：这一行从Transformers库中导入`BertTokenizer`类。`BertTokenizer`类是用于将原始文本转换为BERT模型可以理解的形式（即整数序列）的工具。
2. `tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")`：这一行加载一个预训练的BERT分词器。`from_pretrained()`方法是一个类方法，它根据给定的预训练模型名称（在这里是"bert-base-uncased"）自动下载并加载相应的分词器。"bert-base-uncased"是BERT模型的一个变体，它使用小写字母进行训练。加载完成后，`tokenizer`变量将包含一个可以用于处理文本的预训练BERT分词器。
3. `text = "here is some text to encode"`：这一行定义了一个字符串变量`text`，用于存储要编码的文本。
4. `encoded_input = tokenizer(text, return_tensors='pt')`：这一行使用分词器对输入文本进行编码。`tokenizer()`方法接受一个文本字符串作为输入，并将其转换为模型可以理解的整数序列。`return_tensors='pt'`参数表示将结果转换为PyTorch张量（`pt`是PyTorch的缩写）。编码后的输入包括`input_ids`（表示文本中每个单词/子词在词汇表中的ID）和`attention_mask`（用于区分实际输入和填充）。
5. `print(encoded_input)`：这一行打印编码后的输入。输出将显示`input_ids`和`attention_mask`。

我这里测试时输出的结果如下：

> {'input_ids': tensor([[  101,  2182,  2003,  2070,  3793,  2000,  4372, 16044,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])}

###### 3、文本分类任务示例

以情感分析任务为例，我们可以使用预训练的BERT模型进行文本分类。

首先，加载一个预训练的BERT模型，并添加一个分类头：

```python
python 代码解读复制代码from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch

# 准备输入文本和对应的标签
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
input_text = ["I love this movie!", "This movie is terrible."]
labels = [1, 0]  # 1表示正面情感，0表示负面情感

# 使用tokenizer对输入文本进行编码：将文本转换为模型可以理解的向量（input_ids和attention_mask）
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoded_inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")

# 将编码结果输入到模型中，得到分类结果：
with torch.no_grad():
    outputs = model(**encoded_inputs)
    logits = outputs.logits
    # 对logits进行argmax操作，得到预测的类别
    predictions = torch.argmax(logits, dim=-1)

print(predictions)
```

这段代码展示了使用Transformers库中的BERT模型对输入文本进行分类。

首先，我们加载了预训练的BERT模型和分词器，然后对输入文本进行编码，将文本转换为模型可以理解的向量（`input_ids`和`attention_mask`）。

接着，我们将编码结果输入到模型中，得到分类结果（`logits`）。

最后，我们对`logits`进行`argmax`操作，以找到具有最高分数的类别，从而得到模型的预测结果。

接下来，我将详细解释每一行代码的含义。

1. `from transformers import BertForSequenceClassification`：从Transformers库中导入`BertForSequenceClassification`类。这个类表示用于文本分类任务的BERT模型。
2. `from transformers import BertTokenizer`：从Transformers库中导入`BertTokenizer`类。这个类表示用于将原始文本转换为BERT模型可以理解的形式（即整数序列）的工具。
3. `import torch`：导入PyTorch库，它是一个用于机器学习和深度学习的开源库。
4. `model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)`：加载预训练的BERT模型，并指定分类任务的类别数（在这里是2个类别，表示正面和负面情感）。
5. `input_text = ["I love this movie!", "This movie is terrible."]`：定义输入文本，这是我们希望模型基于的文本。
6. `labels = [1, 0]`：定义输入文本对应的标签。在这个例子中，1表示正面情感，0表示负面情感。
7. `tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")`：加载预训练的BERT分词器。
8. `encoded_inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")`：使用分词器对输入文本进行编码，将文本转换为模型可以理解的向量（`input_ids`和`attention_mask`）。
9. `with torch.no_grad():`：这是一个上下文管理器，它在其内部的代码块中禁用梯度计算。这在评估或推理阶段（而非训练阶段）使用模型时非常有用，因为它可以减少内存使用并提高计算速度。
10. `outputs = model(**encoded_inputs)`：将编码后的输入传递给BERT模型。`**encoded_inputs`是Python的解包操作，它将`encoded_inputs`字典中的键值对作为关键字参数传递给`model`。这相当于将`input_ids`和`attention_mask`作为单独的参数传递给模型。`model`在接收到输入后，会计算每个类别的概率分数（称为`logits`）。
11. `logits = outputs.logits`：从模型的输出中提取`logits`。`logits`是一个张量，表示模型为每个输入文本计算的类别分数。这些分数可以用于确定输入文本属于哪个类别。
12. `predictions = torch.argmax(logits, dim=-1)`：对`logits`进行`argmax`操作，以找到具有最高分数的类别。`torch.argmax()`函数沿着指定的维度（`dim=-1`表示最后一个维度，即类别维度）返回最大值的索引。`predictions`是一个张量，包含每个输入文本的预测类别。
13. `print(predictions)`：打印预测结果。

###### 4、文本生成任务示例

以GPT-2模型为例，我们可以使用预训练模型进行文本生成。

示例如下：

```python
python 代码解读复制代码# 我们从transformers库中导入GPT2LMHeadModel和GPT2Tokenizer
# GPT2LMHeadModel是GPT-2模型的一个版本，专门用于语言建模任务
# GPT2Tokenizer是用于GPT-2模型的分词器
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 一个预训练的GPT-2模型。("gpt2")表示我们使用的是预训练的"gpt2"模型
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 定义一个字符串text，它将作为我们生成文本的起始
text = "Once upon a time,"

# 使用同样的预训练模型"gpt2"的分词器对输入文本进行编码。编码后的结果被存储在input_ids中：
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
input_ids = tokenizer.encode(text, return_tensors="pt")

# 将编码后的input_ids输入到模型中，然后生成文本
# model.generate函数的参数max_length=50表示生成的文本的最大长度为50
# num_return_sequences=1表示我们只生成一条序列
# 生成的文本被存储在outputs中
# 然后我们使用分词器的batch_decode函数将生成的文本解码，得到我们可以阅读的文本
with torch.no_grad():
    outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for i, generated_text in enumerate(generated_texts):
    print(f"Generated text {i + 1}: {generated_text}")
```

这段代码的目的是使用Transformers库中的预训练GPT-2模型生成一段新文本。

GPT-2是一个自然语言处理（NLP）模型，可以根据给定的起始文本生成连贯的文本。

接下来，我将详细解释每一行代码的含义。

1. `from transformers import GPT2LMHeadModel, GPT2Tokenizer`：这一行从Transformers库中导入`GPT2LMHeadModel`类和`GPT2Tokenizer`类。`GPT2LMHeadModel`类表示用于语言建模任务的GPT-2模型。`GPT2Tokenizer`类是用于将原始文本转换为GPT-2模型可以理解的形式（即整数序列）的工具。
2. `model = GPT2LMHeadModel.from_pretrained("gpt2")`：这一行加载一个预训练的GPT-2模型。`from_pretrained()`方法是一个类方法，它根据给定的预训练模型名称（在这里是"gpt2"）自动下载并加载相应的模型权重。加载完成后，`model`变量将包含一个可以用于生成文本的预训练GPT-2模型。
3. `text = "Once upon a time,"`：这一行定义了一个字符串变量`text`，用于存储要生成文本的起始。
4. `tokenizer = GPT2Tokenizer.from_pretrained("gpt2")`：这一行加载一个预训练的GPT-2分词器。`from_pretrained()`方法是一个类方法，它根据给定的预训练模型名称（在这里是"gpt2"）自动下载并加载相应的分词器。加载完成后，`tokenizer`变量将包含一个可以用于处理文本的预训练GPT-2分词器。
5. `input_ids = tokenizer.encode(text, return_tensors="pt")`：这一行使用分词器对输入文本进行编码。`tokenizer.encode()`方法接受一个文本字符串作为输入，并将其转换为模型可以理解的整数序列。`return_tensors="pt"`参数表示将结果转换为PyTorch张量（`pt`是PyTorch的缩写）。编码后的输入包括`input_ids`。
6. `with torch.no_grad():`：这是一个上下文管理器，它在其内部的代码块中禁用梯度计算。这在评估或推理阶段（而非训练阶段）使用模型时非常有用，因为它可以减少内存使用并提高计算速度。由于我们只生成文本而不进行训练，因此不需要计算梯度。
7. `outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)`：这一行使用模型根据输入文本生成新文本。`model.generate()`方法接受一些参数，如`max_length`（生成文本的最大长度）和`num_return_sequences`（要生成的序列数量）。在这个例子中，我们设置`max_length=50`以限制生成文本的长度，设置`num_return_sequences=1`以生成一个序列。
8. `generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)`：这一行使用分词器将生成的输出向量解码为文本。`tokenizer.batch_decode()`方法接受一个整数序列（在这里是`outputs`）并将其转换回原始文本。`skip_special_tokens=True`参数表示在解码过程中跳过特殊字符（如开始、结束和填充字符）。
9. `for i, generated_text in enumerate(generated_texts):`：这一行遍历生成的文本列表。
10. `print(f"Generated text {i + 1}: {generated_text}")`：这一行打印生成的文本。

我这里测试的时候输出结果为:

> *Generated text 1: Once upon a time, the world was a place of great beauty and great danger. The world was a place of great danger, and the world was a place of great danger. The world was a place of great danger, and the world was a*

上述过程可以应用于各种自然语言生成任务，如文本摘要、问答、对话生成等。GPT-2模型是一种强大的自然语言处理工具，可以生成与给定输入文本相关的连贯文本。通过调整`model.generate()`方法的参数，你可以控制生成文本的长度、数量以及其他特性。