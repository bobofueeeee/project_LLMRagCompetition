

# 1. 将文本信息转换本地的向量数据库
# 输入：文本信息地址
# 操作：存入向量数据库

from module.text2vec import text2vec,question2vec
from module.compare_smilary import compare_similary
path = "../data/qa.json"
qaVec = text2vec(path)
print(qaVec)


# 2. 将提问转换为词向量
# 输入：提问

question = "你叫什么名字！！！"
questionVec = question2vec(question)
print(questionVec)


# 3. 比较词向量，本地的向量数据库选出相似度最高的
# 输入：提问转换为的词向量
# 输出：在向量数据库中检索，返回相似都最高的问题+答案

mostSimilary = compare_similary(questionVec,qaVec)
print(mostSimilary)


# 4. 问题+答案输入大语言模型进行整理
# 输入：问题+答案文本信息
# 输出：整理过后的答案，json文件

