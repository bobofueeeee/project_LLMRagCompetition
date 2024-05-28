import jieba  # 用于中文分词的库
from collections import defaultdict

# 假设我们有一个简单的“词向量”字典（实际中你会从预训练的模型中加载）
fake_word_vectors = {
    '你好': [0.1, 0.2, 0.3],
    '世界': [0.4, 0.5, 0.6],
    # ... 其他词的向量
}

# 简化的“向量数据库”，使用字典存储
vector_database = defaultdict(list)

def segment_and_store(sentence):
    # 使用jieba进行分词
    words = jieba.cut(sentence, cut_all=False)
    print(words)
    for word in words:
        print(word)
        # 假设所有词都在我们的fake_word_vectors中（实际中需要处理未登录词）
        if word in fake_word_vectors:
            # 将词向量存入“向量数据库”
            vector_database[sentence].append(fake_word_vectors[word])
        # 示例句子

sentence = "你好，世界！"
segment_and_store(sentence)

# 查看结果
print(vector_database)