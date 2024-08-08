from transformers import AutoTokenizer, AutoModel
from pymilvus import connections, Collection, FieldSchema,  DataType
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
model = SentenceTransformer(r"D:\wk\model\xiaobu-embedding-v2")

connections.connect("default", host='8.137.147.223', port='19530')

# 1. 加载模型
tokenizer = AutoTokenizer.from_pretrained(r'D:\wk\model\text2vec-base-chinese')
model = AutoModel.from_pretrained(r'D:\wk\model\text2vec-base-chinese')

# 假设的中文文本
text = "你好，世界！"

# 2. 将中文文本转换为embedding向量
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    embedding = last_hidden_states.mean(dim=1).squeeze().numpy()  # 取平均向量作为整个句子的embedding

# 3. 连接到Milvus数据库（需要提供你的连接参数）


# 4. 在Milvus中创建集合
if not Collection.has_collection("text_embeddings"):
    field1 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding.shape[-1])
    collection_mapping = CollectionMapping(
        fields=[field1],
        default_vector_dim=embedding.shape[-1]
    )
    Collection.create_collection("text_embeddings", collection_mapping)

# 5. 插入embedding向量到Milvus集合中
collection = Collection("text_embeddings")
collection.insert([[embedding]])

# 断开Milvus连接（可选）
connections.disconnect("default")