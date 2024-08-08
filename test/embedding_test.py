from sentence_transformers import SentenceTransformer
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
sentences_3 = "样例数据-5"
model = SentenceTransformer(r"D:\wk\model\xiaobu-embedding-v2")
embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
embeddings_3 = model.encode(sentences_3, normalize_embeddings=True)

similarity = embeddings_1 @ embeddings_2.T
print(similarity)