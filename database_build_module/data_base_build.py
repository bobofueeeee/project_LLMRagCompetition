from untils.pdf_toolkit import pdf_load_chunk_emb_save
from sentence_transformers import SentenceTransformer


# 1. 读取，切分pdf
pdf_path = r"D:\wk\deepLearning_LLMRagCompetition\data\2389de12d78fe1ca4fa24910e6b1573902098bc3.PDF"
results = pdf_loadAndChunk(pdf_path)

# 2. 向量化
model = SentenceTransformer(r"D:\wk\model\xiaobu-embedding-v2")
for result in results:
    embeddings = model.encode(result['content'], normalize_embeddings=True)
    print(embeddings)

# 3. 存入向量库
