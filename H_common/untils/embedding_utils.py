import torch

from sentence_transformers import SentenceTransformer

class XiaobuEmbeddingTool:
    def __init__(self, model_path="lier007/xiaobu-embedding-v2"):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name_or_path=model_path, local_files_only=True,
                                         device=device)
    def text_embedding(self, text: str):
        """文本向量化"""
        embeddings = self.model.encode(text, normalize_embeddings=True)
        return embeddings

    def text_embedding_complete(self, text: str):
        """文本向量化，完全显示"""
        embeddings = self.model.encode(text, normalize_embeddings=True)
        temp = ', '.join(map(str, embeddings))
        temp = '[' + temp + ']'
        # print(temp)
        return temp

    def similarity_calculation(self, sentences_1: list, sentences_2: list):
        """相似度计算，转换embedding的维度是1792维"""
        embeddings_1 = self.model.encode(sentences_1, normalize_embeddings=True)
        embeddings_2 = self.model.encode(sentences_2, normalize_embeddings=True)
        similarity = embeddings_1 @ embeddings_2.T
        return similarity


# 示例用法
if __name__ == "__main__":
    sentences_1 = ["孙悟空", "无天"]
    sentences_2 = ["紧那罗", "阿依纳伐"]

    embedding_model_xiaobu = XiaobuEmbeddingTool(model_path=r'C:\workspace\root\AISHU AnyShare\dsmm_data\ShareCache\职能部门\信息技术中心\100、2024年项目管理\04-AI内容生成\002、袋鼠妈妈AI大模型\002、代码库\dsmm_model\com\dsmm\data\model\embedding\xiaobu-embedding-v2')

    embedding_vector = embedding_model_xiaobu.text_embedding('Piccolo2主要关注在一种通用的下游微调范式。我们的开源模型使用了stella-v3.5作为初始化，在32张A100上训练了2500 step，对于更多的实现细节，可以参考我们的 技术报告, 以及训练代码')
    print(embedding_vector)

    similarity = embedding_model_xiaobu.similarity_calculation(sentences_1, sentences_2)
    print(similarity)