import torch

# 假设我们有两个向量集合，分别用tensor A和B表示
# A的shape是[m, d]，B的shape是[n, d]，其中d是向量的维度

m, n, d = 100, 200, 128  # 示例维度，你可以根据实际情况调整
A = torch.randn(m, d)  # 随机初始化A
B = torch.randn(n, d)  # 随机初始化B
print(A)
print(B)

# 计算A中每个向量与B中所有向量的余弦相似度
# 注意：PyTorch没有直接的cosine_similarity函数，但我们可以使用mm和norm来计算
# 首先计算A和B的归一化（单位化）
A_norm = A / A.norm(dim=1, keepdim=True)
B_norm = B / B.norm(dim=1, keepdim=True)
print(A_norm)
print(B_norm)

# 计算余弦相似度矩阵
# (A_norm @ B_norm.t()) 的每个元素是 A 的一个行向量与 B 的一个列向量的点积
# 因为A_norm和B_norm都是单位向量，点积就是余弦相似度
cosine_similarity_matrix = A_norm @ B_norm.t()
print(cosine_similarity_matrix)

# 假设我们想要找到A中每个向量在B中最相似的5个向量
# 对余弦相似度矩阵的每一行进行排序，并取出前5个最大的索引
# 注意：这里我们假设我们只关心A中每个向量与B中最相似的向量，而不是所有可能的向量对
top_indices = cosine_similarity_matrix.topk(k=5, dim=1, largest=True, sorted=True).indices
print(top_indices)

# 现在top_indices包含了A中每个向量在B中最相似的5个向量的索引
# 你可以根据需要进一步处理这些索引，例如提取对应的B中的向量

# 例如，提取A中第一个向量在B中最相似的5个向量
most_similar_in_B_for_first_in_A = B[top_indices[0]]
print(most_similar_in_B_for_first_in_A)