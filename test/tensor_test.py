

import torch

tensor1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
tensor2 = torch.tensor([[1.0, 2.0, 3.1], [4.1, 5.0, 6.0]])
print(tensor1.unsqueeze(0))


# 计算两个张量之间的相似性，例如使用余弦相似度
# PyTorch提供了一个直接的余弦相似度函数

# 计算所有向量对之间的余弦相似度（这将返回一个2x2的矩阵）
cosine_similarity = torch.nn.functional.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0), dim=2)
print(cosine_similarity)

# 如果你只关心第一行之间的余弦相似度，可以索引结果
cosine_similarity_first_row = cosine_similarity[0, 0]
print(cosine_similarity_first_row)

