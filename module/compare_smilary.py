

## 找出相似性最高的向量

from torch.nn.functional import cosine_similarity
import torch


def compare_similary(tensor1,vecstore):
    dic = {}
    most_simlarity_score = 0
    for i in range(len(vecstore)):
        print(tensor1.unsqueeze(0))
        print(vecstore[i]['question_tensor'].unsqueeze(0))
        cosine_sim = cosine_similarity(tensor1.unsqueeze(0), vecstore[i]['question_tensor'].unsqueeze(0))
        simlarity_score = cosine_sim.mean().item()
        if simlarity_score > most_simlarity_score:
            most_simlarity_score = simlarity_score
            dic['question_tensor'] = vecstore[i]['question_tensor']
            dic['answer_tensor'] = vecstore[i]['answer_tensor']
            dic['question'] = vecstore[i]['question']
            dic['answer'] = vecstore[i]['answer']

    return dic

# tensor1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# vecstore = [{'id': 1, 'question':torch.tensor([[1.0, 2.0, 3.0]]),'answer':torch.tensor([[4.0, 5.0, 6.0]])},
#             {'id': 2, 'question':torch.tensor([[3.0, 4.0, 5.0]]),'answer':torch.tensor([[6.0, 7.0, 8.0]])}]
#
#
# mostSimilary = compare_similary(tensor1,vecstore)
# print(mostSimilaryVec)


