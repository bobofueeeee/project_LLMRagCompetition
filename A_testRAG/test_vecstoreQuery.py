import os
from transformers import BertTokenizer, BertModel
import torch
from A_testRAG.AA_textLoader.loader.loadfile import load_file

import pinecone




def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    print(token_embeddings.size())
    # print(attention_mask)
    # print(attention_mask.unsqueeze(-1)) # 横向维度转换为了纵向维度
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    print(type(input_mask_expanded))
    print(input_mask_expanded.size())
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


print('-----------------------近似匹配-----------------------------')
# Load model from HuggingFace Hub
tokenizer = BertTokenizer.from_pretrained(r'D:\wk\model\text2vec-base-chinese')
model = BertModel.from_pretrained(r'D:\wk\model\text2vec-base-chinese')

pc = pinecone.Pinecone(api_key="bad20e65-b265-4b00-b0d0-8bbb1bd8b3ba")
index = pc.Index("llmrag")

question = "本次发行前公司总股本有多少股？"
question_input = tokenizer(question, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    model_output = model(**question_input)
question_embeddings = mean_pooling(model_output, question_input['attention_mask'])
question_embeddings = question_embeddings.tolist()[0]
print(question_embeddings)
print(type(question_embeddings))
print(len(question_embeddings))
query= index.query(
    namespace="ns1",
    vector=question_embeddings,
    top_k=3,
    include_values=True,
    include_metadata=True
    # filter={"genre": {"$eq": "action"}}
)
# print(query)
# print(type(query))
# print(type(query.matches))
print(query.matches[0]['score'])
print(query.matches[0]['metadata']['question'])
print(query.matches[1]['score'])
print(query.matches[1]['metadata']['question'])


print('-----------------------近似匹配-----------------------------')
