import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

sentences = ['How is the weather today?', '今天天气怎么样?']

tokenizer = AutoTokenizer.from_pretrained(r'D:\wk\jina-embeddings-v2-base-zh')
model = AutoModel.from_pretrained(r'D:\wk\jina-embeddings-v2-base-zh', trust_remote_code=False)

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    model_output = model(**encoded_input)

print(model_output)
embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
print(embeddings)
embeddings = F.normalize(embeddings, p=2, dim=1)
print(embeddings)