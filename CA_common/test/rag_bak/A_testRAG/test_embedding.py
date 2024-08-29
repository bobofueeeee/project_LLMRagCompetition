from transformers import BertTokenizer, BertModel
import torch
from pymilvus import MilvusClient
    # connections, Collection, FieldSchema, CollectionMapping


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    print(attention_mask)
    print(attention_mask.unsqueeze(-1)) # 横向维度转换为了纵向维度
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = BertTokenizer.from_pretrained(r'D:\wk\model\text2vec-base-chinese')
model = BertModel.from_pretrained(r'D:\wk\model\text2vec-base-chinese')
sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']
# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
# Perform pooling. In this case, mean pooling.
print(model_output)
print(model_output[0])
print(encoded_input['attention_mask'])
print(type(encoded_input['attention_mask']))
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
print("Sentence embeddings:")
print(sentence_embeddings)