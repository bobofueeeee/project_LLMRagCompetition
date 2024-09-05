from transformers import BertTokenizer, BertModel
import torch
from H_common.test.rag_bak import load_file

# 1. 读取文档 + 文档分割
print('-----------------------1. 读取文档 + 文档分割-----------------------------')
filepath=r"D:\wk\data\bs_challenge_financial_14b_dataset\pdf_txt_file\tmp_files\0b46f7a2d67b5b59ad67cafffa0e12a9f0837790.txt"
ZH_TITLE_ENHANCE=True
docs = load_file(filepath,sentence_size=100,using_zh_title_enhance=ZH_TITLE_ENHANCE)
print("----------------------------类型为---------------------------------")
print(type(docs[0]))
print("----------------------------具体值---------------------------------")
print(docs[0])
print(docs[0].page_content)
print(docs[0].metadata)
print('-----------------------1. 读取文档 + 文档分割-----------------------------')

# text_list = [doc.page_content for doc in docs]
# print(text_list)

# 3. 向量转换 + 向量存储
# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    print(token_embeddings.size())
    # print(attention_mask)
    # print(attention_mask.unsqueeze(-1)) # 横向维度转换为了纵向维度
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    print(type(input_mask_expanded))
    print(input_mask_expanded.size())
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = BertTokenizer.from_pretrained(r'D:\wk\model\text2vec-base-chinese')
model = BertModel.from_pretrained(r'D:\wk\model\text2vec-base-chinese')

vectors = []

print('-----------------------3.向量转换 + 向量存储-----------------------------')
# Tokenize sentences
for i in range(len(docs)):
    # 添加原来的question到metadata
    docs[i].metadata['question'] = docs[i].page_content
    # print(print(docs[i].metadata))

    encoded_input = tokenizer(docs[i].page_content, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling. In this case, mean pooling.
    # print(model_output)
    # print(model_output[0])
    # print(encoded_input['attention_mask'])
    # print(type(encoded_input['attention_mask']))
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = sentence_embeddings.tolist()[0]
    print(sentence_embeddings)
    print(len(sentence_embeddings))
    # print(type(sentence_embeddings))
    # print("Sentence embeddings:")
    # print(sentence_embeddings)
    vectors.append({"id": f"vec{i}","values":sentence_embeddings,"metadata":docs[i].metadata})

print(vectors)
print(vectors[0])
print(type(vectors[0]))
print(vectors[0]['id'])
print(vectors[0]['values'])
print(type(vectors[0]['values']))
print(vectors[0]['metadata'])
#
import pinecone
pc = pinecone.Pinecone(api_key="bad20e65-b265-4b00-b0d0-8bbb1bd8b3ba")
index = pc.Index("llmrag")
#
index.upsert(vectors=vectors,namespace= "ns1")
print('-----------------------3.向量转换 + 向量存储-----------------------------')

# 4. 近似匹配
print('-----------------------近似匹配-----------------------------')
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
print(query)
print(type(query))
print(query)
print('-----------------------近似匹配-----------------------------')




