import pinecone
# from langchain.vectorstores import Pinecone

from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY='bad20e65-b265-4b00-b0d0-8bbb1bd8b3ba'
PINECONE_API_ENV='us-east-1'

pc = Pinecone(api_key="bad20e65-b265-4b00-b0d0-8bbb1bd8b3ba")
# index = pc.Index("quickstart")

texts = ['我叫付博','你叫什么']
embeddings = [[0.1,0.2,0.3],[0.4,0.5,0.6]]

index_name = "dental"
for i in range(len(texts)):
    Pinecone.from_texts([t for t in texts[i]],embeddings, index_name=index_name)
    print("done")



texts = ['我叫付博','你叫什么']
embeddings = [[0.1,0.2,0.3],[0.4,0.5,0.6]]

pc = pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)

index_name = "quickstart" # put in the name of your pinecone index here
for i in range(len(texts)):
    Pinecone.from_texts([t.page_content for t in texts[i]], embeddings, index_name=index_name)
    print("done")