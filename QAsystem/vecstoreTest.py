from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# 1. 文本加载
loader = TextLoader(r"D:\wk\bs_challenge_financial_14b_dataset\test.txt",encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(texts)


# 2. 向量模型加载

from langchain.embeddings import HuggingFaceEmbeddings

model_name = "D:/wk/text2vec-base-chinese"
embeddings = HuggingFaceEmbeddings(model_name="D:/wk/text2vec-base-chinese")
print(embeddings)
text = "昇兴集团股份有限公司本次发行的保荐人是？"
query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])
print(query_result)
print(doc_result)

vectorstore = Chroma.from_documents(texts, embeddings)
print(type(vectorstore))
print(vectorstore)


# 3. 历史对话记载
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 4. 模型加载

from langchain.llms import ollama


# 5. chain生成
from langchain.chains import ConversationalRetrievalChain
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(), memory=memory)

