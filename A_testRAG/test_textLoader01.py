from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone #向量数据库
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone

'''
下面的这部分代码是将文件夹中的word文档，上传到自己的向量数据库
'''
#首先进入文件夹查看数据
directory_path=r"D:/wk/data/bs_challenge_financial_14b_dataset/pdf_txt_file/tmp_files"
data = []
# loop through each file in the directory
for filename in os.listdir(directory_path):
    # check if the file is a doc or docx file
    # 检查所有doc以及docx后缀的文件
    if filename.endswith(".doc") or filename.endswith(".txt"):
        # print the file name
        # langchain自带功能，加载word文档
        loader = TextLoader(f'{directory_path}/{filename}',encoding='utf8')
        print(loader)
        data.append(loader.load_and_split())
print(len(data))
print(data)
print(data[0])
print(data[0][0])
#Chunking the data into smaller pieces
#再用菜刀把文档分隔开，chunk_size就是我们要切多大，建议设置700及以下，因为openai有字数限制，chunk_overlap就是重复上下文多少个字
texts = []
docs = loader.load_and_split()
print(docs[0])
