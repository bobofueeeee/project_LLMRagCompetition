# 加载txt文件  
from langchain.document_loaders import UnstructuredFileLoader

text_loader = UnstructuredFileLoader("D:/wk/bs_challenge_financial_14b_dataset\pdf_txt_file/0b46f7a2d67b5b59ad67cafffa0e12a9f0837790.txt")
document = text_loader.load()

# 使用文本拆分器（这里假设我们想要基于字符进行简单拆分）  
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(chunk_size=100)  # 假设我们想要每100个字符一个块  
text_chunks = splitter.split_text(document)

# 输出拆分后的文本块  
for chunk in text_chunks:
    print(chunk)