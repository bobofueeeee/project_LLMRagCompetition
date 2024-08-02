# 输入：文件地址
# 输出：文件内容(json)

import nltk
from nltk.tokenize import sent_tokenize

import pymupdf # imports the pymupdf library

nltk.download('punkt')  # 下载必要的数据
doc = pymupdf.open(r"D:\wk\data\bs_challenge_financial_14b_dataset\pdf\3e0ded8afa8f8aa952fd8179b109d6e67578c2dd.PDF") # open a document
for page in doc: # iterate the document pages
  text = page.get_text() # get plain text encoded as UTF-8
  print(text)
  sentences = sent_tokenize(text)
  print('-------------------------sentences--------------------------------')
  print(sentences)






