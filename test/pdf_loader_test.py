# 输入：文件地址
# 输出：文件内容(json)

from untils.textsplitter import ChineseTextSplitter

import pymupdf # imports the pymupdf library

doc = pymupdf.open(r"D:\wk\data\bs_challenge_financial_14b_dataset\pdf\3e0ded8afa8f8aa952fd8179b109d6e67578c2dd.PDF") # open a document
spliter = ChineseTextSplitter(pdf=True)
for page in doc: # iterate the document pages
  text = page.get_text() # get plain text encoded as UTF-8
  # print(text)
  result = spliter.split_text1(text)
  print(result)
  # sentences = sent_tokenize(text)
  # print('-------------------------sentences--------------------------------')
  # print(sentences)






