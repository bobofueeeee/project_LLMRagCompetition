

# 1. 读取赛题数据

## 1.1 读取问题

import jsonlines

def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content


def write_jsonl(path, content):
    with jsonlines.open(path, "w") as json_file:
        json_file.write_all(content)

def read_txt(path):
    content = []
    with open(path,"r",encoding="utf-8") as txt_file:
        lines = txt_file.readlines()
        for line in lines:
          content.append(line)
    return content


questionpath = r"D:\wk\bs_challenge_financial_14b_dataset\question.json"
question = read_jsonl(questionpath)
# print(question)


## 1.2 读取txt文档
txtpath = r"D:\wk\bs_challenge_financial_14b_dataset\pdf_txt_file\0b46f7a2d67b5b59ad67cafffa0e12a9f0837790.txt"
txt = read_txt(txtpath)
print(txt[0])

## 2. 转换为词向量
## 2.1 加载词向量模型
from transformers import BertModel, BertTokenizer

