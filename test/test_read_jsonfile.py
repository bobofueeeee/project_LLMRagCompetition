import json

def read_large_json(file_path):
    with open(file_path, 'r' ,encoding= 'utf-8') as file:
        for line in file:
            # 解析每一行的 JSON 数据
            data = json.loads(line)
            # 处理数据
            print(data)

# 替换为你的 JSON 文件路径
read_large_json('../data/question.json')