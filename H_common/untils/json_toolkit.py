import re
import json

def extract_jsonstr(text):
    # 使用正则表达式查找JSON字符串
    json_pattern = r'\{.*\}'
    match = re.search(json_pattern, text, re.DOTALL)

    if match:
        json_string = match.group(0)
        try:
            # 解析JSON字符串
            json_data = json.loads(json_string)
            print("解析的JSON数据：", json_data)
            return json_data
        except json.JSONDecodeError as e:
            json_data = json.loads(str({"query": "JSON解析错误","sql": "JSON解析错误"}))
            print("JSON解析错误：", e)
            return json_data
    else:
        json_data = json.loads(str({"query": "未找到JSON字符串", "sql": "未找到JSON字符串"}))
        print("未找到JSON字符串")
        return json_data

def read_jsonfile(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每一行的 JSON 数据
            data = json.loads(line)
            # 处理数据
            print(data)


if __name__ == '__main__':
    # extract_jsonstr 示例用法
    text = """
    这是一个包含JSON字符串的示例文本：
    {
        "name": "Alice",
        "age": 30,
        "city": "New York"
    }
    文本结束。
    """
    result = extract_jsonstr(text)

    # --------------------------------------------------------------------
    # read_jsonfile 示例用法
    read_jsonfile('../../G_data/question.json')