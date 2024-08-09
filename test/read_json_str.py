import re
import json

# 示例文本
text = """
这是一个包含JSON字符串的示例文本：
{
    "name": "Alice",
    "age": 30,
    "city": "New York"
}
文本结束。
"""

# 使用正则表达式查找JSON字符串
json_pattern = r'\{.*\}'
match = re.search(json_pattern, text, re.DOTALL)

if match:
    json_string = match.group(0)
    try:
        # 解析JSON字符串
        json_data = json.loads(json_string)
        print("解析的JSON数据：", json_data)
    except json.JSONDecodeError as e:
        print("JSON解析错误：", e)
else:
    print("未找到JSON字符串")