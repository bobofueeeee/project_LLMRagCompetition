from model_client import api_request
import json
from prompt.intent_recognise_prompt import intent_recog_prompt
from untils.json_toolkit import extract_jsonstr

file_path = '../data/question.json'
url = 'http://1.95.86.245:51000/v1/chat/completions'
model_name = "/home/ruantong/mode/Qwen2-72B-Instruct"
system_prompt = intent_recog_prompt

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 解析每一行的 JSON 数据
        data = json.loads(line)
        # 处理数据
        # print(data)
        question = data['question']
        print('----------------------------------------------------')
        print(question)

        # 1. 意图识别
        user_prompt = question
        result = api_request.local_llm_api_request(url, model_name, system_prompt, user_prompt)
        response_result = result['response_result']
        # print(result)
        print(response_result)
        result_dict = extract_jsonstr(response_result)

        # 2. 任务分发
        if result_dict['label'] == '未匹配到公司':
            print('走数据查询模块')
            print('----------------------------------------------------')
        elif result_dict['label'] == '匹配到公司':
            print('走知识检索模块')
            print('----------------------------------------------------')
        else:
            print('格式错误')





