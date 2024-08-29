import json

from BA_module_intent_recognise.intent_recognise import intent_recongnise
from BE_prompt.intent_recognise_prompt import intent_recog_prompt
import pandas as pd

file_path = '../AA_data/question.json'
url = 'http://1.95.86.245:51000/v1/chat/completions'
model_name = "/home/ruantong/mode/Qwen2-72B-Instruct"
system_prompt = intent_recog_prompt
df_result = pd.DataFrame(columns=['question','sql'])
i = 0

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 解析每一行的 JSON 数据
        data = json.loads(line)
        # 处理数据
        # print(AA_data)
        question = data['question']
        print('----------------------------------------------------')
        print(question)

        # 1. 意图识别
        recongnise_result = intent_recongnise(question,url,model_name,system_prompt)


        # 2. 任务分发
        if recongnise_result['label'] == '未匹配到公司':
            print('走数据查询模块')
            print('----------------------------------------------------')

        elif recongnise_result['label'] == '匹配到公司':
            print('走知识检索模块')
            print('----------------------------------------------------')

        else:
            print('格式错误')







