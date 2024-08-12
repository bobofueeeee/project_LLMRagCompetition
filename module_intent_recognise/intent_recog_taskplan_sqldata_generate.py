from model_client import api_request
import json
from prompt.intent_recognise_prompt import intent_recog_prompt
from untils.json_toolkit import extract_jsonstr
from module_data_query.sql_dataset_generate import sql_dataset_generate
import pandas as pd

file_path = '../data/question.json'
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
            # sql_dataset_generate 示例用法
            sql_seed_pair_path = "../data/sql_generate_result.xlsx"
            # query01 = '帮我查一下鹏扬景科混合A基金在20201126的资产净值和单位净值是多少?'
            # query02 = '嘉实中债1-3年政策性金融债指数C基金在20211231的季报里，前三大持仓占比的债券名称是什么?'
            # query03 = '在20210630的年报(含半年报)中，永赢泽利一年定期开放债券基金的债券持仓，其持有最大仓位的债券类型是什么?'

            result = sql_dataset_generate(sql_seed_pair_path, result_dict['question'])
            print(result)
            df_result.loc[i] = [result['question'],result['sql']]
            i += 1
            if (i % 100) == 0:
                df_result.to_excel(f'../data/sql_dataset{i}.xlsx', index=False, engine='openpyxl')
                df_result = pd.DataFrame(columns=['question', 'sql'])
            print('----------------------------------------------------')
        elif result_dict['label'] == '匹配到公司':
            print('走知识检索模块')
            print('----------------------------------------------------')
        else:
            print('格式错误')







