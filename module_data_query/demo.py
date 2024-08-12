from module_data_query.sql_generate_excute_evalute import sql_excute_evalue
from model_client.api_request import local_llm_api_request,openai_api_request
from prompt.sql_generate_prompt import base_prompt
from untils.sqlite_toolkit import sqliteToolkit
import json
import pandas as pd
import random


# 1. 模型生成sql
# 1.1 准备种子sql对
sql_seed_pair_path = "../data/sql_generate_result.xlsx"

## 1.2 根据种子，利用chatgpt生成1000条数据集
df = pd.read_excel(sql_seed_pair_path)
example_dic = {}
examples = ''
for i in range(len(df)):
    random_num = random.sample(range(len(df)), 5)
    for element in random_num:
        example_dic['query'] = df['question'][element]
        example_dic['sql'] = df['sql'][element]
        example = json.dumps(example_dic,ensure_ascii=False)
        examples = examples + "\n" + example
    print(example_dic)
    print(examples)

    system_prompt = base_prompt.replace("{examples}",examples)
    print(system_prompt)
    user_prompt = df['question'][i]
    result = openai_api_request(system_prompt, user_prompt)
    print(result)


## 1.3 数据提问泛化


# 1.4 模型指令微调


## 1.5 编写提示词，生成sql
## 1.6 结果执行
## 1.7 结果评估
sql_base_path = "../data/sql_seed_result.xlsx"
db_source_path = r'D:\wk\data\bs_challenge_financial_14b_dataset\dataset\博金杯比赛数据.db'
sql_evalue_path = '../data/sql_generate_result.xlsx'
sql_excute_evalue(sql_base_path, db_source_path, sql_evalue_path)


