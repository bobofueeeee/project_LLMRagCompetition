from model_client.api_request import local_llm_api_request
from prompt.sql_generate_prompt import base_prompt
from untils.sqlite_toolkit import sqliteToolkit
import json

# 1. 模型生成sql
# 1.1 准备种子sql对
sql_seed_pair_path = "data/sql_种子.xlsx"

## 1.2 根据种子，利用chatgpt生成1000条数据集




## 1.3 数据提问泛化


# 1.4 模型指令微调


## 1.5 编写提示词，生成sql
user_prompt = "我想知道海富通基金管理有限公司在2020年成立了多少只管理费率小于0.8%的基金？"
examples = """例子1：{"query": "帮我查一下鹏扬景科混合A基金在20201126的资产净值和单位净值是多少?", "sql": "SELECT 资产净值, 单位净值 FROM 基金日行情表 WHERE 基金代码  = (SELECT 基金代码 FROM 基金基本信息 WHERE 基金简称 LIKE '鹏扬景科混合A%') AND 交易日期 = '20201126'"}
 例子2: {"query": "嘉实中债1-3年政策性金融债指数C基金在20211231的季报里，前三大持仓占比的债券名称是什么?", "sql": "SELECT 债券名称 FROM 基金债券持仓明细 WHERE 持仓日期='20211231' ORDER BY 持债市值占基金资产净值比 DESC LIMIT 3"}"""
system_prompt = base_prompt.replace("{examples}",examples)
print(system_prompt)

url = 'http://1.95.86.245:51000/v1/chat/completions'
model_name = "/home/ruantong/mode/Qwen2-72B-Instruct"

result = local_llm_api_request(url, model_name, system_prompt, user_prompt)
response_result = result['response_result']
print(result)
print(response_result)

## 1.6 结果执行
query_dict = json.loads(response_result)
print(query_dict)
print(type(query_dict))
query = query_dict['sql']

db_source = r'D:\wk\data\bs_challenge_financial_14b_dataset\dataset\博金杯比赛数据.db'
sqllite_clinet = sqliteToolkit(db_source)
result = sqllite_clinet.query(query)
print(result)



## 1.7 结果评估



