import pandas as pd
from prompt.sql_generate_prompt import base_prompt
from model_client.api_request import local_llm_api_request
import json
from untils.sqlite_toolkit import sqliteToolkit
from untils.json_toolkit import extract_jsonstr

def sql_excute_evalue(sql_base_path,db_source_path,sql_evalue_path):

    df = pd.read_excel(sql_base_path)
    df['predict_result'] = ''

    ## 1.5 编写提示词，生成sql
    db_source = db_source_path
    sqllite_clinet = sqliteToolkit(db_source)
    count = 0
    for i in range(len(df)):
        print(f'------------------第{i}行数据-------------------')
        user_prompt = df['question'][i]

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

        response_result = extract_jsonstr(response_result)
        # query_dict = json.loads(response_result)
        print(response_result)
        print(type(response_result))
        query = response_result['sql']

        try:
            result = sqllite_clinet.query(query)
            print(result)
            df['predict_result'][i] = result
        except Exception as e:
            result = e
            df['predict_result'][i] = result
        finally:
            sqllite_clinet.conn.close()

        if str(df['result'][i]) == str(df['predict_result'][i]):
            count += 1

    print(df)
    df['acc'] = count/len(df)
    df.to_excel(sql_evalue_path, index=False, engine='openpyxl')
    return df

if __name__ == '__main__':
    sql_base_path = "../data/sql_seed_result.xlsx"
    db_source_path = r'D:\wk\data\bs_challenge_financial_14b_dataset\dataset\博金杯比赛数据.db'
    sql_evalue_path = '../data/sql_generate_result.xlsx'
    sql_excute_evalue(sql_base_path,db_source_path,sql_evalue_path)



