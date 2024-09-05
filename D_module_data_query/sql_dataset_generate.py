from H_common.model_client.api_request import openai_api_request
from E_prompt.sql_generate_prompt import base_prompt
import json
import pandas as pd
import random
from H_common.untils import extract_jsonstr


def sql_dataset_generate(sql_seed_pair_path, query):
    # 1. 模型生成sql
    # 1.1 准备种子sql对
    ## 1.2 根据种子，利用chatgpt生成1000条数据集
    df = pd.read_excel(sql_seed_pair_path)
    # df_result = pd.DataFrame(columns=['question','sql'])

    example_dic = {}
    examples = ''

    random_num = random.sample(range(len(df)), 10)
    for element in random_num:
        example_dic['query'] = df['question'][element]
        example_dic['sql'] = df['sql'][element]
        example = json.dumps(example_dic,ensure_ascii=False)
        examples = examples + "\n" + example
    # print(example_dic)
    print(examples)

    system_prompt = base_prompt.replace("{examples}",examples)
    # print(system_prompt)
    user_prompt = query
    result = openai_api_request(system_prompt, user_prompt)
    result = extract_jsonstr(result)
    # print(result)
    result_dic = {"question" : user_prompt, "sql": result['sql']}
    return result_dic

if __name__ == '__main__':
    # sql_dataset_generate 示例用法
    # sql_seed_pair_path = "../G_data/sql_generate_result.xlsx"
    # query01 = '帮我查一下鹏扬景科混合A基金在20201126的资产净值和单位净值是多少?'
    # query02 = '嘉实中债1-3年政策性金融债指数C基金在20211231的季报里，前三大持仓占比的债券名称是什么?'
    # query03 = '在20210630的年报(含半年报)中，永赢泽利一年定期开放债券基金的债券持仓，其持有最大仓位的债券类型是什么?'
    #
    # result = sql_dataset_generate(sql_seed_pair_path,query01)
    # print(result)

    # 批量读取问题，生成对应的sql
    sql_seed_pair_path = "../G_data/sql_generate_result.xlsx"
    df = pd.read_excel(r'../G_data/intent_recongnise_result.xlsx',sheet_name='未匹配到公司')
    df['sql'] = ''
    print(df)
    for i in range(len(df)):
        print(f'----------------------第{i}行-------------------------')
        try:
            query = df.loc[i]['question']
            result = sql_dataset_generate(sql_seed_pair_path, query)
            df.at[i,'sql'] = result['sql']
        except Exception as e:
            print(f'error: {e}')
    df.to_excel(f'../G_data/sql_lora_result.xlsx', index=False, engine='openpyxl')




