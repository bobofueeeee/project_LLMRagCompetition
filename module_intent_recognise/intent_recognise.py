from model_client import api_request
import json
from prompt.intent_recognise_prompt import intent_recog_prompt
from untils.json_toolkit import extract_jsonstr
import pandas as pd


def intent_recongnise(question,url,model_name,system_prompt):
    # 1. 意图识别
    user_prompt = question
    result = api_request.local_llm_api_request(url, model_name, system_prompt, user_prompt)
    response_result = result['response_result']
    # print(result)
    print(response_result)
    result_dict = extract_jsonstr(response_result)
    print(result_dict)
    return result_dict


def question_split():
    file_path = '../data/question.json'
    url = 'http://1.95.86.245:51000/v1/chat/completions'
    model_name = "/home/ruantong/mode/Qwen2-72B-Instruct"
    system_prompt = intent_recog_prompt
    df_result = pd.DataFrame(columns=['question', 'label','company','keyquestion'])
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
            recongnise_result = intent_recongnise(question, url, model_name, system_prompt)
            df_result.loc[i] = [recongnise_result['question'], recongnise_result['label'], recongnise_result['company'], recongnise_result['keyquestion']]
    df_result.to_excel(f'../data/intent_recongnise_result.xlsx', index=False, engine='openpyxl')
    return df_result

if __name__ == '__main__':
    question_split()
