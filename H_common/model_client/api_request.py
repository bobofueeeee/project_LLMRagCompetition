# -*- coding: utf-8 -*-

# 输入：query，模型
# 输出：answer

import requests
import json
from H_common.untils import OpenAiAzureToolkit


def local_llm_api_request(url,model_name,system_prompt,user_prompt):
    # 准备POST请求的数据
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": user_prompt
            },
            {
                "role": "system",
                "content": system_prompt
            }
        ],
        "max_tokens": 800,
        "top_k": -1,
        "top_p": 1,
        "temperature": 0.7,
        "ignore_eos": False,
        "stream": False
    }
    response = requests.post(url, json=data)
    result_all = json.loads(response.text)
    result = result_all['choices'][0]['message']['content']
    result_dic = {"request_data":data,"response_result":result}
    return result_dic

def openai_api_request(system_prompt,user_prompt):
    openai_client = OpenAiAzureToolkit()

    data = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    result_all = openai_client.client_infer(data)
    result = result_all.choices[0].message.content
    # print(result_all)
    # print(result_all.choices[0].message.content)
    return result




if __name__ == '__main__':
    # local_llm_api_request 测试用力
    # 替换以下URL为你想要请求的API的URL
    url = 'http://1.95.86.245:51000/v1/chat/completions'
    model_name = "/home/ruantong/mode/Qwen2-72B-Instruct"
    system_prompt = "you are a helpful assistant"
    user_prompt = "你好"
    result = local_llm_api_request(url,model_name,system_prompt,user_prompt)
    response_result = result['response_result']
    print(result)
    print(response_result)

    # ------------------------------------------------------------------------
    # _openai_api_request 测试用例
    system_prompt = "you are a helpful assistant"
    user_prompt = "你好"
    result = openai_api_request(system_prompt,user_prompt)
    print(result)
