from model_client import api_request
import re

query01 = '华瑞电器股份有限公司成立以来的是否存在重大资产重组？如果存在，该重组是什么'
query02 = '20210930日，一级行业为机械的股票的成交金额合计是多少？取整。'
query03 = '华瑞电器股份有限公司获得多少项国内专利？其中有多上项发明专利？'
query04 = '根据华瑞电器股份有限公司招股意见书，截至2016年12月31日，宁波胜克总资产为多少万元？'
query05 = '华瑞电器股份有限公司，宁波胜克2016年度实现净利润为多少万元？'
query06 = '报告期内，华瑞电器股份有限公司人工成本占主营业务成本的比例分别为多少？'

# 1. 意图识别
url = 'http://1.95.86.245:51000/v1/chat/completions'
model_name = "/home/ruantong/mode/Qwen2-72B-Instruct"
system_prompt = "you are a helpful assistant"

user_prompt = "请识别以下语句的公司名称、问题：\n" \
              f"{query01}" \
              "如果不包含名称，则返回无，不需要返回其他内容"

result = api_request.local_llm_api_request(url, model_name, system_prompt, user_prompt)
response_result = result['response_result']
print(result)
print(response_result)

# 2. 任务分发
pattern_company = r'公司名称：(.+?)\n'
pattern_question = r'问题：(.+)'

match_company = re.search(pattern_company, response_result)
match_question = re.search(pattern_question, response_result)

if match_company and match_question:
    company_name = match_company.group(1).strip()
    question = match_question.group(1).strip()

    # 构建字典
    result_dict = {
        "公司名称": company_name,
        "问题": question
    }

    print(result_dict)
else:
    print("未能匹配到公司名称或问题。")

if result_dict['公司名称'] == '无':
    print('走数据查询模块')
else:
    print('走知识检索模块')


