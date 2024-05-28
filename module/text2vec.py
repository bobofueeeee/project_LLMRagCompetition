# 1. 读取json
import jsonlines

def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r" ) as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content


def text2vec(path):
    content = []
    qa_json = read_jsonl(path)
    # print(qa_json)

    # for i in range(len(qa_json)):
    #     print(i)
    #     print(qa_json[i]["question"])

    # 2. json内容转换为向量
    from transformers import BertModel, BertTokenizer
    # 加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained(r"D:\wk\bert-base-uncased")
    model = BertModel.from_pretrained(r"D:\wk\bert-base-uncased")

    # 分词
    dic = {}
    for i in range(len(qa_json)):
        question_input = tokenizer(qa_json[i]["question"], return_tensors="pt", padding=True, truncation=True)
        answer_input = tokenizer(qa_json[i]["answer"], return_tensors="pt", padding=True, truncation=True)
        question_output = model(**question_input)
        answer_output = model(**answer_input)

        question_last_hidden_states = question_output.last_hidden_state[:, 0, :]
        answer_last_hidden_states = answer_output.last_hidden_state[:, 0, :]
        content.append({"id":i
                        ,"question_tensor":question_last_hidden_states
                        ,"answer_tensor":answer_last_hidden_states
                        ,"question":qa_json[i]["question"]
                        ,"answer":qa_json[i]["answer"]
                        })

    return content

def question2vec(question):
    # print(qa_json)

    # for i in range(len(qa_json)):
    #     print(i)
    #     print(qa_json[i]["question"])

    # 2. json内容转换为向量
    from transformers import BertModel, BertTokenizer
    # 加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained(r"D:\wk\bert-base-uncased")
    model = BertModel.from_pretrained(r"D:\wk\bert-base-uncased")

    # 分词
    question_input = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    question_output = model(**question_input)
    question_last_hidden_states = question_output.last_hidden_state[:, 0, :]

    return question_last_hidden_states


# path = "../data/qa.json"
# print(text2vec(path))







