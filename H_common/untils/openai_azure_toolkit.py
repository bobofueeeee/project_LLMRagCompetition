# 工具包1：建立openai的访问client
# 工具包2：进行推理
from openai import AzureOpenAI

api_key01 = "6d7b78aa99494cb98e2b2b8ae772b6cf"
api_key02 = "0e42901a140a4dc6a28949496d3a7b89"
api_key = api_key01
api_version = "2024-02-01"
azure_endpoint = "https://gdtrgdopai.openai.azure.com/"
azure_deployment = "USA4TURBO"
model_name = "gpt4.0-tubro"


class OpenAiAzureToolkit:

    def __init__(self):
        # self.api_key = os.getenv('api_key')
        # self.api_version = os.getenv('api_version')
        # self.azure_endpoint = os.getenv('azure_endpoint')
        # self.azure_deployment = os.getenv('azure_deployment')
        # self.azure_model = os.getenv('azure_model')
        self.api_key = api_key
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.azure_model = model_name

        self.client = AzureOpenAI(
            api_key=api_key02,
            api_version="2024-02-01",
            azure_endpoint="https://gdtrgdopai.openai.azure.com/",
            azure_deployment="USA4TURBO")

    def client_infer(self, data):
        response = self.client.chat.completions.create(
            model=self.azure_model,  # model = "deployment_name".
            messages=data
        )
        return response




if __name__ == '__main__':
    data = [
        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
        {"role": "user", "content": "你好"}
    ]
    openai_client = OpenAiAzureToolkit()
    result = openai_client.client_infer(data)
    print(result)
    print(result.choices[0].message.content)
