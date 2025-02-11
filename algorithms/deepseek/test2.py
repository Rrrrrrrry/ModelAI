from openai import OpenAI

api_key = 'sk-5911baca43844734a668ce0f41104d0f'


# 定义一个deepseek类，支持流式操作和非流式操作
# 接收当前问题和历史问题，返回回答的结果
# 增加异步
class DeepseekApi:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    def chat(self, message, history_message):
        messages = [{"role": "user", "content": message}]
        if history_message:
            messages = history_message + messages
        print(f"messages{messages}")
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        return response

if __name__ == '__main__':
    api = DeepseekApi(api_key)
    message = 'Strawberry里有几个r'
    history =[{"role": "user", "content": '你是一个数据专家'}]
    response1 = api.chat(message, history)
    print(response1)
    # message = 'Strawberry里有几个r'
    # response2 = api.chat(message, response1)
    # print(response2)


# client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
#
# # Round 1
# messages = [{"role": "user", "content": "What's the highest mountain in the world?"}]
# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=messages
# )
#
# messages.append(response.choices[0].message)
# print(f"Messages Round 1: {messages}")

# # Round 2
# messages.append({"role": "user", "content": "What is the second?"})
# print(f"messages:{messages}")
# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=messages
# )
#
# messages.append(response.choices[0].message)
# print(f"Messages Round 2: {messages}")