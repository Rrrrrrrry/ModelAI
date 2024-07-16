import os
key = "sk-5c2f6583a9f1450eab5c5875a935a6dc"
os.environ['DASHSCOPE_API_KEY'] = key
import random
from http import HTTPStatus
from dashscope import Generation


def call_with_messages(prompt):
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}]
    response = Generation.call(model="qwen-turbo",
                               messages=messages,
                               # 设置随机数种子seed，如果没有设置，则随机数种子默认为1234
                               seed=random.randint(1, 10000),
                               # 将输出设置为"message"格式
                               result_format='message')
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))


if __name__ == '__main__':
    prompt = "介绍下你自己"
    call_with_messages(prompt)