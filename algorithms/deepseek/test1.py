import datetime
from typing import List, Union
from openai import OpenAI

SYSTEM_PROMPT = """
你好，你是deepseekAi助手，请你帮助解决用户问题。
"""


class DeepSeekApiVllmAsync:

    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

    def make_chat_params(self, prompt: str, history: List, system_prompt: str = None, is_stream: bool = False):
        if not system_prompt:
            system_prompt = SYSTEM_PROMPT.format(time=datetime.datetime.now())

        if not history:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [{"role": "system", "content": system_prompt}] + history + [
                {"role": "user", "content": prompt}]
        return messages

    async def chat(self, prompt: str, history: List, system_prompt: str = None, extra_body: dict = None):

        if extra_body:
            new_items = {key: value['description'] for key, value in extra_body['guided_json']['properties'].items()}

            system_prompt = system_prompt + "\n最终以json格式返回结果\njson格式的字段及其描述如下：\n" + str(new_items)

        messages = self.make_chat_params(prompt, history, system_prompt, is_stream=False)
        if extra_body:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                response_format={
                    'type': 'json_object'
                }
            )
        else:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
            )
        return response.choices[0].message.content

    async def stream_chat(self, prompt: str, history: List, system_prompt=None):
        messages = self.make_chat_params(prompt, history, system_prompt, is_stream=False)

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True
        )
        for item in response:
            yield item
        yield "DONE"


if __name__ == '__main__':
    import asyncio

    api_key = 'sk-dc1b58b458a04e8cb0de60c514e82943'
    async def run_stream(prompt, history):
        # 异步流式返回
        api = DeepSeekApiVllmAsync(api_key)
        result = api.stream_chat(prompt, history)
        print("start response")
        async for line in result:
            print(line)


    async def run_chat(prompt, history):
        # 异步流式返回
        api = DeepSeekApiVllmAsync(api_key)
        result = await api.chat(prompt, history)
        print(result)



    asyncio.run(run_stream("你好", []))
