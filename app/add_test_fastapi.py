from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union

app = FastAPI()

# 请求体数据模型
class AddRequest(BaseModel):
    a: Union[int, float, str]
    b: Union[int, float, str]

# 加法逻辑
def add_data(a, b):
    try:
        a_num = float(a)
        b_num = float(b)
        return a_num + b_num
    except ValueError:
        return str(a) + str(b)

@app.post("/add")
def add_api(data: AddRequest):
    result = add_data(data.a, data.b)
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(app)
    # http://127.0.0.1:8000/docs  查看文档
    # POST->Body->raw->{"a":10, "b":20}->Send