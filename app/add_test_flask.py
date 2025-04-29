from flask import Flask, request, jsonify

app = Flask(__name__)


# 加法功能
def add_data(a, b):
    # 自动判断类型进行加法：支持整数、浮点数和字符串
    try:
        # 尝试转换为数字
        a_num = float(a)
        b_num = float(b)
        return a_num + b_num
    except ValueError:
        # 如果不能转为数字，则当字符串拼接
        return str(a) + str(b)


@app.route('/add', methods=['POST'])
def add_api():
    data = request.get_json()  # 获取 JSON 数据

    # 确保包含两个参数
    if 'a' not in data or 'b' not in data:
        return jsonify({'error': 'Missing parameters: "a" and "b" are required'}), 400

    result = add_data(data['a'], data['b'])

    # 如果是数值类型，尝试保持原类型返回
    if isinstance(result, (int, float)):
        return jsonify({'result': result})
    else:
        return jsonify({'result': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
    # POST->Body->raw->{"a":10, "b":20}->Send


