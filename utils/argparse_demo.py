import argparse

# 创建一个参数解析器对象
parser = argparse.ArgumentParser(description='Argument Parser Example')

# 添加命令行参数
parser.add_argument('--input', type=str, help='Input file path')
parser.add_argument('--output', type=str, help='Output file path')
parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')

# 解析命令行参数
args = parser.parse_args()

# 使用解析得到的参数
if args.verbose:
    print('Verbose mode enabled')

if args.input:
    print('Input file path:', args.input)

if args.output:
    print('Output file path:', args.output)

# python argparse_demo.py --input input.txt --output output.txt --verbose
