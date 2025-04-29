import tkinter as tk
from tkinter import filedialog, messagebox
import shutil
import os

def select_file():
    # 打开文件对话框让用户选择文件
    filepath = filedialog.askopenfilename(
        title="选择文件",
        filetypes=(("所有文件", "*.*"), ("文本文件", "*.txt"))
    )
    print(f"filepath{filepath}")
    if filepath:
        process_file(filepath)

def process_file(filepath):
    try:
        # 假设处理就是复制文件到新位置
        destination = os.path.join(os.getcwd(), "processed_files", os.path.basename(filepath))
        if not os.path.exists(os.path.dirname(destination)):
            os.makedirs(os.path.dirname(destination))
        shutil.copy(filepath, destination)
        messagebox.showinfo("完成", f"文件已处理并保存到 {destination}")
    except Exception as e:
        messagebox.showerror("错误", str(e))

# 创建主窗口
root = tk.Tk()
root.title("文件处理器")
root.geometry("300x150")

# 添加按钮以选择文件
upload_button = tk.Button(root, text="上传文件", command=select_file)
upload_button.pack(pady=20)

# 启动事件循环
root.mainloop()


"""
打包exe命令
D:
cd D:\python_program\ModelAI\app
# pip install pyinstaller
# pyinstaller --onefile --windowed deal_file_exe_demo.py
pyinstaller --onefile --windowed --distpath D:\python_program\ModelAI\app\exe_demo deal_file_exe_demo.py

生成的bulid文件夹为中间文件夹，可删除
.spec文件是一个 Python 脚本，包含了打包过程中使用的各种参数和配置。
通过编辑 .spec 文件，你可以更精细地控制打包过程。下次打包用pyinstaller file_processor.spec即可
可以在打包时修改bulid和spec的保存路径
pyinstaller --onefile --windowed --workpath D:\python_program\ModelAI\app\exe_demo\build --specpath D:\python_program\ModelAI\app\exe_demo --distpath D:\python_program\ModelAI\app\exe_demo deal_file_exe_demo.py

--onefile：将整个项目打包成一个单独的 .exe 文件（所有依赖都压缩到一个文件中）
--windowed：仅 Windows 和 macOS
--distpath: 控制输出路径；PyInstaller 会在当前目录下生成 dist/ 文件夹
--noconfirm：如果目标目录（如 build/, dist/）已经存在，不会提示是否覆盖，直接替换。在自动化构建脚本中有用
--workpath：设置 PyInstaller 构建过程中的临时工作目录 （用来存放中间文件）。
"""