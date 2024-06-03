from paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt
from PIL import Image


# 初始化OCR模型
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

# 指定图像路径
img_path = r'D:\python_program\ModelAI\datasets\Image/1716947141863.jpg'

# 进行OCR识别
result = ocr.ocr(img_path, cls=True)

# 打印识别结果
for line in result:
    print(line)

# 可视化识别结果
image = Image.open(img_path).convert('RGB')
boxes = [elements[0] for elements in line for line in result]
txts = [elements[1][0] for elements in line for line in result]
scores = [elements[1][1] for elements in line for line in result]

im_show = draw_ocr(image, boxes, txts, scores, font_path='path/to/simfang.ttf')
im_show = Image.fromarray(im_show)

# 显示图像
plt.imshow(im_show)
plt.axis('off')
plt.show()

