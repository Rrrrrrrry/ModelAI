from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'D:\software\software\tesseract-ocr-w64-setup-5.3.4.20240503.exe'

img = Image.open(r'D:\python_program\ModelAI\datasets\Image/1716947141863.jpg')
img = img.convert('L')  # 转为灰度图像
img = img.filter(ImageFilter.SHARPEN)  # 增强锐度
img = img.point(lambda x: 0 if x < 128 else 255)  # 二值化

text = pytesseract.image_to_string(img, lang='eng')
print(text)
