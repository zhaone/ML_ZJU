from pathlib import Path
import pytesseract

from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'D:/Tesseract-OCR/tesseract.exe'

captchas_path = Path('./captchas')
cap_list = captchas_path.glob('*.png')
with open('./labels.txt', 'w') as f:
    for cap in cap_list:
        text = pytesseract.image_to_string(Image.open(cap))
        f.write(text+'\n')
