import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import PIL.Image as pilimg

# x train 데이터 불러오기 -------------------------------------
df_pix = []
number = 50000

for a in np.arange(0, number):             
    file_path = '../Users/Admin/Desktop/dacon/dacon12/dirty_mnist_2nd/' + str(a).zfill(5) + '.png'
    image = pilimg.open(file_path)
    pix = np.array(image)
    pix = pd.DataFrame(pix)
    df_pix.append(pix)

x_df = pd.concat(df_pix)
x_df = x_df.values
# 원래 사이즈는 (256,256)

print(type(x_df))  # <class 'numpy.ndarray'>
print(x_df.shape)  # (25600, 256)

image2 = pilimg.open('../Users/Admin/Desktop/비트 숙제/0210/yeona.jpg')
pix2 = image2.resize((56,56))

# ---------------------------------------------------------------------
# train 데이터 불러오기
df_pix = []
number = 48

# for a in np.arange(0, number):
#     file_path = 'D:/lotte/LPD_competition/train/1' str(a).