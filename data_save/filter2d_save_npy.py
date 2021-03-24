import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import PIL.Image as pilimg
import datetime 
start_now = datetime.datetime.now()

# 1000개 라벨의 train 만 선명하게 해서 저장용 

# ---------------------------------------------------------------------
# train 데이터 불러오기
train = list()
label = list()
number1 = 1000
number2 = 48

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# 이미지 저장하면서 라벨까지 저장
# train
# train - label 
for aa in range(number1):
    for a in range(number2):
        temp = cv2.imread('D:/lotte_data/LPD_competition/train/' + str(aa)+ '/' + str(a) + '.jpg')
        temp = cv2.resize(temp, (128, 128))
        temp = cv2.filter2D(temp, -1, kernel)
        temp = np.array(temp)
        train.append(temp)

# np.array로 바꿔서 쉐잎 확인
filter2d_save_npy = np.array(train)
print(filter2d_save_npy.shape)

# ---------------------------------------------------------------------
# npy로 저장

np.save('D:/lotte_data/npy/only_filter2d_data_1000.npy', arr = filter2d_save_npy)
print('===== done =====')


end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time) 

