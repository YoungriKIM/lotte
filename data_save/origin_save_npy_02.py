import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import PIL.Image as pilimg
import datetime 
start_now = datetime.datetime.now()

# 1000개 라벨과 7200개의 pred로 저장해서 진행!

# ---------------------------------------------------------------------
# train 데이터 불러오기
train = list()
label = list()
number1 = 1000
number2 = 48

# 이미지 저장하면서 라벨까지 저장
# train
# train - label 
for aa in range(number1):
    for a in range(number2):
        temp = cv2.imread('C:/lotte_data/LPD_competition/train/' + str(aa)+ '/' + str(a) + '.jpg')
        temp = cv2.resize(temp, (128, 128))
        temp = np.array(temp)
        train.append(temp)
        label.append(aa)

# np.array로 바꿔서 쉐잎 확인
train_data_1000 = np.array(train)
train_label_1000 = np.array(label)
print(train_data_1000.shape)
print(train_label_1000.shape)

# ---------------------------------------------------------------------
# pred 저장
pred = list()
number3 = 72000
for b in range(number3):
    temp = cv2.imread('C:/lotte_data/LPD_competition/test/' + str(b) + '.jpg')
    temp = cv2.resize(temp, (128, 128))
    temp = np.array(temp)
    pred.append(temp)

# np.array로 바꿔서 쉐잎 확인
pred_data_72000 = np.array(pred)
print(pred_data_72000.shape)


# ---------------------------------------------------------------------
# npy로 저장

np.save('C:/lotte_data/LPD_competition/npy/train_data_1000.npy', arr = train_data_1000)
np.save('C:/lotte_data/LPD_competition/npy/train_label_1000.npy', arr = train_label_1000)
np.save('C:/lotte_data/LPD_competition/npy/pred_data_72000.npy', arr = pred_data_72000)
print('===== done =====')


end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time) 