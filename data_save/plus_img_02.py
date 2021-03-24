# 원본 이미지 + 플러스이미지 1,2회차
# 이미지 사이즈 크게!

# 적용하면 문제 있어서 다음 파일로 넘어감

# 현민이는 사랑입니다 ^^
import tensorflow as tf
import numpy as np
from glob import glob
from PIL import Image
import cv2

### train data

train = list()
label = list()
number1 = 1000
number2 = 48

# 이미지 저장하면서 라벨까지 저장
# train
# train - label 
for aa in range(number1):
    img = glob(f'C:/lotte_data/LPD_competition/train/{aa}/*.jpg')
    for j in img :
        temp = cv2.resize(j, (128, 128))
        temp = np.array(temp)
        train.append(temp)
        label.append(aa)

train = np.array(train)
label = np.array(label)

print(train.shape)
print(label.shape)

np.save('C:/lotte_data/npy/1,2sum_data_1000_new.npy', arr=train, allow_pickle=True)
np.save('C:/lotte_data/npy/1,2sum_label_1000_new.npy', arr=label, allow_pickle=True)
print('===== done =====')


