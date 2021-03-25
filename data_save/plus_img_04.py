# 128 + 이미지추가수집 2회차까지

import tensorflow as tf
import numpy as np
from glob import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt

train = list()
label = list()

for aa in range(1000):
    img = glob(f'C:/lotte_data/LPD_competition/train/{aa}/*.jpg')
    for j in img :
        temp = Image.open(j)
        temp = temp.convert("RGB")
        temp = np.array(temp)
        temp = cv2.resize(temp, (128, 128))
        train.append(temp)
        label.append(aa)

train = np.array(train)
label = np.array(label)

print(train.shape)
print(label.shape)

# ---------------------------------------------------------------------
# npy로 저장
np.save('C:/lotte_data/npy/brandnew_1000plus_data.npy', arr = train)
np.save('C:/lotte_data/npy/brandnew_1000plus_label.npy', arr = label)
print('===== done =====')



