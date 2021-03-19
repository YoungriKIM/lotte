import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras import Sequential
from keras.layers import *
import cv2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
'''
test_img = load_img('C:/lotte_data/LPD_competition/train/14/7.jpg')

plt.imshow(test_img)
plt.show()

# 어레이로 바꿔 크기 확인
arr_img = img_to_array(test_img)
print(arr_img.shape)
# (256, 256, 3)

arr_img = arr_img.reshape(1, arr_img.shape[0]*arr_img.shape[1]*arr_img.shape[2])


# 폴리노미날피텨 적용
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_img = poly.fit_transform(arr_img)

print(poly_img.shape)
plt.imshow(poly_img)
plt.show()

'''
# --------------------------------------

test_img = cv2.imread('C:/lotte_data/LPD_competition/train/87/1.jpg', cv2.IMREAD_GRAYSCALE)
test_img = cv2.resize(test_img, (32, 32))/255.

arr_img = img_to_array(test_img)
print(arr_img.shape)
# 32,32

arr_img = arr_img.reshape(32, 32*1)

print(arr_img.shape)

print('2차원이다\n: ', arr_img)

# 폴리노미날피텨 적용
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_img = poly.fit_transform(arr_img)

print(poly_img.shape)
# (32, 4752)
# plt.imshow(poly_img)
# plt.show()

# (32, 32, 3)
# # 2 차원 만들기
# (32, 96)
# # 폴리
# (32, 4752)


print('폴리이다\n: ', poly_img)

poly_img = poly_img.reshape(32, 560, 1)



plt.figure(figsize=(16,10))

plt.subplot(2,1,1)
plt.imshow(test_img)

plt.subplot(2,1,2)
plt.imshow(poly_img)

plt.show()

# 질문: 그럼 용량이 너무너무 커지는데 어떡해요?
# 샘왈: 그것만 쓰면 안되고 다른 거랑 엮어야지!
# pca인가 ?