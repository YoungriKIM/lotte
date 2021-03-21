# 100개 라벨
# 100개의 pred 이미지로 진행
# cnn만들기 

# 그래프 그려서 acc, loss 확인!!
# 파이토치에 있는 라벨별 에큐러시 확인 만들기!
# 스프레드 시트에 정리해라~
# 1) cnn 기본 모델 - 100개 (해당 파일)
# 1-2) cnn 튜닝 - 100개에서 조금씩 늘려서 튜닝
# 2) cnn 모델 + 폴리몰리
# 3) cnn 모델 + 폴리몰리 + pca
# 4) rnn 모델
# 5) dnn 모델
# 6) 100개씩 돌린것 10개 앙상블! > 200개로 5개 > 10개로 100개 등 해보기


import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, BatchNormalization, Flatten, Input
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import pandas as pd
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, EfficientNetB2
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# npy로 불러오자 -----------------------------------------------------------------------------------------------------

# 적용하기 전에 x_train, y_train, x_test, y_test 저장해 둔 npy를 불러오자
x_train = np.load('D:/lotte_data/npy/train_data_100.npy')
y_train = np.load('D:/lotte_data/npy/train_label_100.npy')
x_pred = np.load('D:/lotte_data/npy/pred_data_100.npy')

print(x_train.shape)
print(y_train.shape)
print(x_pred.shape)
# (4800, 128, 128, 3)
# (4800,)
# (100, 128, 128, 3)

print(np.max(x_train), np.min(x_train))
# 255 0

# 전처리하자
# 스케일링
x_train = x_train/255.
x_pred = x_pred/255.

# y벡터화
y_train = to_categorical(y_train)
# (4800,1000)

# 스플릿
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=311)
print('x_train:',x_train.shape, 'x_test:',x_test.shape); print('y_train:',y_train.shape, 'y_test:',y_test.shape); print('x_pred:',x_pred.shape)
# x_train:  (3840, 128, 128, 3) x_test:  (960, 128, 128, 3)
# y_train:  (3840, 100) y_test:  (960, 100)
# x_pred:  (100, 128, 128, 3)




print('°˖✧(ง •̀ω•́)ง✧˖° 잘한다 잘한다 잘한다~')