# 100개 라벨
# 100개의 pred 이미지로 진행
# cnn만들기 

# 파이토치 때려쳐ㅎ알아서한다칵퉤

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

# npy로 불러오자 -----------------------------------------------------------------------------------------------------

# 적용하기 전에 x_train, y_train, x_test, y_test 저장해 둔 npy를 불러오자
x_train = np.load('C:/lotte_data/LPD_competition/npy/train_data_100.npy')
y_train = np.load('C:/lotte_data/LPD_competition/npy/train_label_100.npy')
x_pred = np.load('C:/lotte_data/LPD_competition/npy/pred_data_100.npy')

print(x_train.shape)
print(y_train.shape)
print(x_pred.shape)
# (4800, 128, 128, 3)
# (4800,)
# (100, 128, 128, 3)

print(np.max(x_train), np.min(x_train))
# 255 0

# 전처리하지
# 스케일링
x_train = x_train/255.
x_pred = x_pred/255.

# y벡터화
y_train = to_categorical(y_train)

