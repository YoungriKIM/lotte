# 100개 라벨

# 가우시안 블러
# 평준화 비교하기

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import datetime 
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, EfficientNetB2
# 쌤 힌트! x에 적용
from sklearn.preprocessing import PolynomialFeatures

start_now = datetime.datetime.now()
# -----------------------------------------------------------------------------------------------------
# npy로 불러오자 

# origin_save_npy_01
# 적용하기 전에 x_train, y_train, x_test, y_test 저장해 둔 npy를 불러오자
label_size=100

x_train_origin = np.load('D:/lotte_data/npy/equalizeHist_train_data_100.npy')
y_train_origin = np.load('D:/lotte_data/npy/train_label_100.npy')
x_pred = np.load('D:/lotte_data/npy/pred_data_100.npy')

print(x_train_origin.shape)
print(y_train_origin.shape)
print(x_pred.shape)
# (4800, 128, 128, 3)
# (4800,)
# (100, 128, 128, 3)

print(np.max(x_train_origin), np.min(x_train_origin))
# 255 0

# 전처리하자
# 스케일링
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
x_train_origin2 = x_train_origin.astype('float32')/255.
x_train = x_train_origin.astype('float32')/255.
x_pred = x_pred.astype('float32')/255.

# y벡터화
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train_origin)
# (4800,1000)

# 스플릿
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=311)
print('x_train:',x_train.shape, 'x_test:',x_test.shape); print('y_train:',y_train.shape, 'y_test:',y_test.shape); print('x_pred:',x_pred.shape)
# x_train:  (3840, 128, 128, 3) x_test:  (960, 128, 128, 3)
# y_train:  (3840, 100) y_test:  (960, 100)
# x_pred:  (100, 128, 128, 3)

# -----------------------------------------------------------------------------------------------------
# 모델 구성
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, BatchNormalization, Flatten, Input
from tensorflow.keras.layers import Activation
# model.add(Activation('softmax, elu, selu, softplus, softsign, relu, tahn, sigmoid, hard_sigmoid, exponential'))

model = Sequential()
model.add(Conv2D(filters = 128, kernel_size=(2,2), strides=1, padding='same', input_shape=(x_train.shape[1:])))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add (Conv2D(128, 2))
model.add(Dropout(0.2))
model.add (Conv2D(96, 2))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(label_size))
model.add(Activation('softmax'))

# -----------------------------------------------------------------------------------------------------
# 컴파일, 훈련
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from tensorflow.keras.metrics import binary_accuracy, binary_crossentropy,\
                                     categorical_accuracy, categorical_crossentropy,\
                                     sparse_categorical_accuracy,  sparse_categorical_crossentropy,\
                                     top_k_categorical_accuracy, sparse_top_k_categorical_accuracy
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
patience = 16
modelpath='D:/lotte_data/h5/cnn_01.hdf5'
batch_size = 32
stop = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
mc = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, verbose=1)
lr = ReduceLROnPlateau(factor=0.3, patience=int(patience/2), verbose=1)
# model.fit(x_train, y_train, epochs=16, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[stop, mc,lr])

# -----------------------------------------------------------------------------------------------------
# 평가
model = load_model(modelpath)

model.fit(x_train, y_train, epochs=100, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[stop,lr])

result = model.evaluate(x_test, y_test, batch_size=batch_size)
print('loss: ', result[0], '\nacc: ', result[1])
# ===========================================
# 기본
# loss:  0.4723055064678192
# acc:  0.9177083373069763

# 블러
# loss:  0.46173936128616333
# acc:  0.921875

# 평준화
# loss:  0.4713512063026428
# acc:  0.9010416865348816

# 선명하게 filter2d
# loss:  0.3904246985912323
# acc:  0.9229166507720947

# -----------------------------------------------------------------------------------------------------
end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time) 

print('°˖✧(ง •̀ω•́)ง✧˖° 잘한다 잘한다 잘한다~')
