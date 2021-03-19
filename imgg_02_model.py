# image generator 쓰면 순서 뒤죽박죽 됨  for문으로 만들 것

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, BatchNormalization, Flatten, Input
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import pandas as pd
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, EfficientNetB2

# npy로 불러오자 -----------------------------------------------------------------------------------------------------

# 적용하기 전에 x_train, y_train, x_test, y_test 저장해 둔 npy를 불러오자
x_train = np.load('D:/lotte/npy/just_scale_lpd_train_x.npy')
y_train = np.load('D:/lotte/npy/just_scale_lpd_train_y.npy')
x_pred = np.load('D:/lotte/npy/just_scale_lpd_pred_x.npy')

# 전이학습용 전처리
x_train = tf.keras.applications.resnet50.preprocess_input(x_train)
x_pred = tf.keras.applications.resnet50.preprocess_input(x_pred)

# print(x_train.shape, y_train.shape)
# print(x_pred.shape)
# print(np.max(x_train), np.min(x_train))
# (48000, 150, 150, 3) (48000, 1000)
# (72000, 150, 150, 3)
# 1.0 0.0

# 스팔스 쓰기 전에 y 열을 하나로 -----------------------------
y_train = np.argmax(y_train, axis=1)

# 전처리 동재 바부 -----------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=311)


# 훈련을 시켜보자! 모델구성 -----------------------------------------------------------------------------------------------------

# 전이학습 사용
# VGG-16
# ResNet50
# Inceptionv3
# EfficientNet
input_tensor = Input(shape=(150, 150, 3))
apl = ResNet50(weights='imagenet', include_top=False, input_tensor = input_tensor)
apl.trainable = False

model = Sequential()
model.add(apl)
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1000, activation='softmax'))

# 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-5, epsilon=None), metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
stop = EarlyStopping(monitor='val_acc', patience=8, mode='auto')
lr = ReduceLROnPlateau(monitor='val_acc', factor=0.3, patience=16, mode='max')
filepath = ('D:/lotte/mc/imgg_02.hdf5')
mc = ModelCheckpoint(filepath=filepath, save_best_only=True, verbose=1)

model.fit(x_train, y_train, epochs=3, batch_size=16, verbose=1, validation_data=(x_test, y_test), callbacks=[stop,lr, mc])

#  ----------------------------------------------------------------------------------------------
loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

#  ----------------------------------------------------------------------------------------------
# 예측까지 하자
model = load_model('D:/lotte/mc/imgg_02.hdf5')
result = model.predict(x_pred)

print(result.shape)
sub = pd.read_csv('D:/lotte/LPD_competition/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('D:/lotte/submit/answer_01.csv',index=False)
# =====================================================

# answer_01.csv
# sparse 사용, apl.trainable = False, batch_size=16

