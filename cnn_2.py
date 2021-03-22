# 100개 라벨
# 100개의 pred 이미지로 진행
# cnn만들기 

# 그래프 그려서 acc, loss 확인!!
# 파이토치에 있는 라벨별 에큐러시 확인 만들기!
# 스프레드 시트에 정리해라~
# 1) cnn 기본 모델 - 100개
# 1-2) cnn 튜닝 - 100개에서 조금씩 늘려서 튜닝
# 2) 100개 라벨 cnn + 폴리몰리  (해당 파일) # 계속 터져서 빠쵸서 못함!


import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
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

x_train_origin = np.load('D:/lotte_data/npy/train_data_100.npy')
y_train_origin = np.load('D:/lotte_data/npy/train_label_100.npy')
x_pred = np.load('D:/lotte_data/npy/pred_data_100.npy')

# 폴리돌리기 너무 커서 이미지 사이즈 줄이기
x_train_origin = np.array(tf.image.resize(x_train_origin,(32,32)))
x_pred = np.array(tf.image.resize(x_pred,(32,32)))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

x_train_origin = rgb2gray(x_train_origin)
x_pred = rgb2gray(x_pred)

print(x_train_origin.shape)
print(y_train_origin.shape)
print(x_pred.shape)
# (4800, 32,32)
# (4800,)
# (100, 32, 32)

x_train_origin = x_train_origin.reshape(4800, 32*32*1)
x_pred = x_pred.reshape(100, 32*32*1)

# 전처리하자
# 스케일링
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
minmax = MinMaxScaler()
minmax.fit(x_train_origin)
x_train_origin2 = minmax.fit_transform(x_train_origin)
x_train = minmax.fit_transform(x_train_origin)
x_pred = minmax.fit_transform(x_pred)

# x_train_origin2 = x_train_origin.astype('float32')/255.
# x_train = x_train_origin.astype('float32')/255.
# x_pred = x_pred.astype('float32')/255.

print(np.max(x_train), np.min(x_train))
# 255 0

# 폴리노미널용 특성 강하게
x_train_origin2 = np.where((x_train_origin2 <= 160/255.), 0, x_train_origin2)
x_train = np.where((x_train_origin2 <= 160/255.), 0, x_train_origin2)
x_pred = np.where((x_pred <= 160/255.), 0, x_pred)

# 폴리노미널 x에 적용
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)

poly_x_train = poly.fit_transform(x_train)
poly_x_pred = poly.fit_transform(x_pred)
# MemoryError: Unable to allocate 150. GiB for an array with shape (4800, 8394752) and data type float32
# 폴리몰리야...이것도 터지면 어떡하자는 거니~~~~?????

print(poly_x_train.shape)
print(poly_x_pred.shape)
# (4800, 525824)
# (100, 525824)

poly_x_train = poly_x_train.reshape(4800, 832, 632, 1)
poly_x_pred = poly_x_pred.reshape(100, 832, 632, 1)

# y벡터화
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train_origin)
# (4800,1000)

# 스플릿
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(poly_x_train, y_train, test_size=0.2, shuffle=True, random_state=311)
print('x_train:',x_train.shape, 'x_test:',x_test.shape); print('y_train:',y_train.shape, 'y_test:',y_test.shape); print('poly_x_pred:',poly_x_pred.shape)


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
model.add(MaxPooling2D(pool_size=2))
model.add (Conv2D(96, 2))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(100))
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
patience = 8
modelpath='D:/lotte_data/h5/cnn_02.hdf5'
batch_size = 32
stop = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
mc = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, verbose=1)
lr = ReduceLROnPlateau(factor=0.5, patience=int(patience/2), verbose=1)
model.fit(x_train, y_train, epochs=100, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[stop, mc,lr])

# -----------------------------------------------------------------------------------------------------
# 최고 모델로 평가
model = load_model(modelpath)

result = model.evaluate(x_test, y_test, batch_size=batch_size)
print('loss: ', result[0], '\nacc: ', result[1])
# ===========================================
# ing

'''
# -----------------------------------------------------------------------------------------------------
# 최고 모델로 100가지 애큐러시 확인
for i in range(label_size):
    class_correct = 0
    for j in range(48):
        outputs = model.predict(x_train_origin2[(i*48)+j].reshape(1,128,128,3))
        outputs = np.argmax(outputs)
        if outputs == y_train_origin[(i*48)+j]: class_correct += 1
    each_acc = float(class_correct/48)
    if each_acc <= 0.9: print(str(i)+'번 라벨 acc: ', each_acc)
    else: print(str(i)+'번 라벨 acc: pass')
'''

# -----------------------------------------------------------------------------------------------------
# 예측, 저장
submission = pd.read_csv('D:/lotte_data/LPD_competition/sample.csv', index_col=0)
# npy 파일용
y_pred = model.predict(x_pred)

submission['prediction'][0:label_size] = np.argmax(y_pred, axis = 1)
submission.to_csv('D:/lotte_data/LPD_competition/sub/sub_cnn_01.csv',index=True)
print('==== csv save done ====')


# -----------------------------------------------------------------------------------------------------
end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time) 

print('°˖✧(ง •̀ω•́)ง✧˖° 잘한다 잘한다 잘한다~')

# MemoryError: Unable to allocate 6.02 GiB for an array with shape (3072, 832, 632, 1) and data type float32
# 안 해 칵 퉤