# 100개 라벨
# 100개의 pred 이미지로 진행
# cnn만들기 

# 그래프 그려서 acc, loss 확인!!
# 파이토치에 있는 라벨별 에큐러시 확인 만들기!
# 스프레드 시트에 정리해라~
# 1) cnn 기본 모델 - 100개
# 1-2) cnn 튜닝 - 100개에서 조금씩 늘려서 튜닝
# 1-5) cnn - 1000개 라벨 (해당 파일)



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
label_size=1000

x_train_origin = np.load('C:/lotte_data/LPD_competition/npy/train_data_1000.npy')
y_train_origin = np.load('C:/lotte_data/LPD_competition/npy/train_label_1000.npy')
x_pred = np.load('C:/lotte_data/LPD_competition/npy/pred_data_72000.npy')

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
# x_pred = x_pred.astype('float32')/255.

# y벡터화
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train_origin)
# (4800,1000)

# 스플릿
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=311)
print('x_train:',x_train.shape, 'x_test:',x_test.shape); print('y_train:',y_train.shape, 'y_test:',y_test.shape); print('x_pred:',x_pred.shape)


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
modelpath='C:/lotte_data/h5/cnn_01-5.hdf5'
batch_size = 32
stop = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
mc = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, verbose=1)
lr = ReduceLROnPlateau(factor=0.3, patience=int(patience/2), verbose=1)
model.fit(x_train, y_train, epochs=256, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[stop, mc,lr])

# -----------------------------------------------------------------------------------------------------
# 평가
model = load_model(modelpath)

result = model.evaluate(x_test, y_test, batch_size=batch_size)
print('loss: ', result[0], '\nacc: ', result[1])
# ===========================================
# loss:  0.35801950097084045
# acc:  0.9208333492279053
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
submission = pd.read_csv('C:/lotte_data/LPD_competition/sample.csv', index_col=0)

# npy 파일용
# y_pred = model.predict(x_pred)
# submission['prediction'][0:x_pred.shape[0]] = np.argmax(y_pred, axis = 1)
# submission.to_csv('D:/lotte_data/LPD_competition/sub/sub_cnn_01.csv',index=True)
# print('==== csv save done ====')

# 이미지 불러와서용
y_pred =[]
for imgnumber in range(x_pred.shape[0]):
    pred_img = cv2.imread('C:/lotte_data/LPD_competition/test/'+ str(imgnumber) + '.jpg')
    pred_img = cv2.resize(pred_img, (128, 128))
    pred_img = pred_img.reshape(1, 128, 128, 3)/255.
    pred_img = np.array(pred_img)
    temp = np.argmax(model.predict(pred_img))
    y_pred.append(temp)
    if imgnumber % 100 == 99:
        print(str(imgnumber)+'번째 이미지 작업 완료')
y_pred = np.array(y_pred)
print(y_pred.shape)
submission['prediction'][:x_pred.shape[0]] = y_pred
submission.to_csv('C:/lotte_data/LPD_competition/sub/sub_cnn_01-5.csv',index=True)
print('==== csv save done ====')

# -----------------------------------------------------------------------------------------------------
end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time) 

print('°˖✧(ง •̀ω•́)ง✧˖° 잘한다 잘한다 잘한다~')

