# 천사들이 내려준 코드 ^^

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import datetime 
from tensorflow.keras.applications import VGG16, ResNet101, InceptionV3, EfficientNetB2

start_now = datetime.datetime.now()
# -----------------------------------------------------------------------------------------------------
# npy로 불러오자 
label_size=1000

x_train_origin = np.load('D:/lotte_data/npy/1,2sum_data_1000.npy')
y_train_origin = np.load('D:/lotte_data/npy/1,2sum_label_1000.npy')

# 전이학습용 전처리
from tensorflow.keras.applications.efficientnet import preprocess_input
x_train_origin2 = preprocess_input(x_train_origin)
x_train = preprocess_input(x_train_origin)

# y벡터화
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train_origin)

# 스플릿
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=311)
print('x_train:',x_train.shape, 'x_test:',x_test.shape); print('y_train:',y_train.shape, 'y_test:',y_test.shape)

# -----------------------------------------------------------------------------------------------------
# 모델 구성
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, BatchNormalization, Flatten, Input
from tensorflow.keras.layers import Activation, ZeroPadding2D, Concatenate, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2


mobile = EfficientNetB2(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
mobile.trainable = True
a = mobile.output
a = GlobalAveragePooling2D()(a)
a = Flatten()(a)
a = Dense(4048, activation= 'swish')(a)
a = Dropout(0.3)(a)
a = Dense(1000, activation= 'softmax')(a)

model = Model(inputs = mobile.input, outputs = a)
# ----------------------------------------------------------------------------------------------------
# 컴파일, 훈련
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

optimizer=SGD(lr=0.1, momentum=0.9)

from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
patience = 16
modelpath='D:/lotte_data/h5/syhj_angel.hdf5'
batch_size = 8
stop = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
mc = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, verbose=1)
lr = ReduceLROnPlateau(factor=0.5, patience=int(patience/2), verbose=1)
model.fit(x_train, y_train, epochs=200, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[stop, mc,lr])

# -----------------------------------------------------------------------------------------------------
# 최고 모델로 평가
model = load_model(modelpath)

result = model.evaluate(x_test, y_test, batch_size=batch_size)
print('loss: ', result[0], '\nacc: ', result[1])

# -----------------------------------------------------------------------------------------------------
# 예측, 저장
submission = pd.read_csv('D:/lotte_data/LPD_competition/sample.csv', index_col=0)
pred_size = 72000

# 이미지 불러와서용
y_pred =[]
for imgnumber in range(pred_size):
    pred_img = cv2.imread('D:/lotte_data/LPD_competition/test/'+ str(imgnumber) + '.jpg')
    pred_img = cv2.resize(pred_img, (150, 150))
    pred_img = pred_img.reshape(1, 150, 150, 3)
    pred_img = np.array(pred_img)
    pred_img = preprocess_input(pred_img)
    temp = np.argmax(model.predict(pred_img))
    y_pred.append(temp)
    if imgnumber % 3000 == 2999:
        print(str(imgnumber)+'번째 이미지 작업 완료')
y_pred = np.array(y_pred)
print(y_pred.shape)
submission['prediction'][:pred_size] = y_pred
submission.to_csv('D:/lotte_data/LPD_competition/sub/syhj_angel.csv',index=True)
print('==== csv save done ====')

# ===============================
# syhj_angel
# loss:  0.1700112670660019
# acc:  0.985689103603363
# 5.669
