# 크기 키운 게 결과가 이상해짐..!
# 결과 잘 나온 가중치를 불러와서 확인하자!

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Dropout, Flatten, ZeroPadding2D, BatchNormalization, Input, Concatenate, AveragePooling2D
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.regularizers import l2

# 150 돌리기위한 초석

from tensorflow import keras  # or import keras for standalone version
from tensorflow.keras.layers import Input

model = load_model('C:/lotte_data/h5/imgg_04_8.hdf5')
new_input = Input(shape=(150, 150, 3), name='image_input')

# or kerassurgeon for standalone Keras
from tfkerassurgeon import delete_layer, insert_layer

model = delete_layer(model.layers[0])
# inserts before layer 0
model = insert_layer(model.layers[0], new_input)

model.summary()



'''
# x_train_01 = np.load('C:/lotte_data/npy/train_data_1000.npy')   #128
# y_label_01 = np.load('C:/lotte_data/npy/train_label_1000.npy')

# x_train_02 = np.load('C:/lotte_data/npy/1,2sum_data_1000.npy')  #150
# y_label_02 = np.load('C:/lotte_data/npy/1,2sum_label_1000.npy')

# -----------------------------------------------------------------------------------------------------
model = load_model('C:/lotte_data/h5/imgg_04_8.hdf5')
model.load_weights('C:/lotte_data/h5/imgg_04_8.hdf5')

# -----------------------------------------------------------------------------------------------------
# 예측, 저장
submission = pd.read_csv('C:/lotte_data/LPD_competition/sample.csv', index_col=0)
pred_size = 72000

# 이미지 불러와서용
y_pred =[]
for imgnumber in range(pred_size):
    pred_img = cv2.imread('C:/lotte_data/LPD_competition/test/'+ str(imgnumber) + '.jpg')
    pred_img = cv2.resize(pred_img, (150, 150))
    pred_img = pred_img.reshape(1, 150, 150, 3)
    pred_img = np.array(pred_img)
    pred_img = preprocess_input(pred_img)
    temp = np.argmax(model.predict(pred_img))
    y_pred.append(temp)
    if imgnumber % 10000 == 19999:
        print(str(imgnumber)+'번째 이미지 작업 완료')
y_pred = np.array(y_pred)
print(y_pred.shape)
submission['prediction'][:pred_size] = y_pred
submission.to_csv('C:/lotte_data/LPD_competition/sub/odd_check_01_150.csv',index=True)
print('==== csv save done ====')

# =======================================
# odd_check_01_150
# x_train_02 = np.load('C:/lotte_data/npy/1,2sum_data_1000.npy')
# preprocess_input 적용
# 1000개 예측
'''