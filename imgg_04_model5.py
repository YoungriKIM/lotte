# 100개 라벨
# 100개의 pred 이미지로 진행
# cnn만들기 

# 그래프 그려서 acc, loss 확인!!
# 파이토치에 있는 라벨별 에큐러시 확인 만들기!
# 스프레드 시트에 정리해라~

# 3) 전이학습~ ^^  (해당 파일)
# 구글넷변이+이미지추가

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

# origin_save_npy_01
# 적용하기 전에 x_train, y_train, x_test, y_test 저장해 둔 npy를 불러오자
label_size=1000

x_train_origin = np.load('C:/lotte_data/npy/train_data_1000.npy')
y_train_origin = np.load('C:/lotte_data/npy/train_label_1000.npy')
plus_image = np.load('C:/lotte_data/npy/plus_image.npy')
plus_label = np.load('C:/lotte_data/npy/plus_label.npy')

x_train_origin2 = np.concatenate((x_train_origin, plus_image), axis=0)
x_train = np.concatenate((x_train_origin, plus_image), axis=0)
y_train_origin = np.concatenate((y_train_origin, plus_label), axis=0)

print(x_train_origin2.shape)
print(y_train_origin.shape)
# (48160, 128, 128, 3)
# (48160,)

# 전처리하자
# 전이학습용 전처리
x_train_origin2 = tf.keras.applications.resnet.preprocess_input(x_train_origin)
x_train = tf.keras.applications.resnet.preprocess_input(x_train)

# y벡터화
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train_origin)
# (4800,1000)

# 스플릿
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=311)
print('x_train:',x_train.shape, 'x_test:',x_test.shape); print('y_train:',y_train.shape, 'y_test:',y_test.shape)
# x_train: (3840, 128, 128, 3) x_test: (960, 128, 128, 3)
# y_train: (3840, 100) y_test: (960, 100)

print(x_train.shape, y_train.shape)
print(np.max(x_train), np.min(x_train))
print(np.max(y_train), np.min(y_train))
# x_train: (38528, 128, 128, 3) x_test: (9632, 128, 128, 3)
# y_train: (38528, 1000) y_test: (9632, 1000)
# (38528, 128, 128, 3) (38528, 1000)
# 151.061 -123.68
# 1.0 0.0


# -----------------------------------------------------------------------------------------------------
# 모델 구성
# 모델 구성
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, BatchNormalization, Flatten, Input
from tensorflow.keras.layers import Activation, ZeroPadding2D, Concatenate, AveragePooling2D
from tensorflow.keras.regularizers import l2
# model.add(Activation('softmax, elu, selu, softplus, softsign, relu, tahn, sigmoid, hard_sigmoid, exponential'))

# ==================
input = Input(shape=(128, 128, 3))

input_pad = ZeroPadding2D(padding=(3, 3))(input)
conv1_7x7_s2 = Conv2D(64, (7,7), strides=(2,2), padding='valid', activation='relu', name='conv1/7x7_s2', kernel_regularizer=l2(0.0002))(input_pad)
conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)
pool1_helper = BatchNormalization()(conv1_zero_pad)
pool1_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool1/3x3_s2')(pool1_helper)

conv2_3x3_reduce = Conv2D(64, (1,1), padding='same', activation='relu', name='conv2/3x3_reduce', kernel_regularizer=l2(0.0002))(pool1_3x3_s2)
conv2_3x3 = Conv2D(192, (3,3), padding='same', activation='relu', name='conv2/3x3', kernel_regularizer=l2(0.0002))(conv2_3x3_reduce)
conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_3x3)
pool2_helper = BatchNormalization()(conv2_zero_pad)
pool2_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool2/3x3_s2')(pool2_helper)

inception_3a_1x1 = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_3a/1x1', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
inception_3a_3x3_reduce = Conv2D(96, (1,1), padding='same', activation='relu', name='inception_3a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
inception_3a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3_reduce)
inception_3a_3x3 = Conv2D(32, (3,3), padding='valid', activation='relu', name='inception_3a/3x3', kernel_regularizer=l2(0.0002))(inception_3a_3x3_pad)
inception_3a_5x5_reduce = Conv2D(16, (1,1), padding='same', activation='relu', name='inception_3a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
inception_3a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5_reduce)
inception_3a_5x5 = Conv2D(32, (5,5), padding='valid', activation='relu', name='inception_3a/5x5', kernel_regularizer=l2(0.0002))(inception_3a_5x5_pad)
inception_3a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_3a/pool')(pool2_3x3_s2)
inception_3a_pool_proj = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_3a/pool_proj', kernel_regularizer=l2(0.0002))(inception_3a_pool)
inception_3a_output = Concatenate(axis=1, name='inception_3a/output')([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj])

inception_3b_1x1 = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_3b/1x1', kernel_regularizer=l2(0.0002))(inception_3a_output)
inception_3b_3x3_reduce = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_3b/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_3a_output)
inception_3b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3_reduce)
inception_3b_3x3 = Conv2D(64, (3,3), padding='valid', activation='relu', name='inception_3b/3x3', kernel_regularizer=l2(0.0002))(inception_3b_3x3_pad)
inception_3b_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_3b/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_3a_output)
inception_3b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5_reduce)
inception_3b_5x5 = Conv2D(64, (5,5), padding='valid', activation='relu', name='inception_3b/5x5', kernel_regularizer=l2(0.0002))(inception_3b_5x5_pad)
inception_3b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_3b/pool')(inception_3a_output)
inception_3b_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_3b/pool_proj', kernel_regularizer=l2(0.0002))(inception_3b_pool)
inception_3b_output = Concatenate(axis=1, name='inception_3b/output')([inception_3b_1x1,inception_3b_3x3,inception_3b_5x5,inception_3b_pool_proj])

inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)
pool3_helper = BatchNormalization()(inception_3b_output_zero_pad)
pool3_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool3/3x3_s2')(pool3_helper)

inception_4a_1x1 = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4a/1x1', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
inception_4a_3x3_reduce = Conv2D(96, (1,1), padding='same', activation='relu', name='inception_4a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
inception_4a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4a_3x3_reduce)
inception_4a_3x3 = Conv2D(64, (3,3), padding='valid', activation='relu', name='inception_4a/3x3' ,kernel_regularizer=l2(0.0002))(inception_4a_3x3_pad)
inception_4a_5x5_reduce = Conv2D(16, (1,1), padding='same', activation='relu', name='inception_4a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
inception_4a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4a_5x5_reduce)
inception_4a_5x5 = Conv2D(64, (5,5), padding='valid', activation='relu', name='inception_4a/5x5', kernel_regularizer=l2(0.0002))(inception_4a_5x5_pad)
inception_4a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4a/pool')(pool3_3x3_s2)
inception_4a_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4a/pool_proj', kernel_regularizer=l2(0.0002))(inception_4a_pool)
inception_4a_output = Concatenate(axis=1, name='inception_4a/output')([inception_4a_1x1,inception_4a_3x3,inception_4a_5x5,inception_4a_pool_proj])

loss1_ave_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3), name='loss1/ave_pool')(inception_4a_output)
loss1_conv = Conv2D(128, (1,1), padding='same', activation='relu', name='loss1/conv', kernel_regularizer=l2(0.0002))(loss1_ave_pool)
loss1_flat = Flatten()(loss1_conv)
loss1_fc = Dense(1024, activation='relu', name='loss1/fc', kernel_regularizer=l2(0.0002))(loss1_flat)
loss1_drop_fc = Dropout(rate=0.7)(loss1_fc)
loss1_classifier = Dense(1000, name='loss1/classifier', kernel_regularizer=l2(0.0002))(loss1_drop_fc)
loss1_classifier_act = Activation('softmax')(loss1_classifier)

model = Model(inputs=input, outputs=loss1_classifier_act)
model.summary()

# ----------------------------------------------------------------------------------------------------
# 컴파일, 훈련
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from tensorflow.keras.metrics import binary_accuracy, binary_crossentropy,\
                                     categorical_accuracy, categorical_crossentropy,\
                                     sparse_categorical_accuracy,  sparse_categorical_crossentropy,\
                                     top_k_categorical_accuracy, sparse_top_k_categorical_accuracy
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
patience = 16
modelpath='C:/lotte_data/h5/imgg_04_9.hdf5'
batch_size = 24
stop = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
mc = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, verbose=1)
lr = ReduceLROnPlateau(factor=0.5, patience=int(patience/2), verbose=1)
model.fit(x_train, y_train, epochs=1000, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[stop,lr])# ,mc])

# -----------------------------------------------------------------------------------------------------
# 최고 모델로 평가
model = load_model(modelpath)

result = model.evaluate(x_test, y_test, batch_size=batch_size)
print('loss: ', result[0], '\nacc: ', result[1])

# -----------------------------------------------------------------------------------------------------
# 예측, 저장
submission = pd.read_csv('C:/lotte_data/LPD_competition/sample.csv', index_col=0)
pred_size = 72000

# 이미지 불러와서용
y_pred =[]
for imgnumber in range(pred_size):
    pred_img = cv2.imread('C:/lotte_data/LPD_competition/test/'+ str(imgnumber) + '.jpg')
    pred_img = cv2.resize(pred_img, (128, 128))
    pred_img = pred_img.reshape(1, 128, 128, 3)
    pred_img = np.array(pred_img)
    pred_img = preprocess_input(pred_img)
    temp = np.argmax(model.predict(pred_img))
    y_pred.append(temp)
    if imgnumber % 3000 == 2999:
        print(str(imgnumber)+'번째 이미지 작업 완료')
y_pred = np.array(y_pred)
print(y_pred.shape)
submission['prediction'][:pred_size] = y_pred
submission.to_csv('C:/lotte_data/LPD_competition/sub/imgg_04_9.csv',index=True)
print('==== csv save done ====')



# # -----------------------------------------------------------------------------------------------------
# # 최고 모델로 100가지 애큐러시 확인
# for i in range(label_size):
#     class_correct = 0
#     for j in range(48):
#         outputs = model.predict(x_train_origin2[(i*48)+j].reshape(1,128,128,3))
#         outputs = np.argmax(outputs)
#         if outputs == y_train_origin[(i*48)+j]: class_correct += 1
#     each_acc = float(class_correct/48)
#     if each_acc <= 0.9: print(str(i)+'번 라벨 acc: ', each_acc)
#     else: print(str(i)+'번 라벨 acc: pass')


# -----------------------------------------------------------------------------------------------------
end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time) 

print('°˖✧(ง •̀ω•́)ง✧˖° 잘한다 잘한다 잘한다~')


# =====================
# imgg_04_3 > 이미지 추가 전 + 전처리 안 하고 들어감
# loss:  0.23780444264411926
# acc:  0.9947916865348816

# 04_4  >  process_imput

# imgg_04_model5 > 이미지 추가 + 전처리 /255.
# loss:  0.11936328560113907
# acc:  0.9953280687332153
# time >>  4:29:40.401953
# > 롯데 스코어 이상 전처리 안해야 하는 ㅏ모양

# imgg_04_model6 > 전처리 뺌
# 트레인은 전처리 빼고 프레드는 전처리함..ㅡㅡ
# trash

# 04_7 > process_imput + plus image 
# 앞뒤모두 process_input
# loss:  0.1611163467168808
# acc:  0.9948089718818665
# time >>  3:16:14.859025

# imgg_04_8
# > _7 에서 에폭 1000번으로

# imgg_04_9
# sgd 0.001 으로
# loss:  0.2573263347148895
# acc:  0.9934592843055725

