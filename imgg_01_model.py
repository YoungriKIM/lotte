# image generator 쓰면 순서 뒤죽박죽 됨  for문으로 만들 것

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, BatchNormalization, Flatten, Input
import matplotlib.pyplot as plt
from keras.optimizers import Adam

# npy로 불러오자 -----------------------------------------------------------------------------------------------------

# 적용하기 전에 x_train, y_train, x_test, y_test 저장해 둔 npy를 불러오자
x_train = np.load('D:/lotte/npy/lpd_train_x.npy')
y_train = np.load('D:/lotte/npy/lpd_train_y.npy')
x_test = np.load('D:/lotte/npy/lpd_test_x.npy')
y_test = np.load('D:/lotte/npy/lpd_test_y.npy')

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# print(np.max(x_train), np.min(x_train))
# (39000, 150, 150, 3) (39000, 1000)
# (9000, 150, 150, 3) (9000, 1000)
# 1.0 0.0

# 훈련을 시켜보자! 모델구성 -----------------------------------------------------------------------------------------------------

# 전이학습 사용
# VGG-16
# ResNet50
# Inceptionv3
# EfficientNet
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, EfficientNetB2
input_tensor = Input(shape=(150, 150, 3))
apl = InceptionV3(weights='imagenet', include_top=False,input_tensor = input_tensor)
apl.trainable = True

model = Sequential()
model.add(apl)
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1000, activation='softmax'))

# 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
stop = EarlyStopping(monitor='val_acc', patience=20, mode='auto')
lr = ReduceLROnPlateau(monitor='val_lacc', factor=0.3, patience=10, mode='max')
# filepath = ('../data/modelcheckpoint/k67_-{val_acc:.4f}.hdf5')
# mc = ModelCheckpoint(filepath=filepath, save_best_only=True, verbose=1)

model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=1, validation_split=0.2, callbacks=[stop,lr])#,mc])

#  ----------------------------------------------------------------------------------------------
loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

#  ----------------------------------------------------------------------------------------------
# 예측까지 하자
import PIL.Image as pilimg

image = pilimg.open('D:/lotte/LPD_competition/test/0.jpg').resize((150,150))
pix = np.array(image)
pred_img = pix.reshape(1, 150, 150, 3)/255.

y_pred = model.predict(pred_img)
print('y_pred: \n', y_pred.argmax(axis=1))
print('y_test[:5]: \n', y_test[:5].argmax(axis=1))

# =====================================================