import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input

# 적용하기 전에 x_train, y_train, x_test, y_test 저장해 둔 npy를 불러오자
x_train = np.load('C:/lotte_data/LPD_competition/npy/train_data_100.npy')
y_train = np.load('C:/lotte_data/LPD_competition/npy/train_label_100.npy')
x_pred = np.load('C:/lotte_data/LPD_competition/npy/pred_data_100.npy')

x_train = preprocess_input(x_train)
x_pred = preprocess_input(x_pred)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)



print(x_train.shape)
# print(x_pred.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
# x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1]*x_pred.shape[2]*x_pred.shape[3])

print(x_train.shape)
# print(x_pred.shape)

# 폴리노미날피텨 적용
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train = poly.fit_transform(x_train)
# x_pred = poly.fit_transform(x_pred)

print(x_train.shape)
# print(x_pred.shape)



'''
idg = ImageDataGenerator(
    width_shift_range=(-1,1),  
    height_shift_range=(-1,1), 
    # rotation_range=20,
    zoom_range=0.15)
    # horizontal_flip=True,
    # fill_mode='nearest')


idg2 = ImageDataGenerator()

print(x_train.shape, y_train.shape, x_pred.shape)
# print(y_train[0])

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train,y_train, train_size = 0.9, shuffle = True, random_state=42)


train_generator = idg.flow(x_train,y_train,batch_size=64)
# seed => random_state
valid_generator = idg2.flow(x_valid,y_valid)
test_generator = x_pred
print(x_train.shape, y_train.shape)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation
from tensorflow.keras.applications import VGG19, MobileNet, EfficientNetB4,EfficientNetB7


ef = EfficientNetB4(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
ef.trainable = True
top_model = ef.output
top_model = Flatten()(top_model)
# top_model = Dense(1024, activation="relu")(top_model)
# top_model = Dropout(0.2)(top_model)
top_model = Dense(100, activation="softmax")(top_model)

model = Model(inputs=ef.input, outputs = top_model)

mc = ModelCheckpoint('C:/lotte_data/LPD_competition/h5/minsunmearong.h5',save_best_only=True, verbose=1)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(patience= 20)
lr = ReduceLROnPlateau(patience= 10, factor=0.5)

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.9), loss = 'categorical_crossentropy', metrics=['accuracy'])

learning_history = model.fit_generator(train_generator,epochs=1, validation_data=valid_generator, callbacks=[es,lr,mc])
# predict
model.load_weights('C:/lotte_data/LPD_competition/h5/minsunmearong.h5')
result = model.predict(test_generator,verbose=True)
print(result.shape)

sub = pd.read_csv('C:/lotte_data/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('C:/lotte_data/sample_02.csv',index=False)

# ================================
# 폴리몰ㄹ ㅣ적용 전 에폭 10


# 폴리몰리 적용 후 에폭10'''