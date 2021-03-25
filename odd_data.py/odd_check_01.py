# 점수는 잘 나왔는데 제출할 때 점수가 trash인 현상...발생 ^^
# 무슨 일인지 파헤쳐보자...

import numpy as np
import cv2

'''
x_train_01 = np.load('C:/lotte_data/npy/train_data_1000.npy')

x_train_02 = np.load('C:/lotte_data/npy/1,2sum_data_1000.npy')

x_train_03 = np.load('C:/lotte_data/npy/1,2sum_data_1000_new.npy')

print('x_train_01.shape: ', x_train_01.shape)
print('x_train_02.shape: ', x_train_02.shape)
print('x_train_03.shape: ', x_train_03.shape)
# x_train_01.shape:  (48000, 128, 128, 3)
# x_train_02.shape:  (48214, 150, 150, 3)
# x_train_03.shape:  (48160, 150, 150, 3)

print('type(x_train_01): ', type(x_train_01))
print('type(x_train_02): ', type(x_train_02))
print('type(x_train_03): ', type(x_train_03))
# type(x_train_01):  <class 'numpy.ndarray'>
# type(x_train_02):  <class 'numpy.ndarray'>
# type(x_train_03):  <class 'numpy.ndarray'>

print('max(x_train_01): ', np.max(x_train_01), '\tmin(x_train_01): ', np.min(x_train_01))
print('max(x_train_02): ', np.max(x_train_02), '\tmin(x_train_02): ', np.min(x_train_02))
print('max(x_train_03): ', np.max(x_train_03), '\tmin(x_train_03): ', np.min(x_train_03))
# max(x_train_01):  255   min(x_train_01):  0
# max(x_train_02):  255   min(x_train_02):  0
# max(x_train_03):  255.0         min(x_train_03):  0.0

from keras.applications.imagenet_utils import preprocess_input
x_train_01 = preprocess_input(x_train_01)
x_train_02 = preprocess_input(x_train_02)
x_train_03 = preprocess_input(x_train_03)
print('max(x_train_01): ', np.max(x_train_01), '\tmin(x_train_01): ', np.min(x_train_01))
print('max(x_train_02): ', np.max(x_train_02), '\tmin(x_train_02): ', np.min(x_train_02))
print('max(x_train_03): ', np.max(x_train_03), '\tmin(x_train_03): ', np.min(x_train_03))

'''


y_label_01 = np.load('C:/lotte_data/npy/train_label_1000.npy')
y_label_02 = np.load('C:/lotte_data/npy/1,2sum_label_1000.npy')
y_label_03 = np.load('C:/lotte_data/npy/1,2sum_label_1000_new.npy')

print('y_label_01.shape: ', y_label_01.shape)
print('y_label_02.shape: ', y_label_02.shape)
print('y_label_03.shape: ', y_label_03.shape)
# y_label_01.shape:  (48000,)
# y_label_02.shape:  (48214,)
# y_label_03.shape:  (48160,)

print('max(y_label_01): ', np.max(y_label_01), '\tmin(y_label_01): ', np.min(y_label_01))
print('max(y_label_02): ', np.max(y_label_02), '\tmin(x_train_02): ', np.min(y_label_02))
print('max(y_label_03): ', np.max(y_label_03), '\tmin(y_label_03): ', np.min(y_label_03))
# max(y_label_01):  999   min(y_label_01):  0
# max(y_label_02):  999   min(x_train_02):  0
# max(y_label_03):  999   min(y_label_03):  0

print('y_label_01(50개): ',(y_label_01[1000:1050]))
print('y_label_02(50개): ',(y_label_02[1000:1050]))
print('y_label_03(50개): ',(y_label_03[1000:1050]))
# y_label_01(50개):  [20 20 20 20 20 20 20 20 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21
#  21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21
#  21 21]
# y_label_02(50개):  [20 20 20 20 20 20 20 20 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21
#  21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21
#  21 21]
# y_label_03(50개):  [20 20 20 20 20 20 20 20 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21
#  21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21
#  21 21]

