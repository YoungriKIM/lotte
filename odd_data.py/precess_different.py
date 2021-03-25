import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
# from keras.applications.resnet import preprocess_input

# -----------------------------------------------------------------------------------------------------
# 예측용 불러오는 이미지랑 비교
pred_size = 1
submission = pd.read_csv('C:/lotte_data/LPD_competition/sample.csv', index_col=0)
# modelpath='C:/lotte_data/h5/imgg_05.hdf5'
# model = load_model(modelpath)

pred_check =[]
for imgnumber in range(pred_size):
    pred_img = cv2.imread('C:/lotte_data/LPD_competition/test/'+ str(imgnumber) + '.jpg')
    pred_img = cv2.resize(pred_img, (150, 150))
    pred_img = pred_img.reshape(1, 150, 150, 3)
    print('pred_img.shape: ', pred_img.shape)
    print('type(pred_img): ', type(pred_img))
    print('max(pred_img): ', np.max(pred_img), '\tmin(pred_img): ', np.min(pred_img))
    pred_img = preprocess_input(pred_img)
    print('max(pred_img): ', np.max(pred_img), '\tmin(pred_img): ', np.min(pred_img))



# pred_img.shape:  (1, 150, 150, 3)
# type(pred_img):  <class 'numpy.ndarray'>
# max(pred_img):  254     min(pred_img):  4
# max(pred_img):  150.061         min(pred_img):  -119.68
