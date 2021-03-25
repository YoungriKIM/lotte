# 점수는 잘 나왔는데 제출할 때 점수가 trash인 현상...발생 ^^
# 무슨 일인지 파헤쳐보자...

# _02 > (150, 150) 일 떄 뭉게진다니까 확인!

import numpy as np
import cv2
from matplotlib import pyplot as plt
import PIL.Image as pilimg
from keras.applications.imagenet_utils import preprocess_input
# 원본 이미지
origin = cv2.imread('C:/lotte_data/LPD_competition/train/0/0.jpg')
plt.subplot(3,1,1)
plt.imshow(origin)

# cv2로 불러와서 크기 지정 후 저장
x_train_01 = np.load('C:/lotte_data/npy/train_data_1000.npy')  
# image_dataset_from_directory 으로 저장
x_train_02 = np.load('C:/lotte_data/npy/1,2sum_data_1000.npy')

npy_01 = x_train_01[0]
npy_02 = x_train_02[0]
npy_prep = preprocess_input(npy_02)


plt.subplot(3,1,2)
plt.imshow(npy_02)

plt.subplot(3,1,3)
plt.imshow(npy_prep)

plt.show()



'''
from keras.applications.imagenet_utils import preprocess_input

aaa = 0
pred_img = cv2.imread('C:/lotte_data/LPD_competition/test/'+ str(aaa) + '.jpg')
pred_img = cv2.resize(pred_img, (150, 150))
pred_img = np.array(pred_img)/255.
# pred_img = preprocess_input(pred_img)
plt.imshow(pred_img)
plt.show()
'''