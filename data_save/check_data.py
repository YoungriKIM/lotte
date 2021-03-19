# 데이터 잘 저장되었나 확인

import numpy as np


y_train = np.load('D:/lotte/npy/just_scale_lpd_train_y.npy')

print(y_train.shape)
print(y_train[0])
# imagg로 만든 npy의 라벨링 순서가 이상하게 되어있음... 그냥 이미지 for문으로 붙이고
# y라벨링은..알아서 만드는 편이 좋을 듯

