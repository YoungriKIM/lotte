# keras66_3_save_npy.py
# 이미지데이터를 불러와서 증폭하지 말고 저장하자

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# train
# -----------------------------------------------------------------------------------------------------
# 이미지 제너레이터 선언
train_datagen = ImageDataGenerator(
    rescale=1./255)

# -----------------------------------------------------------------------------------------------------
# 폴더(디렉토리)에서 불러와서 적용하기! fit과 같다. 이 줄을 지나면 x와 y가 생성이 된다.

# train_generator
xy_train = train_datagen.flow_from_directory(
    'D:/lotte/LPD_competition/train',  
    target_size=(150,150), 
    batch_size=72000,
    class_mode='categorical')     

# test_generator
xy_pred = train_datagen.flow_from_directory(
    'D:/lotte/LPD_competition/pred',  
    target_size=(150,150), 
    batch_size=72000,
    class_mode='categorical') 

# -----------------------------------------------------------------------------------------------------
# print(xy_train[0][0].shape)
# print(xy_pred[0][1].shape)


# npy로 저장하자 -----------------------------------------------------------------------------------------------------
np.save('D:/lotte/npy/lpd_train_x.npy', arr = xy_train[0][0])
np.save('D:/lotte/npy/lpd_train_y.npy', arr = xy_train[0][1])
np.save('D:/lotte/npy/lpd_pred_x.npy', arr = xy_pred[0][0])
# np.save('D:/lotte/npy/lpd_test_y.npy', arr = xy_pred[0][1])
print('===== save complete =====')

# ===========================
# just_scale_