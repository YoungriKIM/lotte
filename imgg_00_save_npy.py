# keras66_3_save_npy.py
# 이미지데이터를 불러와서 증폭을 해보자! > npy로 저장하자

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

'''
# train
# -----------------------------------------------------------------------------------------------------
# 이미지 제너레이터 선언
train_datagen = ImageDataGenerator(
    rescale=1./255,             # 전처리/ 스케일링. 흑백이니까 1./255
    horizontal_flip=True,       # 수평 뒤집기
    vertical_flip=True,         # 수직 뒤집기
    width_shift_range=0.1,      # 수평 이동
    height_shift_range=0.1,     # 수직 이동
    rotation_range=5,           # 회전
    zoom_range=1.2,             # 확대
    shear_range=0.7,            # 왜곡
    fill_mode='nearest'         # 빈자리는 근처에 있는 것으로(padding='same'과 비슷)
    , validation_split=0.2
)

# -----------------------------------------------------------------------------------------------------
# 폴더(디렉토리)에서 불러와서 적용하기! fit과 같다. 이 줄을 지나면 x와 y가 생성이 된다.

# train_generator
xy_train = train_datagen.flow_from_directory(
    'D:/lotte/LPD_competition/train',  
    target_size=(150,150), 
    batch_size=48000,
    class_mode='categorical',
    subset='training'
)     
# Found 48000 images belonging to 1000 classes.

# test_generator
xy_test = train_datagen.flow_from_directory(
    'D:/lotte/LPD_competition/train',  
    target_size=(150,150), 
    batch_size=48000
    ,subset='validation'
) 

# -----------------------------------------------------------------------------------------------------
print(xy_train[0][0].shape)
print(xy_train[0][1].shape)

# npy로 저장하자 -----------------------------------------------------------------------------------------------------
np.save('D:/lotte/npy/lpd_train_x.npy', arr = xy_train[0][0])
np.save('D:/lotte/npy/lpd_train_y.npy', arr = xy_train[0][1])
np.save('D:/lotte/npy/lpd_test_x.npy', arr = xy_test[0][0])
np.save('D:/lotte/npy/lpd_test_y.npy', arr = xy_test[0][1])
print('===== save complete =====')

'''

# predict]

# -----------------------------------------------------------------------------------------------------
# 이미지 제너레이터 선언
pred_datagen = ImageDataGenerator(
    rescale=1./255
)

# pred_generator
xy_pred = pred_datagen.flow_from_directory(
    'D:/lotte/LPD_competition/test',  
    target_size=(150,150)
)

print(xy_pred[0][0].shape)

