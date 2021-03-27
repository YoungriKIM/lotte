# train에 추가한 이미지 불러오기
# 리스트는 스프레드 시트에서 확인!

import tensorflow as tf
import numpy as np

read_img = tf.keras.preprocessing.image_dataset_from_directory('C:/lotte_data/LPD_competition/train', \
    image_size=(150,150), batch_size=50000, shuffle=False)

for image, label in read_img:
    image = image
    label = label


print(image.shape)
print(label.shape)

# ---------------------------------------------------------------------
# npy로 저장
np.save('C:/lotte_data/npy/1,2sum_data_1000_new.npy', arr = image)
np.save('C:/lotte_data/npy/1,2sum_label_1000_new.npy', arr = label)
print('===== done =====')
