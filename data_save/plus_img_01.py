# train에 추가한 이미지 불러오기
# 리스트는 스프레드 시트에서 확인!

import tensorflow as tf
import numpy as np

read_img = tf.keras.preprocessing.image_dataset_from_directory('D:/lotte_data/LPD_competition/plus_train', \
    image_size=(128,128), batch_size=160, shuffle=False)

for image, label in read_img:
    a = np.array(label)
    a = np.where(a == 0, 48, a);a = np.where(a == 1, 90, a);a = np.where(a == 2, 129, a);a = np.where(a == 3, 135, a)
    a = np.where(a == 4, 141, a);a = np.where(a == 5, 227, a);a = np.where(a == 6, 273, a);a = np.where(a == 7, 329, a)
    a = np.where(a == 8, 408, a);a = np.where(a == 9, 445, a);a = np.where(a == 10, 448, a);a = np.where(a == 11, 523, a)
    a = np.where(a == 12, 545, a);a = np.where(a == 13, 582, a);a = np.where(a == 14, 737, a); a = np.where(a == 15, 788, a)
    a = np.where(a == 16, 843, a);a = np.where(a == 17, 914, a)
    a = label

print(image.shape)
print(label.shape)

# ---------------------------------------------------------------------
# npy로 저장
np.save('D:/lotte_data/npy/plus_image.npy', arr = image)
np.save('D:/lotte_data/npy/plus_label.npy', arr = label)
print('===== done =====')
