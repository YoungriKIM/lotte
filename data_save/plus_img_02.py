# 이미지 추가 1회에서 한 것 + 2회차에 한것
# 거기에 이미지 선명하게 까지~~~ 해서 저장!

import tensorflow as tf
import numpy as np

batch_size = 32
img_height = 128
img_width = 128
data_dir = "C:/lotte_data/LPD_competition/plus_train"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, image_size=(img_height, img_width), batch_size=batch_size)

# # ---------------------------------------------------------------------
# # npy로 저장
# np.save('D:/lotte_data/npy/plus_image.npy', arr = image)
# np.save('D:/lotte_data/npy/plus_label.npy', arr = label)
# print('===== done =====')
