# 데이터 확인을 위한 파일

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# 이미지 확인
test_img = load_img('D:/lotte/LPD_competition/train/8/8.jpg',)

plt.imshow(test_img)
plt.show()

# 어레이로 바꿔 크기 확인
arr_img = img_to_array(test_img)
print(arr_img.shape)
# (256, 256, 3)

# 256,256의 컬러(3)이미지
# 1000개의 라벨
# train : 1000개의 라벨 각 48개의 이미지
# test : 72000의 개별 이미지 > 프레딕트 이미지