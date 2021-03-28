import numpy as np
import cv2
pred_size = 7200

y_pred =[]
y_pred_result =[]

for z in range(10):
    for imgnumber in range(z*pred_size, (z*pred_size)+pred_size):
        y_pred.append(imgnumber)
y_pred = np.array(y_pred)
print(y_pred.shape)
