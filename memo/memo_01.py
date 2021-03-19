import numpy as np

bbb = []

for i in range(0, 1000):
    aaa = (str(i) + ' ')*48
    ccc = aaa.split()
    bbb.append(ccc)

print(bbb)
append = np.array(bbb)
print(append.shape)

apple = append.reshape(48000, 1)
print(apple)
print(apple.shape)
# (48000, 1)

from sklearn.