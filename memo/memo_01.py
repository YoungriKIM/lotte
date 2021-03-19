import numpy as np



bbb = []
for i in range(0, 1000):
    aaa = (str(i) + ' ')
    ccc = aaa.split()
    bbb.append(ccc)

apple = tuple(np.array(bbb).squeeze())
print(apple)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
