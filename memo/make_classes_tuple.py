import numpy as np

#  classes 만들려고~
# 이번에는 100으로

makeclasses = []
for i in range(0, 100):
    aaa = (str(i) + ' ')
    ccc = aaa.split()
    makeclasses.append(ccc)
classes = tuple(np.array(makeclasses).squeeze())

print(classes)
