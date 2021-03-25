# 성은황 동공지능에게 감사를

# 높은 점수 csv 모아서 확률 높은 걸로 만들기

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from scipy import stats

x = []
for i in range(1,4):           # 파일의 갯수
    if i != 10:                 # 10번파일은 빼고 확인해보겠다.
        df = pd.read_csv(f'D:/lotte_data/LPD_competition/sumsub/answer ({i}).csv', index_col=0, header=0)
        data = df.to_numpy()
        x.append(data)

x = np.array(x)

# print(x.shape)
a= []
df = pd.read_csv(f'D:/lotte_data/LPD_competition/sumsub/answer ({i}).csv', index_col=0, header=0)
for i in range(72000):
    for j in range(1):
        b = []
        for k in range(3):         # 파일의 갯수
            b.append(x[k,i,j].astype('int'))
        a.append(stats.mode(b)[0]) 
# a = np.array(a)
# a = a.reshape(72000,4)

# print(a)

sub = pd.read_csv('D:/lotte_data/LPD_competition/sample.csv')
sub['prediction'] = np.array(a)
sub.to_csv('D:/lotte_data/LPD_competition/sub/sumsub_01.csv',index=False)

