import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from TimeSeriesRegression import load_data_2 as load
T = 20
L = 1000
N = 100
np.random.seed(2)
x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4*T, 4*T, N).reshape(N, 1)

# data = np.sin(x / 1.0 / T).astype('float64')
# print('data: ', len(data), data)

# plt.figure()
#
# x1 = plt.plot(x, data)
#
# plt.show(x1)

x_train, y_train,x_test,y_test, norm_var_x, norm_var_y = load.get_data_json()

data1 = np.array([x_train[:, 0]]).astype('float64')
OpenPrice = data1.reshape(1000)

data2 = np.array([x_train[:, 1]]).astype('float64')
HighPrice = data2.reshape(1000)

data3 = np.array([x_train[:, 2]]).astype('float64')
LowPrice = data3.reshape(1000)

data4 = np.array([x_train[:, 3]]).astype('float64')
ClosePrice = data4.reshape(1000)


# data = np.array([OpenPrice, HighPrice, LowPrice, ClosePrice])


print('open price', OpenPrice)
newOpenPrice = list()
for i in range(100):
    newOpenPrice.append(OpenPrice * 1+i/1000)

newOpenPrice = np.array(newOpenPrice)

print('newopenprice' , newOpenPrice)

data = newOpenPrice


x1 = plt.plot(range(1000), OpenPrice)
x2 = plt.plot(range(1000), HighPrice)
x3 = plt.plot(range(1000), LowPrice)
x4 = plt.plot(range(1000), ClosePrice)

plt.show(x1)
plt.show(x2)
plt.show(x3)
plt.show(x4)


print('data : ', len(data), data)
# print(x_train)

torch.save(data, open('traindata.pt', 'wb'))

