# coding : utf-8
'''
@creat_time = 2021/8/15,10:08
@auther = MrCrimson
Emal : mrcrimson@163.com
'''
import numpy as np
import teethtext_module
import random
import os
'''
x = np.array(range(0,10))
y = np.array(range(10,20))
data = np.array((x,y)).T

data = np.load('model/train_data.npy')
#data = data.reshape(200,2,27)
print(len(data))
print(data)
print(data[0])

label = np.load('model/train_label.npy')
print(len(label))
print(label)
print(label[0])
'''

'''
a = np.array([1,2,3])
b = np.array([4,5,6])
new = np.vstack((a,b)).T
print(new)
print(type(new))

x = np.array(range(0,10))
y = np.array(range(10,20))
data = np.array((x,y)).T
train_data = np.empty([2,2])
print(train_data)
train_data = np.append(train_data,data,axis=0)
print(train_data)
train_label = np.empty([0,1])
print(np.empty([2,2],dtype=int))


np.savetxt('test.txt',data)
data_load = np.loadtxt('test.txt')
print(data_load)

train_data = np.empty([2,2])
train_label = np.empty([0, 1])

    # 二维这里不需要这个reshape,格式在之前的处理已经解决
    # data_raw = data_raw.reshape((len(data_raw),1))#转换为len(data_raw)行，1列
data_line = teethtext_module.data_extend(data,max_length=50)
print(data_line)
data_line = teethtext_module.norm_data(data_line, 1, initial_value=0)
print(data_line)
    # data_line = data_line.reshape((1, len(data_line)))
train_data = np.append(train_data, data_line, axis=0)
ran_label = random.randint(0,7)
train_label = np.append(train_label, ran_label)

np.save('train_data.npy', train_data)  # npy文件为numpy专用的二进制文件
np.save('train_label.npy', train_label)

train_data_test = np.load('train_data.npy')
train_label_test = np.load('train_label.npy')
print(train_data_test)
print(train_label_test)

a = 1.
b = 2.
c = 3.
d = 4.
data = np.empty([0,2])
print(data)
test = np.vstack((a,b)).T
new = np.vstack((c,d)).T
data = np.append(data,test,axis=0)
data = np.append(data,new,axis=0)
print(test)
print(data)

a = np.array([3,3,3])
b = np.array([1,1,1])
c = a>b
d = abs(a-b)
print(c)
print((a/b)*d)
if c.all() :
    print(d)
print(np.array([[2,2]]))
f = np.array([[2,2]])
e = np.vstack((1,2)).T
f = np.append(f,e,axis=0)
print(f)
'''
a = np.array([1,2])
b = np.array([3,4])
print(a*b)