# chao
# 时间：2023/11/6 09:47

#创建数据集，写入数据

import os
import pandas as pd
import torch

os.makedirs(os.path.join('..', 'data'), exist_ok=True) #创建一个目录
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f: #文件写入
    f.write('NumRooms,Alley,Price\n') #列名
    f.write('NA,Pave,127500\n') #每一列对应的数据，每一行表示一个样本
    f.write('4,NA,2200\n')
    f.write('2,NA,302222\n')

#利用pandas读取数据集

data = pd.read_csv(data_file) #通过pandas包中的read_csv来读取创建的csv文件中的数据
print(data)

#处理缺失值（Nah值）
inputs, outputs = data.iloc[:, 0:1], data.iloc[:, 2] #利用iloc索引将数据分为input和output
inputs2 = data.iloc[:, 0:2]
inputs = inputs.fillna(inputs.mean()) #data.fillna()方法，填充
print(inputs)

#将inputs中的类别值或者离散值区分，如Alley包含Pave和Nah，pandas中有方法可以区分开
inputs2 = pd.get_dummies(inputs2, dummy_na=True)
print(inputs2)

#将inputs转换为tensor
X = torch.tensor(inputs2.to_numpy(dtype=float)) #input.to_numpy(dtype=int) 利用该方法将数据转换为tensor，并规定数据类型
Y = torch.tensor(outputs.to_numpy(dtype=float))

print(X,Y)


#创建数据集csv格式，使用pandas中的方法读取数据集、处理Nah值（插入、删除），将数据集转换为tensor