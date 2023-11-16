# chao
# 时间：2023/11/7 10:12


#创建数据集csv格式，使用pandas中的方法读取数据集(inputs,outputs)、处理Nah值（插入、删除），将数据集转换为tensor

import os
import pandas as pd
import torch

#os.makedirs(os.path.join('..','data'), exist_ok=True)
dataset = os.path.join('..', 'data', 'cat_tiny.csv')
with (open(dataset, 'w')) as f:
    f.write('Fur Length,Fur Color,Eyes Color,Size,Personality,Category\n')
    f.write('short,black,blue,small,nice,british shorthair\n')
    f.write('medium,white,green,medium,independent,persian\n')
    f.write('long,yellow,gold,large,active,bombay\n')
    f.write('medium,blue,brown,medium,quiet,russian blue\n')
    f.write('short,gray,blue,small,friendly,bengal\n')
    f.write('NA,NA,NA,NA,active,NA\n')

data = pd.read_csv(dataset)
print(data)
inputs, outputs = data.iloc[:, 0:4], data.iloc[:, 5]
print(inputs)
print(outputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

x = torch.tensor(inputs.to_numpy(dtype=int))
#y = torch.tensor(outputs.to_numpy()) #离散值的情况下，没办法直接把数据转换为tensor
print('输入张量为：\n', x)
#print('输出的张量为：\n', y)
#getdummies的使用实例，利用个getdummies方法实现对离散数据的one-hot-encode
import pandas as pd
df = pd.DataFrame([
            ['green' , 'A'],
            ['red'   , 'B'],
            ['blue'  , 'A']])

df.columns = ['color',  'class']
print(df)
df = pd.get_dummies(df)
print('dummies后的df is:\n', df)
