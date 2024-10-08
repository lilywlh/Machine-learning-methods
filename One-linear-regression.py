#导入包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

'''！！！一元回归'''

#加载数据
train=pd.read_csv(r'D:\highway\ML\Regression\click.csv')
train.info()

#切分解释变量和被解释变量,train_x是自变量，train_y是因变量。绘制散点图
train_x=train.iloc[:,0]
train_y=train.iloc[:,1]
#plt.plot(train_x,train_y,'o')
#plt.show()


'''
将预测函数的方程设定为f（x）=theta0+theta1*x需要初始化两个未知参数，并且设定目标函数是离差平方和
'''
#初始化参数
theta0=0.22
theta1=0.11
#预测函数
def f(x):
    return theta0+theta1*x
#目标函数
def E(x,y):
    return 0.5*np.sum((y-f(x))**2)

#标准化(只对横轴进行标准化)
mu=train_x.mean()
sigma=train_x.std()
def standardize(x):
    return(x-mu)/sigma

train_z=standardize(train_x)
#plt.plot(train_z,train_y,'o')

'''
根据线性回归的方式，对权重和偏置进行更新
'''
#学习率
eta=1e-3
#误差的差值
diff=1
#更新的次数
count=0
#迭代次数
epoch=10
#重复学习
error=E(train_z,train_y)
errors=[]

while count<epoch:
    # 更新结果保存到临时变量
    tmp0 = theta0 - eta * np.sum((f(train_z) - train_y))
    tmp1 = theta1 - eta * np.sum((f(train_z) - train_y) * train_z)
    # 更新参数
    theta0 = tmp0
    theta1 = tmp1
    # 计算与上一次误差的差值(theta0和theta1有更新)
    current_error = E(train_z, train_y)
    errors.append(current_error)
    diff = error - current_error
    error = current_error
    count+=1
    log='第{}次:theta0={:.3f},theta1={:.3f},差值={:.4f}'
    print(log.format(count,theta0,theta1,diff))

# x=np.linspace(-3,3,100)
# plt.plot(train_z,train_y,'o')
# plt.plot(x,f(x))
#plt.show()



