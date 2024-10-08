import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

'''！！！多元回归'''

#加载数据
train=pd.read_csv(r'D:\highway\ML\Regression\click.csv')
train.info()

#切分解释变量和被解释变量,train_x是自变量，train_y是因变量。绘制散点图
train_x=train.iloc[:,0]
train_y=train.iloc[:,1]
plt.plot(train_x,train_y,'o')
plt.show()

#标准化(只对横轴进行标准化)
mu=train_x.mean()
sigma=train_x.std()
def standardize(x):
    return(x-mu)/sigma

train_z=standardize(train_x)


#初始化参数
theta=np.random.rand(3)
#创建训练数据的矩阵
'''
vstack表示在竖直方向上堆叠&hstack表示水平方向平铺，第一个特征是生成一个长度与输入数组 x 相同的全1数组
x是第二个特征，x^2是第三个特征
'''
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]),x,x**2]).T
X=to_matrix(train_z)#训练数据变成一个X矩阵

#预测函数
def f(x):
    return np.dot(x,theta)
#目标函数
def E(x,y):
    return 0.5*np.sum((y-f(x))**2)

#学习率
eta=1e-3
# 误差的差值
diff = 1
# 重复学习
error = E(X, train_y)
errors = []

while diff > 1e-2:
    # 更新结果保存到临时变量
    theta = theta - eta * np.dot(f(X) - train_y, X)

    # 计算与上一次误差的差值
    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error

#结果输出
x=np.linspace(-3,3,100)
plt.plot(train_z,train_y,'o')#‘o’是以实心圆点表示点坐标
plt.show()
plt.plot(x,f(to_matrix(x)))
plt.show()
'''
定义均方误差函数MSE
'''
def MSE(x,y):
    return(1/x.shape[0]*np.sum((y-f(x))**2))#shape[0]就是第一列的行数，1/x.shape[0]就相当于求均值

#用随机值初始化参数
theta=np.random.rand(3)
#均方误差的历史记录
errors=[]
#误差的差值
diff=1
#重复学习
errors.append(MSE(X,train_y))#必须的，需要给errors一个初始值
while diff>1e-2:
    # 更新结果保存到临时变量
    theta=theta-eta*np.dot(f(X)-train_y,X)
    errors.append(MSE(X,train_y))
    diff=errors[-2]-errors[-1]

#绘制误差变化
x=np.arange(len(errors))
plt.plot(x,errors)



'''
随机梯度下降
'''
theta = np.random.rand(3)

# 均方误差的历史记录
errors = []
# 误差的差值
diff = 1
# 重复学习
errors.append(MSE(X, train_y))
while diff > 1e-2:
    # 调整训练数据的顺序
    p = np.random.permutation(X.shape[0])
    # 随机取出训练数据，使用随机梯度下降的方法更新参数
    for x, y in zip(X[p, :], train_y[p]):
        theta = theta - eta * (f(x) - y) * x
        # 计算与上一次的差值
        errors.append(MSE(X, train_y))
        diff = errors[-2] - errors[-1]

# 结果输出
x = np.linspace(-3, 3, 100)
plt.plot(train_z, train_y, 'o')  # ‘o’是以实心圆点表示点坐标
plt.plot(x, f(to_matrix(x)))
plt.show()
