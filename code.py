##Project: Predicting the aqi for delhi using Multivariate Linear regression model from scratch

#importing libraries
import math as m
import matplotlib.pyplot as mp
import numpy as np
import pandas as pa
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

#fetching data
ds=pa.read_csv('data.csv')
##filling NaN values
df=ds.fillna(dict(ds.median()))
del(ds)
n=df.shape[0]


#splitting test and train data (70,30 split)
dtrain=df[:int(7/10*n)]
dtest=df[int(7/10*n):]

y_train=dtrain.iloc[:,14].values
x_train=dtrain.iloc[:,2:14].values
y_test=dtest.iloc[:,14].values
x_test=dtest.iloc[:,2:14].values

#using builtin linear regression model
regressor=LinearRegression()
regressor.fit(x_train,y_train)
res=regressor.predict(x_test)

#using Random forest(builtin)
reg=RandomForestRegressor(n_estimators=100)
reg.fit(x_train,y_train)
pre=reg.predict(x_test)

#using Arificial Neural Network(builtin)
reg=MLPRegressor()
reg.fit(x_train,y_train)
pre=reg.predict(x_test)

#using Linear Regression(Scratch)
xa=[[]]
x_train=x_train.T
xa[0].append(n)
fea=x_train.shape[0]
for i in range(fea):
    xa[0].append(sum(x_train[i]))
for i in range(fea):
    st=x_train[i]
    tmp=[sum(st)]
    for j in range(fea):
        tmp.append(sum(st*x_train[j]))
    xa.append(tmp)
xa=np.array(xa)

ya=[]
y_train=y_train.T
y_train=np.array(y_train)
ya.append(np.sum(y_train[0]))
for i in range(fea):
    ya.append(sum(y_train[0]*x_train[i]))
ya=np.array(ya)
ya=np.mat(ya)

X=np.matmul(ya,np.linalg.inv(xa))
X=X.T

##prediction
#generating the x matrix for Testset since x * X = A
xa=[]
    
for j in range(y_test.shape[0]):
    tmp=[0]
    for i in range(fea):
        tmp.append(x_test[j][i])
    xa.append(tmp)
xa=np.mat(xa)
pre=np.matmul(xa,X)

