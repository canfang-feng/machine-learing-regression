#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""ML problems involve data points with real-valued labels  𝑦 , which represent some quantity of interest. 
We often refer to such ML problems as regression problems. 
Here are some basic ML methods: Linear Regression, Huber regression, """


# In[ ]:


# import "Pandas" library/package (and use shorthand "pd" for the package) 
# Pandas provides functions for loading (storing) data from (to) files
import pandas as pd  
from matplotlib import pyplot as plt 
from IPython.display import display, HTML
import numpy as np   
from sklearn.datasets import load_boston  #Load and return the boston house-prices dataset (regression).
import random

def GetFeaturesLabels(m=10, n=10):

    house_dataset = load_boston()
    house = pd.DataFrame(house_dataset.data, columns=house_dataset.feature_names) 
    x1 = house['RM'].values.reshape(-1,1)   # vector whose entries are the average room numbers for each sold houses
    x2 = house['NOX'].values.reshape(-1,1)  # vector whose entries are the nitric oxides concentration for sold houses


    x1 = x1[0:m]
    x2 = x2[0:m]
    np.random.seed(30)
    X = np.hstack((x1,x2,np.random.randn(m,n))) 
    
    X = X[:,0:n] 

    y = house_dataset.target.reshape(-1,1)  # creates a vector whose entries are the labels for each sold house
    y = y[0:m]

    return X, y


# In[9]:


# use scatter plots to visulize the data
from matplotlib import pyplot as plt 

X,y = GetFeaturesLabels(10,10)
fig, axs = plt.subplots(1, 2,figsize=(15,5))
axs[0].scatter(X[:,0], y)
axs[0].set_title('average number of rooms per dwelling vs. price')
axs[0].set_xlabel(r'feature $x_{1}$')
axs[0].set_ylabel('house price $y$')

axs[1].scatter(X[:,1], y)
axs[1].set_xlabel(r'feature $x_{2}$')
axs[1].set_title('nitric oxide level vs. price')
axs[1].set_ylabel('house price $y$')

plt.show()


# In[4]:


########### Linear Regression ##############
"a basic linear predictor"
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from IPython.display import display, Math

reg = LinearRegression(fit_intercept=False) 
reg = reg.fit(X, y)
training_error = mean_squared_error(y, reg.predict(X))

display(Math(r'$\mathbf{w}_{\rm opt} ='))
optimal_weight = reg.coef_
optimal_weight = optimal_weight.reshape(-1,1)
print(optimal_weight)
print("\nThe resuling training error is ",training_error)


# In[5]:


########### Linear Regression ##############
"Varying Number of Features"
import time

m = 10                            # we use 10 data points of the house sales database 
max_r = 10                        # maximum number of features used 

#X,y = GetFeaturesLabels(m,max_r)  # read in m data points using max_r features 

linreg_time = np.zeros(max_r)     # vector for storing the exec. times of LinearRegresion.fit() for each r
linreg_error = np.zeros(max_r)    # vector for storing the training error of LinearRegresion.fit() for each r

#For r in range(int(max_r)):
for r in list(range(1,max_r+1)):
    x=X[:,:r]
    start_time=time.time()
    reg=LinearRegression(fit_intercept=False) 
    end_time=(time.time()-start_time)*1000
    reg=reg.fit(x,y)
    training_error = mean_squared_error(y, reg.predict(x))
    linreg_time[r-1]=end_time
    linreg_error[r-1]=training_error

plot_x = np.linspace(1, max_r, max_r, endpoint=True)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
axes[0].plot(plot_x, linreg_error, label='MSE', color='red')
axes[1].plot(plot_x, linreg_time, label='time', color='green')
axes[0].set_xlabel('features')
axes[0].set_ylabel('empirical error')
axes[1].set_xlabel('features')
axes[1].set_ylabel('time (ms)')
axes[0].set_title('training error vs number of features')
axes[1].set_title('computation time vs number of features')
axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.show()


# In[6]:


########### Linear Regression ##############
" Varying Number of Data Points."

import time
max_m = 10                            # maximum number of data points 
X, y = GetFeaturesLabels(max_m, 2)      # read in max_m data points using n=2 features 
train_error = np.zeros(max_m)         # vector for storing the training error of LinearRegresion.fit() for each r
 
for r in range(max_m):
    t_reg = LinearRegression(fit_intercept=False) 
    t_reg = t_reg.fit(X[:(r+1),:], y[:(r+1)])
    y_pred = t_reg.predict(X[:(r+1),:])
    train_error[r] = mean_squared_error(y[:(r+1)], y_pred)

print(train_error[2])
plot_x = np.linspace(1, max_r, max_r, endpoint=True)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
axes.plot(plot_x, train_error, label='MSE', color='red')
axes.set_xlabel('number of data points (sample size)')
axes.set_ylabel('training error')
axes.set_title('training error vs. number of data points')
axes.legend()
plt.tight_layout()
plt.show()


# In[14]:


########### Linear Regression ##############
"  Generate Training and Validation Set."

"  step1_split the dataset"
from sklearn.model_selection import train_test_split # Import train_test_split function

m = 20                        # we use the first m=20 data points from the house sales database 
n = 10                        # maximum number of features used 
X,y = GetFeaturesLabels(m,n)  # read in m data points using n features 
X_train, X_val, y_train, y_val =train_test_split(X,y,test_size=0.2,random_state=2)

"step2_  Compute Training and Validation Error."

err_train = np.zeros([n,1]) 
err_val = np.zeros([n,1])

for r_minus_1 in range(n):
    reg=LinearRegression(fit_intercept=False)
    reg=reg.fit(X_train[:,:(r_minus_1+1)],y_train)
    
    pred_train=reg.predict(X_train[:,:(r_minus_1+1)])
    err_train[r_minus_1]=mean_squared_error(y_train,pred_train)
    
    pred_val=reg.predict(X_val[:,:(r_minus_1+1)])
    err_val[r_minus_1]=mean_squared_error(y_val,pred_val)
    
best_model = np.argmin(err_val)+1

"step3_   Plot the training and validation errors for the different number of features r"

plt.plot(range(1, n + 1), err_train, color='black', label=r'$E_{\rm train}(r)$', marker='o')
plt.plot(range(1, n + 1), err_val, color='red', label=r'$E_{\rm val}(r)$', marker='x')

plt.title('Training and validation error for different number of features')
plt.ylabel('Empirical error')
plt.xlabel('r features')
plt.xticks(range(1, n + 1))
plt.legend()
plt.show()

print(err_val[:4])


# In[8]:


######### huber Regression #############
"""Huber regression can be used to against a data point which is intrinsically different from all 
other data points, e.g., due to measurement errors.""" 

from sklearn import linear_model
from sklearn.linear_model import HuberRegressor

m = 10                            # we use 100 data points of the house sales database 
max_r = 10                        # maximum number of features used 

X,y = GetFeaturesLabels(m,max_r)  # read in 100 data points using 10 features 

linreg_time = np.zeros(max_r)     # vector for storing the exec. times of LinearRegresion.fit() for each r
linreg_error = np.zeros(max_r)    # vector for storing the training error of LinearRegresion.fit() for each r


for r in range(max_r):
    reg_hub = HuberRegressor(fit_intercept=False) 
    start_time = time.time()
    reg_hub = reg_hub.fit(X[:,:(r+1)], y)
    end_time = (time.time() - start_time)*1000
    linreg_time[r] = end_time
    pred = reg_hub.predict(X[:,:(r+1)])
    linreg_error[r] = mean_squared_error(y, pred)

plot_x = np.linspace(1, max_r, max_r, endpoint=True)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
axes[0].plot(plot_x, linreg_error, label='MSE', color='red')
axes[1].plot(plot_x, linreg_time, label='time', color='green')
axes[0].set_xlabel('features')
axes[0].set_ylabel('empirical error')
axes[1].set_xlabel('features')
axes[1].set_ylabel('Time (ms)')
axes[0].set_title('training error vs number of features')
axes[1].set_title('computation time vs number of features')
axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.show()


# In[ ]:


#########  Multiple functions
from sklearn.preprocessing import PolynomialFeatures

x=np.array(list(move_data["stw"])).reshape(-1,1)  # change the data 
y=np.array(list(move_data["power@propulsion"])).reshape(-1,1) # change the data 

poly = PolynomialFeatures(degree=2) # change the degree to meet the features of data. 
x_poly=poly.fit_transform(x)

print(np.shape(x_poly))
print(x_poly)

poly.fit(x_poly,y)
lin2=LinearRegression()
lin2.fit(x_poly,y)
pred_y=lin2.predict(poly.fit_transform(x))

print(np.shape(pred_y))

trainning_error=mean_squared_error(y,pred_y)
print(trainning_error)

fig2=plt.figure()
plt.scatter(x,y,color="blue",alpha=0.3)
plt.plot(x,pred_y,color="red")
plt.xlabel("stw")
plt.ylabel("power@propulsion")
plt.show()
plt.savefig('speed and energy model regression')

