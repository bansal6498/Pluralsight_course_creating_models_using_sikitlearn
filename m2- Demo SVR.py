#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd


# In[60]:


auto_data= pd.read_csv("D:\\Data for Machine Learning Projects\\Pluralsight\\SVR\\auto-mpg.data", delim_whitespace= True, header = None,
                      names=['mpg', 'cylinders', 'displacement', 'horsepower','weight','acceleration','model','origin','car_name'])


# In[61]:


auto_data.head()


# In[62]:


len(auto_data['car_name'].unique())


# In[63]:


auto_data= auto_data.drop('car_name', axis=1)
auto_data.head()


# In[64]:


auto_data['origin']= auto_data['origin'].replace({1:'America', 2:'Europe', 3:'Asia'})
auto_data.head()


# In[65]:


auto_data= pd.get_dummies(auto_data, columns=['origin'])
auto_data.head()


# In[66]:


# pd.get_dummies() is used to convert column into ONE-HOT ENCODING


# In[67]:


import numpy as np
auto_data= auto_data.replace('?', np.nan)


# In[68]:


auto_data= auto_data.dropna()
auto_data.head()


# In[69]:


from sklearn.model_selection import train_test_split
X= auto_data.drop('mpg', axis=1)
# Taking the Lables (mpg)
Y= auto_data['mpg']
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state=0)


# In[83]:


from sklearn.svm import SVR
regression_model= SVR(kernel='linear', C=0.5)
regression_model.fit(X_train, Y_train)
# we choose kernel as linear as we are performing the linear regression here and C is the hyperperameter or penalty


# In[84]:


regression_model.coef_
# to find the coefficients of our model


# In[85]:


regression_model.score(X_train, Y_train)
#     this is for checking the score or the accuracy of our model which comes out to be about 62%


# In[86]:


from pandas import Series 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# this was for display our output in jupyter notebook here

predictors = X_train.columns
coef = Series(regression_model.coef_[0].predictors).sort_values()
coef.plot(kind= 'bar', title='Modal Coefficient')


# In[87]:


y_predict = regression_model.predict(x_test)


# In[88]:


get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] =(15,6)

plt.plot(y_predict, label='Predicted')
plt.plot(y_test.values, label='Actual')
plt.ylabel('MPG')

plt.legend()
plt.show


# In[89]:


regression_model.score(x_test, y_test)


# In[90]:


from sklearn.metrics import mean_squared_error
regression_model_mse= mean_squared_error(y_predict, y_test)
regression_model_mse


# In[91]:


import math
math.sqrt(regression_model_mse)


# In[ ]:




