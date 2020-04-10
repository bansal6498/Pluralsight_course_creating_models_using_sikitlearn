#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


mnist_data= pd.read_csv("D:\\Data for Machine Learning Projects\\MNIST Digit Recognizer\\train.csv")
mnist_data.head()
# it returns the data in forward direction
# data is downloaded from kaggle - https://www.kaggle.com/c/digit-recognizer/data


# In[3]:


mnist_data= pd.read_csv("D:\\Data for Machine Learning Projects\\MNIST Digit Recognizer\\train.csv")
mnist_data.tail()
# it returns the data in opposite as last row comes first.


# In[4]:


from sklearn.model_selection import train_test_split

features = mnist_data.columns[1:]
X= mnist_data[features]
Y= mnist_data['label']

X_train, X_test, Y_train, Y_test= train_test_split(X/255., Y, test_size=0.1, random_state=0)


# In[5]:


from sklearn.svm import LinearSVC

clf_svm = LinearSVC(penalty="l2", dual= False, tol= 1e-5)
clf_svm.fit(X_train, Y_train)


# In[6]:


from sklearn.metrics import accuracy_score

y_pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(Y_test, y_pred_svm)
print('SVM Accuracy= ', acc_svm)


# In[7]:


# from sklearn.model_selection import GridSearchCV

# penalties = ['l1', 'l2']
# tolerances = [1e-3, 1e-4, 1e-5]
# max_iter = [100000]
# param_grid = {'penalty': penalties, 'tol':tolerances, 'max_iter':max_iter}

# grid_search = GridSearchCV(LinearSVC(dual = False), param_grid, cv=3)
# grid_search.fit(X_train, Y_train)

# grid_search.best_params_
# grid search is used to find the best out of many, this is done by making a grid or matrix fromwhich it makes the best combination
# for calculating the output
# cv=3 (cross validation) is the command for making our data into 3 fold 


# In[8]:


clf_svm = LinearSVC(penalty= "l1", dual = False, tol= 1e-4)
clf_svm.fit(X_train, Y_train)


# In[9]:


# the above value comes as the result from grid search which gives that l1 norm penalty is best with 1e-4 tolerance


# In[12]:


from sklearn.metrics import accuracy_score

y_pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(Y_test, y_pred_svm)
print('SVM Accuracy: ',acc_svm )


# In[ ]:




