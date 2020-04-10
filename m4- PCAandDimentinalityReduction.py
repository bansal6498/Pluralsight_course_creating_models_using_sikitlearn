#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


wine_data = pd.read_csv('D:\\Data for Machine Learning Projects\\Wine quality\\winequality-white.csv',
                       names=['Fixed Acidity',
                              'Volatile Acidity',
                              'Citric Acid',
                              'Residual Sugar',
                              'Chlorides',
                              'Free Sulfur Dioxide',
                              'Total Sulfur Dioxide',
                              'Density',
                              'pH',
                              'Sulphates',
                              'Alcohol',
                              'Quality'
                             ],
                        skiprows=1,
                       sep=r'\s*;\s*', engine= 'python')
wine_data.head()


# In[3]:


wine_data['Quality'].unique()


# In[4]:


X = wine_data.drop('Quality', axis=1)
Y = wine_data['Quality']
# it is a unsupervised learning so we have only X data
from sklearn import preprocessing
X = preprocessing.scale(X)
# standarize the data by using this function

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.2, random_state=0)


# In[5]:


from sklearn.svm import LinearSVC

clf_svc = LinearSVC(penalty="l1", dual= False, tol=1e-3)
clf_svc.fit(X_train, Y_train)


# In[6]:


accuracy= clf_svc.score(X_test, Y_test)
print(accuracy)


# In[7]:


# to check the correlated variables
import matplotlib.pyplot as plt
import seaborn as sns

corrmat = wine_data.corr()
f, ax = plt.subplots(figsize=(8,8))
sns.set(font_scale= 0.8)
sns.heatmap(corrmat, vmax=.8, square= True, annot= True, fmt= '0.2f', cmap= "winter")
plt.show()
# cells are in lightgreen are strong correlation 


# In[8]:


from sklearn.decomposition import PCA

pca = PCA(n_components=11, whiten = True)
X_reduced = pca.fit_transform(X)


# In[9]:


pca.explained_variance_
# this gives us the magnitude of variation captured by each of the principal component and here total PC are 11 as we have defined in above function


# In[10]:


pca.explained_variance_ratio_


# In[11]:


import matplotlib.pyplot as plt
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Dimension')
plt.ylabel('Explain Variance Ratio')
plt.show()
# this graph is known as "SCREE PLOT" as it is having the elbow


# In[12]:


X_train, X_test, Y_train, Y_test = train_test_split(X_reduced, Y, test_size= 0.2, random_state=0)
clf_svc_pca = LinearSVC(penalty="l1", dual = False, tol = 1e-3)
clf_svc_pca.fit(X_train, Y_train)


# In[13]:


accuracy = clf_svc_pca.score(X_test, Y_test)
print(accuracy)
# now also ccuracy is same as we didn't reduce the component till yet as we have decided it as 11


# In[14]:


from sklearn.decomposition import PCA

pca = PCA(n_components=9, whiten = True)
X_reduced = pca.fit_transform(X)
# by this converting components from 11 to 9 the components having least variations are dropped


# In[15]:


pca.explained_variance_
# this gives us the magnitude of variation captured by each of the principal component and here total PC are 9 as we have defined in above function


# In[16]:


pca.explained_variance_ratio_


# In[17]:


import matplotlib.pyplot as plt
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Dimension')
plt.ylabel('Explain Variance Ratio')
plt.show()
# this graph is known as "SCREE PLOT" as it is having the elbow


# In[18]:


X_train, X_test, Y_train, Y_test = train_test_split(X_reduced, Y, test_size= 0.2, random_state=0)
clf_svc_pca = LinearSVC(penalty="l1", dual = False, tol = 1e-3)
clf_svc_pca.fit(X_train, Y_train)


# In[19]:


accuracy = clf_svc_pca.score(X_test, Y_test)
print(accuracy)
# now also ccuracy is same as we didn't reduce the component till yet as we have decided it as 11


# In[ ]:





# In[20]:


from sklearn.decomposition import PCA

pca = PCA(n_components=6, whiten = True)
X_reduced = pca.fit_transform(X)
# by this converting components from 9 to 6 the components having least variations are dropped


# In[21]:


pca.explained_variance_
# this gives us the magnitude of variation captured by each of the principal component and here total PC are 6 as we have defined in above function


# In[22]:


pca.explained_variance_ratio_


# In[23]:


import matplotlib.pyplot as plt
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Dimension')
plt.ylabel('Explain Variance Ratio')
plt.show()
# this graph is known as "SCREE PLOT" as it is having the elbow


# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(X_reduced, Y, test_size= 0.2, random_state=0)
clf_svc_pca = LinearSVC(penalty="l1", dual = False, tol = 1e-3)
clf_svc_pca.fit(X_train, Y_train)


# In[25]:


accuracy = clf_svc_pca.score(X_test, Y_test)
print(accuracy)
# now also ccuracy is same as we didn't reduce the component till yet as we have decided it as 11


# In[ ]:




