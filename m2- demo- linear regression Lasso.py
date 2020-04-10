#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


print(pd.__version__)


# In[3]:


auto_data= pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data', sep=r'\s*,\s*', engine='python')
auto_data


# In[4]:


import numpy as np


# In[6]:


auto_data= auto_data.replace('?',np.NaN)
auto_data.head()


# In[7]:


auto_data.describe()


# In[12]:


auto_data.describe(include='all')


# In[10]:


#auto_data['price'].describe()


# In[13]:


auto_data= auto_data.drop('normalized-losses',axis=1)
auto_data.head()


# In[ ]:




