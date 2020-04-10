#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd


# In[15]:


titanic_data = pd.read_csv("D:\\Data for Machine Learning Projects\\Titenic\\train.csv")
titanic_data.head()
# this data was taken from kaggke.com/c/3136


# In[16]:


titanic_data.drop(['PassengerId', 'Name','Ticket','Cabin'], 'columns', inplace = True)
titanic_data.head()


# In[17]:


# here survived =0 means they died in sinking and 1 means they didn't
# embarked means the boarding station as S stands for 'Southampton', C stand for 'Cherbourg' and Q stands for 'Queenstown'


# In[18]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
titanic_data['Sex'] = le.fit_transform(titanic_data['Sex'].astype(str))
titanic_data.head()
# this convert the sex column into binary from string and 1 equals to male and 0 equals to female


# In[20]:


titanic_data = pd.get_dummies(titanic_data, columns=['Embarked'])
titanic_data.head()
# for converting into one-hot encoding


# In[23]:


titanic_data[titanic_data.isnull().any(axis=1)]
# to check if there is any null data 


# In[25]:


titanic_data = titanic_data.dropna()
# all the missing values are dropped


# In[26]:


from sklearn.cluster import MeanShift as MS

analyzer = MS(bandwidth=50)
analyzer.fit(titanic_data)
# standard deviation is directly proportional to the bandwidth and smaller the BW larger/tall/skinny kernels and vice-versa 


# In[28]:


from sklearn.cluster import estimate_bandwidth as EB
EB(titanic_data)


# In[29]:


labels= analyzer.labels_


# In[30]:


import numpy as np

np.unique(labels)
# this is the no. of clusters formed


# In[31]:


import numpy as np

titanic_data['cluster_group'] = np.nan
data_length = len(titanic_data)
for i in range(data_length):
    titanic_data.iloc[i, titanic_data.columns.get_loc('cluster_group')] = labels[i]
# to add one more column of cluster group in our data


# In[32]:


titanic_data.head()


# In[33]:


titanic_data.describe()


# In[39]:


titanic_cluster_data = titanic_data.groupby(['cluster_group']).mean()
titanic_cluster_data


# In[40]:


titanic_cluster_data['counts'] = pd.Series(titanic_data.groupby(['cluster_group']).size())
titanic_cluster_data


# In[41]:


titanic_data[ titanic_data['cluster_group'] == 1 ].describe()


# In[42]:


titanic_data[ titanic_data['cluster_group'] == 1]


# In[43]:


from sklearn.cluster import MeanShift as MS

analyzer = MS(bandwidth=30)
analyzer.fit(titanic_data)
# now woth the estimated bandwidth


# In[44]:


labels= analyzer.labels_


# In[45]:


import numpy as np

np.unique(labels)
# this is the no. of clusters formed


# In[46]:


import numpy as np

titanic_data['cluster_group'] = np.nan
data_length = len(titanic_data)
for i in range(data_length):
    titanic_data.iloc[i, titanic_data.columns.get_loc('cluster_group')] = labels[i]
# to add one more column of cluster group in our data


# In[47]:


titanic_data.head()


# In[48]:


titanic_data.describe()


# In[49]:


titanic_cluster_data = titanic_data.groupby(['cluster_group']).mean()
titanic_cluster_data


# In[50]:


titanic_cluster_data['counts'] = pd.Series(titanic_data.groupby(['cluster_group']).size())
titanic_cluster_data


# In[51]:


titanic_data[ titanic_data['cluster_group'] == 1 ].describe()


# In[52]:


titanic_data[ titanic_data['cluster_group'] == 1]


# In[57]:


import matplotlib.pyplot as plt
plt.plot(titanic_cluster_data)
plt.show()


# In[ ]:




