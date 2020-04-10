#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[5]:


examdata= pd.read_csv('C:\Users\bansa\Downloads\exams.csv', quotechar='"')
examdata


# In[6]:


get_ipython().system('pip install opencv-python')


# In[13]:


import cv2


# In[14]:


imagePath = '‪D:\Pics\Family\IMG-20190611-WA0002.jpg'
image= cv2.imread(imagePath)


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
#for showing image in this jupyter notebook itself
import matplotlib
import matplotlib.pyplot as plt
plt.imshow(image)


# In[10]:


image.shape


# In[ ]:




