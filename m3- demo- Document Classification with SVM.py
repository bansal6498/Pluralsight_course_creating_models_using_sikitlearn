#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_20newsgroups
twenty_train= fetch_20newsgroups(subset= 'train', shuffle=True)


# In[2]:


twenty_train.keys()


# In[3]:


print(twenty_train.data[0])


# In[4]:


twenty_train.target_names


# In[5]:


twenty_train.target


# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect= CountVectorizer()
X_train_counts= count_vect.fit_transform(twenty_train.data)
X_train_counts.shape


# In[8]:


print(X_train_counts[0])


# In[11]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf= tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[12]:


print(X_train_tfidf[0])


# In[18]:


from sklearn.svm import LinearSVC

clf_svc= LinearSVC(penalty= "l2", dual=False, tol= 1e-3)
clf_svc.fit(X_train_tfidf, twenty_train.target)
# penalty is L2 norm and tol is tolerance when to stop training on our model i.e. if the losses on our training model goes below this level then we think that our model is good well enough and we stop training


# In[21]:


from sklearn.pipeline import Pipeline

clf_svc_pipeline = Pipeline([
    ('vect',CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf',LinearSVC(penalty='l2', dual= False, tol= 0.001)
)])
# sklearn pipeline allows a linear sequence of data to be chained 


# In[22]:


clf_svc_pipeline.fit(twenty_train.data, twenty_train.target)


# In[23]:


twenty_test = fetch_20newsgroups(subset='test', shuffle= True)


# In[24]:


predicted= clf_svc_pipeline.predict(twenty_test.data)


# In[26]:


from sklearn.metrics import accuracy_score
acc_svm = accuracy_score(twenty_test.target, predicted)


# In[29]:


acc_svm
# if we use L1 norm then we get less accuracy equals to 81.5%


# In[30]:


clf_svc_pipeline = Pipeline([
    ('vect',CountVectorizer()),
    ('clf',LinearSVC(penalty='l2', dual= False, tol= 0.001)
)])


# In[32]:


clf_svc_pipeline.fit(twenty_train.data, twenty_train.target)
predicted= clf_svc_pipeline.predict(twenty_test.data)

acc_svm = accuracy_score(twenty_test.target, predicted)
acc_svm
# here we can see that the accuracy decreses as we fed thenoutput of countvectorizer directly into the svc predictor and in previous model we fed the output of Tfidf into the model which increases the accuracy of the model


# In[ ]:




