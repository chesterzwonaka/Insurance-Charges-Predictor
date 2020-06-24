#!/usr/bin/env python
# coding: utf-8

# Come up with a model to predict monthly payments

# In[42]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression


# In[43]:


df=pd.read_csv(r"C:\Users\chest\OneDrive\Desktop\Insurance\dataset\datasets_13720_18513_insurance.csv")


# In[44]:


df.head()


# <b> EXPLORATORY DATA ANALYSIS

# In[45]:


df.info()


# In[46]:


sns.heatmap(df.isnull())


# In[47]:


sns.countplot('sex',data=df)


# In[48]:


sns.countplot('smoker',data=df)


# In[49]:


sns.countplot('region',data=df)


# In[50]:


sns.boxplot(x='sex', y='charges', data=df)


# In[51]:


df.head()


# In[52]:


dummies=pd.get_dummies(df[{'sex','smoker','region'}],drop_first=True)


# In[53]:


dummies


# In[54]:


df=pd.concat([df,dummies],axis=1)


# In[55]:


df


# In[56]:


df=df.drop(columns=['sex','smoker','region'])


# In[59]:


X=df.drop(columns='charges')


# In[60]:


X


# In[61]:


y=df.charges
y


# In[62]:


#class object
model=linear_model.LinearRegression()


# In[63]:


model.fit(X,y)


# In[64]:


model.score(X,y)


# In[ ]:




