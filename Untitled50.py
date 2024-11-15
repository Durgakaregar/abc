#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np 
import pandas as pd 
data={'pencil':[300,350,260,450,500],
     'textbooks':[250,350,400,420,500],
      'drawing sheet':[100,125,190,210,250],
      'total units':[700,1075,1320,1510,np.nan],
      'profit':[80000,9500,10256,12000,15000]}
df=pd.DataFrame(data)
df


# In[14]:


print(df.describe())


# In[15]:


import matplotlib.pyplot as plt 
plt.figure(figsize=(10,6))
plt.plot(df['total units'],df['profit'])
plt.xlabel('total units')
plt.ylabel('progit')
plt.title('total profit')
plt.show()


# In[16]:


df.isnull


# In[17]:


a=df['drawing sheet'].max()
a


# In[18]:


a=df['profit'].sum()
a


# In[22]:


import pandas as pd 
iris=pd.read_csv('iris.csv')
iris


# In[23]:


df.fillna(df.mean(),inplace=True)
df


# In[24]:


df.fillna(df.sum())


# In[26]:


df.fillna(df.mean())


# In[30]:


df.fillna(df.sum())


# In[ ]:




