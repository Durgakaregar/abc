#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd


# In[8]:


import numpy as np


# In[10]:


import pandas as pd
import numpy as np
data=pd.read_csv("StudentStudyHour.csv")
data


# In[13]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error


# In[14]:


data=pd.read_csv("StudentStudyHour.csv")
data


# In[15]:


data.dropna(inplace=True,axis=0)
data


# In[25]:


y=data['Scores']
X=data.drop('Scores',axis=1)


# In[26]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
print(X_train)


# In[27]:


print(X_test)


# In[28]:


print(y_test)


# In[35]:


scaler=StandardScaler()
scaler.fit(X_test)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
print(X_train)
print(X_test)


# In[36]:


lr=LinearRegression()
model=lr.fit(X_train,y_train)


# In[42]:


y_pred=model.predict(X_test)
data=pd.DataFrame({'y_test':y_test,'y_pred':y_pred})
data


# In[34]:


y_pred=model.predict(X_test)
data=pd.DataFrame({'y_test':y_test,'y_pred':y_pred})
data


# In[38]:


import pandas as pd

# Assuming y_test and y_pred are numpy arrays or pandas Series
data = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})



# In[41]:


import pandas as pd

# Assuming y_test and y_pred are pandas Series or numpy arrays
# Reset indices of y_test if necessary to match with y_pred
y_test.reset_index(drop=True, inplace=True)

# Create DataFrame
data = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
print(data)


# In[43]:


import pandas as pd

# Assuming y_test is a pandas Series or DataFrame with correct indices
y_pred = model.predict(X_test)  # Assuming X_test is your test data

# Assuming y_test is already in the correct form with correct indices
data = pd.DataFrame({'y_test': y_test.values.flatten(), 'y_pred': y_pred.flatten()})
print(data)


# In[44]:


print(len(y_test), len(y_pred))


# In[45]:


import pandas as pd

# Assuming y_test and y_pred are pandas Series or numpy arrays
y_pred = model.predict(X_test)  # Make predictions

# Convert to pandas Series if necessary
y_test = pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test
y_pred = pd.Series(y_pred) if not isinstance(y_pred, pd.Series) else y_pred

# Check lengths
if len(y_test) != len(y_pred):
    raise ValueError("Lengths of y_test and y_pred must be the same")

# Create DataFrame
data = pd.DataFrame({'y_test': y_test.values.flatten(), 'y_pred': y_pred.values.flatten()})
print(data)


# In[ ]:




