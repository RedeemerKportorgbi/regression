#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd


# In[61]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing 
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[62]:


df= pd.read_csv("dataforclass.csv")


# In[63]:


df.head()


# In[64]:


df.columns


# In[65]:


df.shape


# In[66]:


df.describe()


# In[67]:


df.describe


# In[68]:


df.isnull().sum


# In[69]:


sn.heatmap(df.corr(),cmap='YlGnBu',annot=True)


# In[26]:


sn.pairplot(df)


# In[70]:


sn.heatmap(df.corr(),cmap='YlGnBu',annot=True);


# # simple linear regression

# In[79]:


#Y= aX+b
y=df['crmrte']
x=df['density']


# In[80]:


#crmte=a*density+b
#splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=100)


# In[81]:


x_test


# In[82]:


x_train_sm = sm.add_constant(x_train)


# In[83]:


#OlS model
crime_fr= sm.OLS(y_train, x_train_sm).fit()


# In[85]:


crime_fr.params


# In[86]:


crime_fr.summary()


# In[92]:


#Ploting the regression line
plt.scatter(x_train,y_train)
plt.plot(x_train,0.019380+0.008522*x_train,'r');
plt.title('Regression')
plt.xlabel('density')
plt.ylabel('crime rate')
plt.show()


# In[94]:


#generating the residuals
yhat=crime_fr.predict(x_train_sm)
error=(y_train-yhat)
error


# In[97]:


#plotting the error
fig=plt.figure()
sn.displot(error,bins=15)
plt.title('Residual')
plt.xlabel('Error terms')
plt.ylabel('Density')
plt.show()


# In[101]:


#Testing the Model
x_test_sm=sm.add_constant(x_test)
y_test_pred= crime_fr.predict(x_test_sm)
y_test_pred


# In[105]:


r2= r2_score(y_test,y_test_pred)
round(r2,2)*100


# In[ ]:




