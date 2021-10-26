#!/usr/bin/env python
# coding: utf-8

# # PRINCIPAL COMPONENT ANALYSIS -DIMENSIONALITY REDUCTION TECHNIQUE

# In[193]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.datasets import load_breast_cancer


# In[194]:


data=load_breast_cancer() #data loaded in data it is like key-value pair now


# In[195]:


data.keys() #calling key values


# In[196]:


print(data["DESCR"]) #DESCRIBING all the data present in iris data


# In[197]:


print(data.feature_names)


# In[198]:


data_new=pd.DataFrame(data["data"],columns=data["feature_names"])
data_new.head()


# In[199]:


data.target


# In[200]:


data_new.head()


# MOVING TOWARDS PCA TECHNIQUE
step1
scaling the data with help of StandardScaler()
it means it will standardize the data with mean 0 and std 1 ie x-u/sigma all the feature_values will be equally standardized
# In[201]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[202]:


sc.fit(data_new)#fitting our data


# In[203]:


scaled_data=sc.transform(data_new) #transform means it will calculate mean and variance for entire feature present in our data


# In[204]:


from sklearn.decomposition import PCA
pca=PCA(n_components=2)#it means converting 30 features present in our data to 2 feature dataset


# In[205]:


pca.fit(scaled_data)


# In[206]:


new_pca=pca.transform(data_new)


# In[207]:


scaled_data.shape


# In[208]:


new_pca.shape


# In[209]:


#yupp we have transformed 4d data into 2d data 


# In[210]:


#visualisation part


# In[214]:


plt.figure(figsize=(10,10))#10x10 
plt.scatter(new_pca[:,0],new_pca[:,1],c=data["target"],cmap="plasma")
plt.title("PRINCIPAL COMPONENT ANALYSIS VISUALISATION")
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




