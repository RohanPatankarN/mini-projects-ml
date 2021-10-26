#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv("breast_cancer.csv")


# In[3]:


data.head(10)


# In[4]:


data=data.drop(["id","Unnamed: 32"],axis=1)


# SUMMARISING THE DATASET

# In[5]:


print(data.shape)
print(data.describe())
print()
print(data.groupby("diagnosis").size())
data.isna().any()


# In[6]:


from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


# In[7]:


#CALLING 6 ALGOS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#training and testing
from sklearn.model_selection import train_test_split

#for our accuracy score
from sklearn.model_selection import cross_val_score as cvs
from sklearn.model_selection import StratifiedKFold as skf


# In[8]:


x=data.drop(["diagnosis"],axis=1)
y=data.diagnosis
x_train,x_validation,y_train,y_validation=train_test_split(x,y,test_size=0.30,random_state=20,shuffle=True)


# In[9]:


models=[]
models.append(("LR",LogisticRegression()))
models.append(("CART",DecisionTreeClassifier()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("NB",GaussianNB()))
models.append(("SVM",SVC(gamma=1)))
models.append(("LDA",LinearDiscriminantAnalysis()))


# In[10]:


results=[]
names=[]
res=[]
for name,model in models:
    kfold=skf(n_splits=10)
    cv_result=cvs(model,x_train,y_train,cv=kfold,scoring="accuracy")
    results.append(cv_result)
    names.append(name)
    res.append(cv_result.mean())
    #print("%s : %f  (%f)" %(name,cv_result.mean(),cv_result.std()))
    print("{} : {}% mean , ({}%) std ".format(name,cv_result.mean()*100,cv_result.std()*100))


# In[11]:


plt.ylim(.60,0.98)
plt.bar(names,res,width=0.6,edgecolor="red",color="black")
plt.title("Algorithm Comparision")
plt.show()

