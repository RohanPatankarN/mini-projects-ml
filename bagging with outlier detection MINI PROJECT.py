#!/usr/bin/env python
# coding: utf-8

# 
# # bagging technique with outlier detection 

# it is also called bootstrap aggregation where it uses subset of resample with replacemt like if we have 100 sample it will take random 70 sample of sets like we need to want and after training using cv we will take mean for its score this is bootstrap aggregation

# In[2]:


import pandas as pd


# In[3]:


data=pd.read_csv("heart.csv")


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.isna().any().sum()


# In[7]:


data.Sex.value_counts()


# In[8]:


(725/193)


# In[9]:


data.describe()


# In[10]:


upperlim=data.Cholesterol.mean()+3*data.Cholesterol.std()


# In[11]:


lowerlim=data.Cholesterol.mean()-data.Cholesterol.std()


# In[12]:


upperlim,lowerlim


# In[13]:


data[(data.Cholesterol>upperlim)|(data.Cholesterol<lowerlim)]


# In[14]:


data[data.Cholesterol>(data.Cholesterol.mean()+3*data.Cholesterol.std())]


# In[15]:


data1=data[data.Cholesterol<=(data.Cholesterol.mean()+3*data.Cholesterol.std())]
data1.shape


# In[16]:


data[data.RestingBP>(data.RestingBP.mean()+data.RestingBP.std()*3)]


# In[17]:


data2=data[data.RestingBP<=(data.RestingBP.mean()+3*data.RestingBP.std())]
data2.shape


# In[18]:


data[data.FastingBS>(data.FastingBS.mean()+3*data.FastingBS.std())]


# In[19]:


data[data.MaxHR>(data.MaxHR.mean()+3*data.MaxHR.std())]


# In[20]:


data[data.Oldpeak>(data.Oldpeak.mean()+3*data.Oldpeak.std())]


# In[21]:


data3=data[data.Oldpeak<=(data.Oldpeak.mean()+3*data.Oldpeak.std())]
data.shape[0]-data3.shape[0]


# In[22]:


data.RestingECG.unique()


# In[23]:


data.ChestPainType.unique()


# In[24]:


data.ST_Slope.unique()


# In[25]:


data.ExerciseAngina.unique()


# In[26]:


data4=data3.copy()


# In[27]:


data4.ExerciseAngina=data4.ExerciseAngina.map({"N":"0","Y":"1"}).astype(int)


# In[28]:


data4.RestingECG=data4.RestingECG.map({'Normal':0, 'ST':1, 'LVH':2}).astype(int)


# In[29]:


data4.ChestPainType=data4.ChestPainType.map({'ATA':0, 'NAP':1,'ASY':2,"TA":3}).astype(int)


# In[30]:


data4.ST_Slope=data4.ST_Slope.map({'Up':0, 'Flat':1, 'Down':2})


# In[31]:


data4.Sex=data4.Sex.map({"M":1,"F":0}).astype(int)


# In[ ]:





# In[32]:


data4


# In[33]:


data4.shape


# In[34]:


x=data4.drop(["HeartDisease"],axis=1)
y=data4.HeartDisease


# In[35]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scaledX=sc.fit_transform(x)


# In[36]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaledX, y, test_size=0.2, random_state=10)


# In[37]:


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
cvs=cross_val_score(SVC(),x,y,cv=6)
cvs.mean()


# In[38]:


from sklearn.ensemble import BaggingClassifier
bg=BaggingClassifier( base_estimator=SVC(),n_estimators=150,max_samples=0.9)
scores=cross_val_score(bg,x,y)
bg.fit(X_train,y_train)
scores.mean()


# In[39]:


from sklearn.tree import DecisionTreeClassifier
score=cross_val_score(DecisionTreeClassifier(),X_train,y_train)
scores.mean()


# In[40]:


from sklearn.ensemble import BaggingClassifier
bg=BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100,oob_score=True)
scores=cross_val_score(bg,x,y,cv=6)
scores.mean()


# OUT OF ALL THE ALGORITHM I USED ABOVE HERE I THINK FROM ENSEMBLE BAGGINGCLASSIFIER WORKS VERY WELL COMPARED TO ALL THE OTHER TECHNIQUES I HAVE USED 
