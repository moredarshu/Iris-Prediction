#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Life cycle of ML Model
# 1.Importing the libraries
# 2.Importing the dataset
# 3.Preprocessing
# 4.Exploratory Data Analysis
# 5.Feature engg. & selection
# 6.Train Test Split
# 7.Fitting the Model
# 8.Testing
# 9.Accuracy
# 10.Production Deployment
# 11.Prediction


# In[2]:


## Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# In[3]:


## Import Data Set
iris = pd.read_csv(r"C:\Users\Darshana\Desktop\DSC_WKND20092020\Python\DataSet\Iris.csv")
iris.head()


# In[4]:


iris.shape


# ## Preprocessing

# In[5]:


iris.groupby('Species')['Species'].count()


# In[6]:


iris['SepalLengthCm'].isnull().sum()


# In[7]:


iris['SepalWidthCm'].isnull().sum()


# In[8]:


iris['PetalLengthCm'].isnull().sum()


# In[9]:


iris['PetalWidthCm'].isnull().sum()


# In[10]:


iris.groupby('Species')['Species'].count()


# In[11]:


iris['Species'].unique()


# In[12]:


iris.info()


# In[13]:


iris.describe()


# In[14]:


iris['Species'] = iris['Species'].map({"Iris-setosa":0,'Iris-versicolor':1,'Iris-virginica':2})
iris.head(30)


# ## Exploratory Data Analysis

# In[15]:


sns.set_style('whitegrid')
sns.countplot(x='Species',data=iris)


# In[16]:


sns.heatmap(data=iris.corr(),cmap='Blues',annot=True)


# In[17]:


sns.boxplot(x='Species',y='SepalLengthCm',
                        data=iris)


# In[18]:


sns.boxplot(x='Species',y='SepalWidthCm',data=iris)


# In[19]:


sns.boxplot(x='Species',y='PetalLengthCm',data=iris)


# In[20]:


sns.boxplot(x='Species',y='PetalWidthCm',data=iris)


# In[21]:


x = iris.iloc[:,1:5]
y = iris['Species']


# In[22]:


x


# In[23]:


y


# In[24]:


# Splitting the data
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=40)


# In[25]:


xtrain


# In[26]:


ytrain


# In[27]:


xtest


# In[28]:


xtestindex=xtest.index
xtestindex


# In[29]:


ytest


# In[30]:


## Standardization
stdscaler = StandardScaler()
xtrain = stdscaler.fit_transform(xtrain)
xtrain


# In[31]:


xtest = stdscaler.transform(xtest)
xtest


# ## Instantiating and fitting the model-Logistic Regression

# In[32]:


# logreg = LogisticRegression()


# In[33]:


# logreg.fit(xtrain,ytrain)


# In[34]:


# ypred = logreg.predict(xtest)


# In[35]:


np.array(ytest)


# In[36]:


# ypred


# In[37]:


# accuracy = accuracy_score(ytest,ypred)
# print(accuracy)


# ## Instantiating & Fitting the model - Decision Tree

# In[38]:


# decisiontree = DecisionTreeClassifier(random_state =40,max_depth=3)
# decisiontree.fit(xtrain,ytrain)


# In[39]:


# ypred_desc = decisiontree.predict(xtest)
# ypred_desc


# In[40]:


# np.array(ytest)


# In[41]:


# accuracy_desc = accuracy_score(ytest,ypred_desc)
# print(accuracy_desc)


# ## Instantiating & Fitting the model - Random Forest

# In[42]:


# randomforest = RandomForestClassifier(random_state=40,n_estimators=2)
# randomforest.fit(xtrain,ytrain)


# In[43]:


# ypredrf = randomforest.predict(xtest)
# ypredrf


# In[44]:


# np.array(ytest)


# In[45]:


# accuracy_rf = accuracy_score(ytest,ypredrf)
# print(accuracy_rf)


# In[46]:


rf = RandomForestClassifier(random_state=40)


# In[47]:


param_dist = {'max_depth':[2,3,4],
             'criterion':['gini','entropy'],
             'max_features':[2,3,4],
             'bootstrap':[True,False],
            'n_estimators':[1,2,3]}


# In[48]:


rf_gscv = GridSearchCV(rf,cv=5,param_grid=param_dist,n_jobs=3)


# In[49]:


rf_gscv.fit(xtrain,ytrain)
print(rf_gscv.best_params_)


# In[50]:


rf.set_params(bootstrap= True, criterion= 'gini', max_depth= 3, max_features= 4, n_estimators= 3)


# In[51]:


rf.fit(xtrain,ytrain)


# In[52]:


ypred_rfgscv = rf.predict(xtest)
ypred_rfgscv


# In[53]:


accuracy_rfgscv = accuracy_score(ytest,ypred_rfgscv)
accuracy_rfgscv


# ## Put back the prediction in CSV file

# In[54]:


iris.head()


# In[56]:


results = pd.DataFrame(ypred_rfgscv,columns=['Predictions'])
results


# In[57]:


results.index=xtestindex


# In[58]:


results


# In[59]:


results['Predictions'] = results['Predictions'].map({0:"Iris-setosa",1:'Iris-versicolor',2:'Iris-virginica'})
results


# In[60]:


iris.shape


# In[61]:


results.shape


# In[62]:


iris_result = pd.concat([iris,results],axis=1)
iris_result


# In[63]:


iris_result.tail(100)


# In[64]:


iris_result['Species'] = iris['Species'].map({0:"Iris-setosa",1:'Iris-versicolor',2:'Iris-virginica'})
iris_result.tail(100)


# In[65]:


# iris_result.where(iris_result['Predictions']=='NaN',iris_result['Species'],inplace=True,axis=1)
# iris_result

# df.loc[df['set_of_numbers'] == 0, 'set_of_numbers'] = 999

iris_result.loc[iris_result['Predictions'].isnull()==True,'Predictions'] = iris_result['Species']
iris_result.tail(100)


# In[67]:


iris_result.to_csv(r"C:\Users\Darshana\Desktop\DSC_WKND20092020\Python\DataSet\Iris_result.csv")

FileName = 'IRIS_MODEL.pkl'
joblib.dump(rf,FileName)