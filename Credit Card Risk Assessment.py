#!/usr/bin/env python
# coding: utf-8

# ## Credit Card Risk Assessment

# In[1]:


import pandas as pd


# In[3]:


credit_df=pd.read_csv('Credit_default_dataset.csv')
credit_df.head(5)


# In[5]:


#We don't need the ID column,so lets drop it.
credit_df = credit_df.drop(["ID"],axis=1)


# In[6]:


#changing the name of  pay_0 column to pay_1 to make the numbering correct
credit_df.rename(columns={'PAY_0':'PAY_1'}, inplace=True)


# In[8]:


credit_df.head(5)


# In[10]:


#Removing Unwanted categorical levels as mentioned in data exploration
credit_df['EDUCATION'].value_counts()


# ## Data Preprocessing Steps

# In[12]:


credit_df["EDUCATION"]=credit_df["EDUCATION"].map({0:4,1:1,2:2,3:3,4:4,5:4,6:4})
credit_df["MARRIAGE"]=credit_df["MARRIAGE"].map({0:3,1:1,2:2,3:3})


# In[20]:


from sklearn.preprocessing import StandardScaler
scaling=StandardScaler()
X=credit_df.drop(['default.payment.next.month'],axis=1)
X=scaling.fit_transform(X)


# In[22]:


y=credit_df['default.payment.next.month']


# In[30]:


## Hyper Parameter Optimization

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}


# In[31]:


## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost


# In[32]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# In[33]:


classifier=xgboost.XGBClassifier()


# In[34]:


random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)


# In[36]:



from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X,y)
timer(start_time) # timing ends here for "start_time" variable


# In[37]:


random_search.best_estimator_


# In[38]:


random_search.best_params_


# In[39]:


classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.4, gamma=0.1, learning_rate=0.25,
       max_delta_step=0, max_depth=3, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)


# In[41]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X,y,cv=10)


# In[42]:


score


# In[43]:


score.mean()


# In[ ]:




