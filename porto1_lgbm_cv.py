
# coding: utf-8

# In[16]:

import pandas as pd
import numpy as np
import time
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV


# In[17]:

path = '/axp/rim/imsadsml/warehouse/sagra39/Kaggle/porto/'
train_data = pd.read_csv(path + 'train_data.csv')
test_data = pd.read_csv(path + 'test_data.csv')
print (train_data.shape, test_data.shape)


# In[18]:



# ## Data Split

# In[19]:

from sklearn.model_selection import train_test_split
final_data1 = train_data
train_data_x = final_data1.drop(['id', 'target'], axis=1)
train_data_y = final_data1['target']
train_x, valid_x, train_y, valid_y = train_test_split(train_data_x, train_data_y, test_size=0.20, random_state=42)
print (train_x.shape, valid_x.shape, train_y.shape, valid_y.shape)




# ## LightGBM CV model

# In[ ]:

start_time = time.time()
estimator = lgb.LGBMClassifier(nthread=3,silent=True)

param_grid = {
    'learning_rate': [0.002],
    'n_estimators': [4000, 5000],
    'num_leaves': [20, 30, 40],
#     'max_depth': [-1],
    'boosting_type' : ['gbdt'],
    'objective' : ['binary'],
    'seed' : [777],
    'colsample_bytree' : [0.8],
    'subsample' : [0.8],
    'reg_alpha' : [0],
    'reg_lambda' : [0]
}
cv = StratifiedKFold(n_splits=4)
gbm = RandomizedSearchCV(estimator, param_distributions=param_grid,cv=cv,scoring='roc_auc', n_iter=2)

gbm.fit(train_data_x, train_data_y)

print (time.time() - start_time)


# In[ ]:

print ("best params are : ", gbm.best_params_)