
# coding: utf-8

# In[22]:

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


# In[23]:

path = '/axp/rim/imsadsml/warehouse/sagra39/Kaggle/porto/'
train_data = pd.read_csv(path + 'train_data.csv')
test_data = pd.read_csv(path + 'test_data.csv')
print (train_data.shape, test_data.shape)


# In[33]:

train_data.head(2)


# ## Data Split

# In[34]:

from sklearn.model_selection import train_test_split
final_data1 = train_data
train_data_x = final_data1.drop(['id', 'target'], axis=1)
train_data_y = final_data1['target']
train_x, valid_x, train_y, valid_y = train_test_split(train_data_x, train_data_y, test_size=0.20, random_state=42)
print (train_x.shape, valid_x.shape, train_y.shape, valid_y.shape)


# ## XGBoost CV

# In[19]:

# clf = GridSearchCV(xgb_model, parameters, n_jobs=5, 
#                    cv=StratifiedKFold(n_splits=3), 
#                    scoring='roc_auc',
#                    verbose=2, refit=True)

# clf.fit(train_x, train_y)


# In[ ]:

gbm = xgb.XGBClassifier()
gbm_params = {
    'learning_rate': [0.01],
    'n_estimators': [1000, 2000],
    'max_depth': [8],
    'subsample':[0.8],
    'colsample_bytree':[0.8],
    'objective':['binary:logistic'],
    'min_child_weight':[100, 200, 500],
    'seed':[1008],
    'nthread':[4]
}
cv = StratifiedKFold(n_splits=4)
grid = GridSearchCV(gbm, gbm_params,scoring='roc_auc',cv=cv,verbose=10,n_jobs=-1)
grid.fit(train_data_x, train_data_y)

print (grid.best_params_)




