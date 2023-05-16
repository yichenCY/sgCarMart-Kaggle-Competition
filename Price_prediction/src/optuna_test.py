import optuna
import pandas as pd
import xgboost as xgb
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import sys
import numpy as np
from pandas.core.frame import DataFrame
test = pd.read_pickle('test_add_cate.pkl')
train = pd.read_pickle('train_add_cate.pkl')
label = train['price']
train = train.drop(['price'],axis=1)
# train = train.drop(['listing_id','make_model_combined','svd9_category','suggested_price','price'],axis=1)
# test = test.drop(['listing_id','make_model_combined','suggested_price','svd9_category'],axis=1)

X_train,X_test,y_train,y_test = train_test_split(train,label,test_size=0.2,random_state=111)
data_train = xgb.DMatrix(X_train.fillna(-1), label=y_train)
data_test = xgb.DMatrix(X_test.fillna(-1), label=y_test)
data_rmse_test = xgb.DMatrix(X_test.fillna(-1))
test_DMatrix = xgb.DMatrix(test.fillna(-1))
watch_list = [(data_test, 'eval'), (data_train, 'train')]

# train = pd.read_csv('train_data3.csv',index_col=0)
# test = pd.read_csv('test_data3.csv',index_col=0)
# feature_use=['make',
#  'model',
#  'type_of_vehicle',
#  'transmission',
#  'curb_weight',
#  'power',
#  'fuel_type',
#  'engine_cap',
#  'no_of_owners',
#  'depreciation',
#  'coe',
#  'dereg_value',
#  'mileage',
#  'omv',
#  'des1',
#  'des2',
#  'des3',
#  'des4',
#  'des5',
#  'des6',
#  'des7',
#  'fea1',
#  'fea2',
#  'fea3',
#  'fea4',
#  'fea5',
#  'fea6',
#  'fea7',
#  'acc1',
#  'acc2',
#  'acc3',
#  'acc4',
#  'acc5',
#  'acc6',
#  'acc7',
#  'engine_type',
#  'coe_validity',
#  'low_mileage',
#  'warranty',
#  'reg_year',
#  'reg_month',
#  'vintage cars',
#  'coe car',
#  'consignment car',
#  'sta evaluated car',
#  'sgcarmart warranty cars',
#  'low mileage car',
#  'electric cars',
#  'direct owner sale',
#  'parf car',
#  'opc car',
#  'premium ad car',
#  'hybrid cars',
#  'imported used vehicle',
#  'almost new car',
#  'rare & exotic',
#  'lifespan_year',
#  'lifespan_month',
#  'usable_days',
#  'car_age',
#  'make_target_mean',
#  'model_target_mean',
#  'type_of_vehicle_target_mean',
#  'fuel_type_target_mean',
#  'transmission_target_mean']

# X_train,X_test,y_train,y_test = train_test_split(train[feature_use],train['price'],test_size=0.02,random_state=111)
# data_train = xgb.DMatrix(X_train.fillna(-1), label=y_train)
# data_test = xgb.DMatrix(X_test.fillna(-1), label=y_test)
# data_rmse_test = xgb.DMatrix(X_test.fillna(-1))
# test_DMatrix = xgb.DMatrix(test.fillna(-1))
# watch_list = [(data_test, 'eval'), (data_train, 'train')]

def train_xgboost(trial):
    param = {
                'max_depth': trial.suggest_int("max_depth", 8, 30), 
                'eta': 0.01, 
                'silent': 1, 
                'gamma':trial.suggest_float("gamma", 0, 0.8),
                'objective': 'reg:linear',
                'reg_alpha':trial.suggest_float("reg_alpha", 0, 100),
                'reg_lambda':trial.suggest_float("reg_lambda", 0, 100),
#                 'num_boost_round':trial.suggest_int("num_boost_round", 800, 3000),
                'subsample':trial.suggest_float("subsample", 0.5, 1),
                'colsample_bytree':0.6,
                'min_child_weight':trial.suggest_int("min_child_weight", 1, 30)
            }
    print(param)
    bst = xgb.train(param, data_train, num_boost_round=trial.suggest_int("num_boost_round", 800, 3000),evals=watch_list)
    rmse = np.sqrt(mean_squared_error(y_test.tolist(),list(bst.predict(data_rmse_test))))
    res = {

                'Predicted':list(bst.predict(test_DMatrix))
            }
    res=DataFrame(res)
    res.to_csv('model_saving4/xgboost_res_'+str(trial.number)+'.csv')
    return rmse

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "xgboost-study4"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name, pruner=optuna.pruners.MedianPruner())
study.enqueue_trial(
    {
        "max_depth": 11,
        "gamma": 0,
        "reg_alpha": 1,
         "reg_lambda": 1,
        "num_boost_round": 1500,
        "subsample": 0.9,
        "min_child_weight": 3,
    }
)

'''
param = {
            'max_depth': 11, 
            'eta': 0.01, 
            'silent': 1, 
            'gamma':0,
            'objective': 'reg:linear',
            'reg_alpha':1,
            'reg_lambda':1,
            'num_boost_round':1500,
            'subsample':0.9,
            'colsample_bytree':0.6,
            'min_child_weight':3
        }
'''

import logging
import sys

# optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study.optimize(train_xgboost, n_trials=5000)