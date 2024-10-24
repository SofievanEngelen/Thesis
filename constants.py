import pandas as pd

WINDOW_SIZE = 20000
GLOBAL_FEATURES = ["A"]
LOCAL_FEATURES = ["B"]

# Dataframes
# AOI_DF = pd.read_csv('./original-Data/aoi-boxes.csv', delimiter=';', decimal=',')
# RAW_GAZE_DF = pd.read_csv('./original-Data/raw_gaze_data.csv')
# PRC_GAZE_DF = pd.read_csv('./processed-Data/processed_gaze_data.csv')

# Model parameters
XGB_FINAL = {}
XGB_PARAMS = {'eta': [0.01, 0.05, 0.1, 0.3],
              'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
              'max_depth': [3, 5, 7, 9, 12],
              'min_child_weight': [1, 3, 5, 7],
              'subsample': [0.6, 0.7, 0.8, 0.9],
              'colsample_bytree': [0.3, 0.5, 0.7, 0.9],
              'lambda': [0.1, 1, 10],
              'alpha': [0, 0.5, 1],
              'gamma': [0, 0.1, 0.5, 1, 2, 5],
              'objective': 'binary:logistic',
              'tree_method': "hist"}

LR_FINAL = {}
LR_PARAMS = {'penalty': 'l2',
             'C': 1.0,
             'max_iterations': [100, 200, 500],
             'solver': ['liblinear', 'newton-cg', 'lbfgs', 'saga']}

RF_FINAL = {}
RF_PARAMS = {}

LSVM_FINAL = {}
LSVM_PARAMS = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
               'max_iter': 1000}

NB_FINAL = {}
NB_PARAMS = {}
