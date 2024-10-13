import pandas as pd

WINDOW_SIZE = 20000
GLOBAL_FEATURES = ["A"]
LOCAL_FEATURES = ["B"]
AOI_DF = pd.read_csv('./original-data/aoi-boxes.csv', delimiter=';', decimal=',')
