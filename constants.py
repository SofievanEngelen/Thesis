WINDOW_SIZE = 20000
MIN_FIXATION_DURATION = 10000
RANDOM_SEED = 42
GLOBAL_FEATURES = ['participant', 'group_id', 'num_fixations',
                   'fixation_duration_mean', 'fixation_duration_median',
                   'fixation_duration_std', 'fixation_duration_max',
                   'fixation_duration_min', 'fixation_duration_range',
                   'fixation_duration_kurtosis', 'fixation_duration_skewness',
                   'fixation_dispersion_mean', 'fixation_dispersion_median',
                   'fixation_dispersion_std', 'fixation_dispersion_max',
                   'fixation_dispersion_min', 'fixation_dispersion_range',
                   'fixation_dispersion_kurtosis', 'fixation_dispersion_skewness',
                   'num_saccades', 'saccade_duration_mean', 'saccade_duration_median',
                   'saccade_duration_std', 'saccade_duration_max', 'saccade_duration_min',
                   'saccade_duration_range', 'saccade_duration_kurtosis',
                   'saccade_duration_skewness', 'saccade_amplitude_mean',
                   'saccade_amplitude_median', 'saccade_amplitude_std',
                   'saccade_amplitude_max', 'saccade_amplitude_min',
                   'saccade_amplitude_range', 'saccade_amplitude_kurtosis',
                   'saccade_amplitude_skewness', 'saccade_angle_mean',
                   'saccade_angle_median', 'saccade_angle_std', 'saccade_angle_max',
                   'saccade_angle_min', 'saccade_angle_range', 'saccade_angle_kurtosis',
                   'saccade_angle_skewness', 'horizontal_saccade_ratio',
                   'fixation_saccade_ratio', 'num_fixations_AOI']
LOCAL_FEATURES = ['participant', 'group_id', 'fixation_AOI_duration__mean', 'fixation_AOI_duration__median',
                  'fixation_AOI_duration__std', 'fixation_AOI_duration__max',
                  'fixation_AOI_duration__min', 'fixation_AOI_duration__range',
                  'fixation_AOI_duration__kurtosis', 'fixation_AOI_duration__skewness',
                  'fixation_AOI_dispersion_mean', 'fixation_AOI_dispersion_median',
                  'fixation_AOI_dispersion_std', 'fixation_AOI_dispersion_max',
                  'fixation_AOI_dispersion_min', 'fixation_AOI_dispersion_range',
                  'fixation_AOI_dispersion_kurtosis', 'fixation_AOI_dispersion_skewness']
WINDOW_MAPPING = {4: 1, 10: 2, 15: 3, 20: 4, 26: 5, 30: 6, 36: 7}


def log_message(message: str, verbose: bool) -> None:
    """
    Prints a message if verbose mode is enabled.

    :param message: The message to print.
    :param verbose: Flag indicating whether to print the message.
    """
    if verbose:
        print(message)


# Dataframes
AOI_DF_PATH = '/Users/sofie/dev/Python/Uni/Thesis/Thesis - code/Data/original-data/aoi-boxes.csv'
RAW_GAZE_DF_PATH = '/Users/sofie/dev/Python/Uni/Thesis/Thesis - code/Data/original-data/raw_gaze_data.csv'
PRC_GAZE_DF_PATH = '/Users/sofie/dev/Python/Uni/Thesis/Thesis - code/Data/processed-data/processed_gaze_data.csv'
TRAIN_FEATURES_PATH = '/Users/sofie/dev/Python/Uni/Thesis/Thesis - code/Data/processed-data/train_windows_features.csv'

# Model parameters
XGB_PARAMS = {'eta': [0.01, 0.05, 0.1],
              'n_estimators': [100, 500, 1000],
              'max_depth': [3, 7, 12],
              'min_child_weight': [1, 5, 10],
              'subsample': [0.6, 0.7, 0.8],
              'colsample_bytree': [0.4, 0.6, 0.8],
              'scale_pos_weight': ['(num_neg / num_pos)'],
              'lambda': [1],
              'alpha': [0],
              'gamma': [0],
              'objective': ['binary:logistic'],
              'tree_method': ['hist']}

LR_PARAMS = {'penalty': ['l2'],
             'class_weight': ['balanced'],
             'C': [0.0001, 0.01, 1],
             'max_iter': [500, 1000, 2000],
             'solver': ['liblinear', 'newton-cg', 'lbfgs', 'saga']}

RF_PARAMS = {'class_weight': ['balanced'],
             'n_estimators': [100, 200, 500],
             'max_depth': [None, 10, 30],
             'max_features': ['sqrt', 'log2', 10],
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 5, 10]}

LSVM_PARAMS = {'class_weight': ['balanced'],
               'C': [0.001, 0.1, 1]}
