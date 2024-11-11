import re
from collections import defaultdict
from typing import List

import pandas as pd
from matplotlib import pyplot as plt
from windows import training_windows
from preprocessing import preprocess_gaze_data, preprocess_probe_data
from constants import *
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from feature_extraction import compute_features
import ML
import numpy as np
from sklearn import svm, datasets
from ML import train_and_evaluate_models


def plot_gazes(participants, AOI_df, df):
    plt.figure(figsize=(10, 6))

    for i, participant in enumerate(participants):
        print(i, participant)
        participant_data = df.loc[df['Participant'] == participant]
        print(participant_data.head())
        if not participant_data.empty:
            num = AOI_df.loc[(AOI_df['Participant'] == participant), 'AOIGazes'].values[0]
            plt.scatter(participant_data['x'], participant_data['y'], label=f"{participant}: {num}", s=10)

    # Plot the gaze Data to visualize
    plt.title(f"Gaze Points P{participants}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(title="Participants")
    plt.viridis()
    plt.grid()
    plt.show()


def main():
    pass
    # Preprocessing
    # processed_gaze_data = preprocess_gaze_data(filepath=RAW_GAZE_DF_PATH,
    #                                            to_file=PRC_GAZE_DF_PATH)

    # dropped_windows = {'2': [4], '4': [4, 10, 15, 20, 30], '5': [4, 10, 15, 30, 36], '6': [4, 10, 26], '10': [4, 15, 30],
    #                    '11': [15, 26, 30], '13': [26, 36], '16': [4, 15], '20': [4, 10, 15, 20, 26, 30, 36], '21': [30],
    #                    '23': [4, 10, 20, 26, 30, 36], '26': [20, 26], '27': [4, 10, 15, 26, 30, 36], '28': [10, 30],
    #                    '29': [4, 15, 30], '30': [4, 10, 15, 20, 26, 30, 36], '32': [36],
    #                    '33': [4, 10, 15, 20, 26, 30, 36], '35': [10, 15, 20, 26], '36': [10, 15, 20, 26], '37': [30, 36],
    #                    '38': [30], '40': [10], '41': [4, 15, 36], '42': [36], '43': [4, 15, 20, 26, 30, 36],
    #                    '44': [10, 15], '46': [4, 10, 15, 20, 26, 30, 36], '48': [10, 20, 26, 36],
    #                    '49': [4, 10, 20, 26, 30, 36], '50': [4, 10, 15, 20, 26, 30, 36], '51': [4, 26], '52': [4],
    #                    '55': [4, 10, 15, 20, 26, 30, 36], '57': [36], '58': [4, 10, 15, 20, 26, 30, 36],
    #                    '59': [15, 26, 36], '60': [10], '61': [4, 10, 15, 26, 30, 36], '63': [20], '64': [4, 26],
    #                    '66': [4, 10, 15, 20, 26, 30, 36], '68': [10], '69': [4, 20, 26, 30, 36], '70': [15],
    #                    '71': [4, 10, 15, 26], '72': [10, 20, 26, 36], '75': [15, 20, 26, 30, 36],
    #                    '76': [10, 20, 30, 36], '78': [4, 10, 15, 26, 30, 36], '79': [4, 10, 15, 20, 26, 30, 36],
    #                    '80': [30], '81': [4, 10, 15, 30], '84': [10, 26], '85': [4, 10, 15, 20, 26, 30, 36], '86': [30],
    #                    '87': [10], '89': [4, 10, 15, 20, 26, 30, 36], '91': [15], '92': [20, 36],
    #                    '93': [4, 10, 15, 20, 26, 30, 36], '94': [10, 15, 20, 26], '96': [4, 15, 30, 36],
    #                    '97': [4, 10, 15, 20, 26, 30, 36], '99': [4, 10, 15, 20, 36], '100': [15],
    #                    '101': [4, 15, 20, 26, 36], '102': [4], '103': [30], '104': [4, 10, 15, 26, 30, 36],
    #                    '105': [4, 10, 26, 30, 36], '106': [4, 10, 15, 20, 30, 36], '108': [4, 15, 20, 26, 30, 36],
    #                    '109': [4, 10, 15, 20, 26, 30, 36], '110': [4, 15, 20, 30], '111': [36],
    #                    '112': [4, 10, 15, 20, 26, 30, 36], '113': [10, 15], '115': [4, 15], '118': [4, 10, 20, 26, 36],
    #                    '119': [4, 15, 20, 26, 30, 36], '122': [30, 36], '123': [4, 26, 30, 36],
    #                    '124': [4, 20, 26, 30, 36], '125': [4, 10, 15, 20, 26, 30, 36],
    #                    '126': [4, 10, 15, 20, 26, 30, 36], '128': [26, 36], '129': [15], '130': [10, 36],
    #                    '134': [4, 10, 15, 20, 26, 30, 36], '135': [4, 10, 15, 20, 26, 30, 36],
    #                    '136': [4, 15, 20, 26, 30, 36], '137': [10, 15, 20, 36], '138': [30], '140': [4, 15, 20, 30, 36],
    #                    '144': [10], '145': [4, 10, 15, 20, 26], '147': [4, 26], '149': [4, 15, 20, 26],
    #                    '151': [4, 10, 15, 20], '152': [10, 15, 20, 26, 30, 36], '153': [4, 20, 26, 36],
    #                    '156': [4, 10, 15, 20, 30, 36], '157': [20, 26], '159': [10, 15, 26], '161': [4, 15],
    #                    '162': [4, 10, 20, 26, 36], '164': [4, 10, 15, 20, 26, 30, 36], '165': [26], '167': [4, 10, 36],
    #                    '168': [10], '169': [4, 10, 15, 20, 26, 30], '171': [4, 10, 15, 20, 26, 30, 36],
    #                    '174': [4, 10, 15, 20, 26, 30, 36], '175': [4, 26], '176': [30], '177': [10, 15, 26, 30],
    #                    '178': [26, 30, 36], '179': [4, 15, 20, 26, 30], '181': [4, 15, 20, 26, 30], '183': [4],
    #                    '184': [4, 10, 15, 20, 26, 30, 36], '185': [26], '187': [10, 26, 30, 36], '188': [4, 26, 30, 36],
    #                    '189': [10, 26], '190': [20], '191': [10, 15, 20, 26, 30, 36], '193': [36], '195': [20],
    #                    '196': [10, 26, 36], '198': [4, 10, 15, 20, 26, 30, 36], '200': [4, 10, 15, 20, 26, 30, 36],
    #                    '203': [10, 15, 20, 26, 30, 36], '204': [4, 10, 15, 20, 26, 30, 36], '205': [4, 15, 36],
    #                    '206': [4, 15, 36], '207': [4, 10, 26, 36], '209': [4, 15, 20], '211': [10, 15, 26, 36],
    #                    '213': [10, 30], '218': [20, 26], '219': [15, 20], '222': [4, 10, 15, 20, 26, 30, 36],
    #                    '223': [10, 15, 36], '224': [20, 36], '225': [15], '226': [4, 10, 20, 26, 30, 36],
    #                    '227': [15, 26, 36], '228': [36], '229': [4, 26, 30, 36], '232': [10, 15], '233': [10, 15],
    #                    '234': [10, 26], '236': [15, 20, 36], '237': [10, 15, 36], '238': [4, 26], '241': [15, 36],
    #                    '243': [4, 10, 15, 20, 26, 30, 36], '244': [4, 10, 15, 20, 26, 30, 36], '246': [36], '247': [30],
    #                    '248': [15, 20, 26], '249': [15, 26, 30], '252': [4, 10, 20, 26, 30, 36],
    #                    '253': [4, 10, 15, 20, 26, 30, 36], '254': [4, 36], '255': [4, 10, 15, 20, 30, 36],
    #                    '256': [15, 26, 30], '259': [20, 26, 36], '261': [4, 15, 20, 26, 30], '262': [4],
    #                    '265': [4, 10, 20, 26, 30, 36], '266': [4, 10, 15, 20, 30], '268': [4, 10, 15, 20, 26, 30, 36],
    #                    '271': [30], '272': [15], '273': [4, 10, 15, 20, 26, 30, 36], '274': [4, 10, 15, 36], '278': [10],
    #                    '279': [10, 15, 20, 30], '280': [26, 30], '284': [4, 10, 15, 20, 26, 30], '285': [15, 36],
    #                    '286': [4, 10, 15, 20, 26, 30, 36], '288': [4, 10, 15, 20, 26, 30, 36], '289': [4, 10, 15],
    #                    '290': [10, 20, 30, 36], '291': [10, 15, 20, 26, 30], '293': [30, 36], '296': [15, 20],
    #                    '297': [4, 10, 15, 20, 26, 30], '298': [4, 10, 15, 26, 30, 36], '299': [4, 26, 36],
    #                    '300': [20, 30, 36], '301': [10, 15, 20, 26, 30, 36], '305': [10, 20, 26], '306': [20],
    #                    '307': [4, 10, 15], '310': [4, 26, 30], '312': [4, 10, 15, 20, 30], '313': [4, 20, 26, 30],
    #                    '314': [30], '315': [10, 20, 36], '316': [4, 10, 15, 20, 26, 30, 36], '318': [20, 26, 36],
    #                    '319': [30], '323': [10], '324': [4, 10, 15, 20, 26, 30, 36], '326': [4, 10, 15, 20, 26, 30, 36],
    #                    '327': [20, 26, 30, 36], '329': [10, 15, 20, 26, 30], '333': [4, 20],
    #                    '334': [10, 15, 20, 26, 30, 36], '335': [26], '337': [4, 10, 15, 20, 26, 30, 36],
    #                    '338': [4, 10, 20, 30, 36], '340': [4, 10, 15, 26, 30, 36], '342': [4, 10, 15, 26, 30, 36],
    #                    '343': [4, 10, 15, 20, 26, 30, 36], '344': [4, 10, 15, 20, 26, 30, 36], '345': [10, 20],
    #                    '347': [20, 26], '349': [20], '351': [4, 10, 15, 20, 30, 36]}
    # print(len(dropped_windows))
    # processed_probe_data = preprocess_probe_data(filepath='Data/original-data/probe_data.csv',
    #                                              dropped_windows=dropped_windows,
    #                                              verbose=True,
    #                                              to_file="Data/processed-data/processed_probe_data.csv")

    # Training windows
    # test_window = pd.read_csv("/Users/sofie/dev/Python/Uni/Thesis/Thesis - code/Data/processed-data/test_window.csv")
    # compute_features(test_window).to_csv('/Users/sofie/dev/Python/Uni/Thesis/Thesis - code/Data/processed-data/'
    #                                      'test_window_features.csv', header=True, index=False)

    # _, dropped_windows = training_windows(processed_gaze_data, WINDOW_SIZE, to_file=TRAIN_FEATURES_PATH)
    # print(dropped_windows)
    # print(len(dropped_windows))
    # for dropped_window in dropped_windows:
    #     probe

    # train_windows_df = pd.concat(train_windows)
    # train_windows_df.to_csv('train_windows_features.csv', index=False, header=True)
    probe_data = pd.read_csv('Data/processed-data/processed_probe_data.csv')
    train_windows = pd.read_csv('Data/processed-data/train_windows_features.csv')

    # print("train windows", len(train_windows))
    # probes = pd.read_csv('Data/processed-data/processed_probe_data.csv')
    # print(len(probes))
    #

    # optimal_n_features, best_rfe_model = select_optimal_features_with_rfe(X_pca_df, processed_probe_data,
    #                                                                       max_features=10)
    # print(optimal_n_features)
    # X_rfe_selected = RFE_selection(X_pca_df, processed_probe_data, n_features_rfe=10)
    #
    # print(X_rfe_selected.shape)
    # rfe_features = ['PC1', 'PC10', 'PC11', 'PC14', 'PC21', 'PC29', 'PC32', 'PC35', 'PC41', 'PC45']

    # models = ["LinearSVM"]
    models = ["LinearSVM", "RandomForest", "LogReg", "NaiveBates", "XGBoost"]

    train_and_evaluate_models(models, train_windows, probe_data['TUT'])

    # train_windows_df = training_windows('./processed_gaze_data.csv', constants.WINDOW_SIZE, 'train_windows.csv')
    # events_df, saccade_df = detect_events(df)
    # print(events_df)
    # events_df.to_csv("events_test.csv")
    # saccade_df.to_csv("sac_test.csv")
    # print(compute_global_features(events_df, saccade_df))

    # Windows
    # start_sliding_window(1, Data=processed_gaze_data, window_size=WINDOW_SIZE)  # Window size in milliseconds => 20 seconds

    # ML
    # Data = pd.read_csv('training/eyetracking_by_event.csv')
    # train_model(Data, model)


if __name__ == "__main__":
    main()
