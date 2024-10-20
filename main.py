import time

import pandas as pd
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from pandas import unique

# from sklearn.datasets import make_classification

from windows import training_windows
from preprocessing import preprocess_gaze_data
from constants import *
from feature_extraction import compute_features


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
    # processed_gaze_data = preprocess_gaze_data(filepath="./original-Data/raw_gaze_data.csv",
    #                                            to_file='processed-Data/processed_gaze_data.csv')

    # Feature extraction
    test_window = pd.read_csv('window_data.csv')
    print(compute_features(test_window))

    # Training windows
    # df = pd.read_csv('processed-Data/processed_gaze_data.csv')
    # training_windows(df, WINDOW_SIZE, to_file='processed-Data/train_windows_features.csv')
    #     display(p_window_df)
    #     train_windows.append(p_window_df)
    #
    # train_windows_df = pd.concat(train_windows)
    # train_windows_df.to_csv('train_windows_features.csv', index=False, header=True)

    # X, y = make_classification(n_classes=2, class_sep=2,
    #                            weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    #                            n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    # train_model(X, y, "XGBoost", features="G")

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
