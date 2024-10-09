# import json
# import random
# import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from windows import start_sliding_window
from preprocessing import preprocess_gaze_data


def plot_gazes(participants, AOI_df, df):
    plt.figure(figsize=(10, 6))

    for i, participant in enumerate(participants):
        print(i, participant)
        participant_data = df.loc[df['Participant'] == participant]
        print(participant_data.head())
        if not participant_data.empty:
            num = AOI_df.loc[(AOI_df['Participant'] == participant), 'AOIGazes'].values[0]
            plt.scatter(participant_data['x'], participant_data['y'], label=f"{participant}: {num}", s=10)

    # Plot the gaze data to visualize
    plt.title(f"Gaze Points P{participants}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(title="Participants")
    plt.viridis()
    plt.grid()
    plt.show()


def main():
    # Preprocessing
    # preprocess_gaze_data("./original-test-data/raw_gaze_data.csv", to_file='processed_gaze_data.csv')

    # Windows
    gaze_data = pd.read_csv("processed_gaze_data.csv")
    start_sliding_window(1, data=gaze_data, window_size=20000)  # Window size in milliseconds => 20 seconds

    # ML
    # data = pd.read_csv('training/eyetracking_by_event.csv')
    # train_model(data)


if __name__ == "__main__":
    main()
