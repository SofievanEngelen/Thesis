import os
import time

import pandas as pd
from IPython.core.display_functions import display
from pandas import DataFrame, unique
from feature_extraction import compute_features


def create_window(df: pd.DataFrame, start_time: int, end_time: int) -> (pd.DataFrame, pd.DataFrame):
    """
        Create a Data window for a specific time range and process it to detect fixations.

        :param df: The input DataFrame containing participant Data.
        :param start_time: The start time for the window (in milliseconds).
        :param end_time: The end time for the window (in milliseconds).

        :return: A DataFrame containing detected fixation events for the time window.
    """
    # filter out Data in given window
    window_data = df[(df['time'] > start_time) & (df['time'] < end_time)]

    # set trial to  start_time in seconds
    window_data.loc[:, 'trial'] = int(start_time / 1000)

    window_data.reset_index(inplace=True)
    window_data.to_csv('window_data.csv', index=False)

    # detect gaze events in window Data
    window_features = compute_features(window_data)

    return window_features


def start_sliding_window(participant: int, data: DataFrame, window_size: int) -> None:
    """
        Process Data using a sliding window approach for a specific participant.

        :param participant: The ID of the participant whose Data is being processed.
        :param data: The input DataFrame containing participant Data.
        :param window_size: The size of the sliding window (in milliseconds).

        :return: The function prints the window Data for each sliding window step.
    """

    start_time = 0

    data['Participant'] = data['Participant'].astype(int)
    p_data = data[data['Participant'] == participant]

    if p_data.empty:
        raise Exception(f'No participant {participant} found')

    while True:
        end_time = start_time + window_size
        window_df = create_window(p_data, start_time, end_time)
        print(window_df.head())
        start_time += 1000

        if start_time > (int(p_data['time'].max()) - window_size + 1000):
            break


def training_windows(df: pd.DataFrame, window_size: int, features: bool = False, to_file: str = None) -> pd.DataFrame:
    """
        Generate training Data using sliding windows from paragraphs of interest.

        :param df: Dataframe containing participant Data.
        :param window_size: The size of the sliding window (in milliseconds).
        :param features: Boolean indicating whether to compute the features of the probe window
        :param to_file: The path to save the resulting DataFrame. If the file already exists, the function will read
        from it.

        :return: A DataFrame containing training Data with detected fixations in each window.
    """
    # if to_file and os.path.isfile(to_file):
    #     return pd.read_csv(to_file)

    probe_paragraphs = [4, 10, 15, 20, 26, 30, 36]

    # 5, 11-30, 20, 23, 26?,

    # train_window_data = []
    train_features_df = []
    for p in range(26, len(unique(df['Participant'])) + 1):
        print("Participant ", p)
        p_df = df[df['Participant'] == p]
        # find the last entry of the paragraph = endtime
        for paragraph in probe_paragraphs:
            end_time = p_df.loc[p_df['Paragraph'] == paragraph, 'time'].iloc[-1]
            start_time = end_time - window_size

            features_df = create_window(p_df, start_time, end_time)
            print(features_df)

            train_features_df.append(features_df)

    train_df = pd.concat(train_features_df)

    if to_file:
        train_df.to_csv(to_file, index=False, header=True)

    return train_df

