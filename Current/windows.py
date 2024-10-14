import os
import time

import pandas as pd
from IPython.core.display_functions import display
from pandas import DataFrame


def create_window(df: pd.DataFrame, start_time: int, end_time: int) -> pd.DataFrame:
    """
        Create a data window for a specific time range and process it to detect fixations.

        :param df: The input DataFrame containing participant data.
        :param start_time: The start time for the window (in milliseconds).
        :param end_time: The end time for the window (in milliseconds).

        :return: A DataFrame containing detected fixation events for the time window.
    """
    # filter out data in given window
    window_data = df[(df['time'] > start_time) & (df['time'] < end_time)]

    # set trial to  start_time in seconds
    window_data.loc[:, 'trial'] = int(start_time / 1000)

    # detect gaze events in window data
    # event_df = detect_fixations(window_data)

    return window_data


def start_sliding_window(participant: int, data: DataFrame, window_size: int) -> None:
    """
        Process data using a sliding window approach for a specific participant.

        :param participant: The ID of the participant whose data is being processed.
        :param data: The input DataFrame containing participant data.
        :param window_size: The size of the sliding window (in milliseconds).

        :return: The function prints the window data for each sliding window step.
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


def training_windows(df: pd.DataFrame, window_size: int, to_file: str = None) -> pd.DataFrame:
    """
        Generate training data using sliding windows from paragraphs of interest.

        :param df: Dataframe containing participant data.
        :param window_size: The size of the sliding window (in milliseconds).
        :param to_file: The path to save the resulting DataFrame. If the file already exists, the function will read
        from it.

        :return: A DataFrame containing training data with detected fixations in each window.
    """
    if to_file and os.path.isfile(to_file):
        return pd.read_csv(to_file)

    probe_paragraphs = [4, 10, 15, 20, 26, 30, 36]

    train_window_data = []

    # find the last entry of the paragraph = endtime
    for paragraph in probe_paragraphs:
        print(paragraph)
        end_time = df.loc[df['Paragraph'] == paragraph, 'time'].iloc[-1]
        start_time = end_time - window_size

        probe_df = create_window(df, start_time, end_time)
        train_window_data.append(probe_df)

    train_df = pd.concat(train_window_data)

    if to_file:
        train_df.to_csv(to_file, index=False)

    return train_df

