import operator
import os
import re

import pyreadr
import pandas as pd
from IPython.core.display_functions import display
from sklearn.preprocessing import OneHotEncoder


def convert_cartesian_to_pixels(df):
    """
    Convert Cartesian coordinates in a DataFrame to pixel coordinates.

    :param df: DataFrame with 'x' and 'y' columns.
    :param winwidth: Screen width in pixels.
    :param winheight: Screen height in pixels.
    :return: DataFrame with 'x_pixel' and 'y_pixel' columns.
    """
    # Apply the transformation formulas
    df['x_pixel'] = df['x'] * df['WinHeight'] + (df['WinWidth'] / 2)
    df['y_pixel'] = (df['WinHeight'] / 2) - (df['y'] * df['WinHeight'])

    return df


def preprocess_gaze_data(filepath: str, verbose: bool = True, to_file: str = None) -> pd.DataFrame:
    """
    Preprocesses gaze Data from a CSV file by renaming participants, converting x and y from Carthesian coordinates to
    pixels, and calculating cumulative time for each paragraph.

    :param filepath: Path to the CSV file containing gaze Data.
    :param verbose:  If True, prints progress messages. Defaults to True.
    :param to_file: If provided, the processed Data will be saved to this file instead of returned. Defaults to None.

    :return: If `to_file` is None, returns the processed DataFrame. Otherwise,
        it saves the processed DataFrame to the specified filepath and then returns it.
    """
    if to_file and os.path.isfile(to_file):
        if verbose:
            print("That file already exists, reading from existing file...")
        return pd.read_csv(to_file)

    if verbose:
        print(f"Loading Data from {filepath}...")

    data = pd.read_csv(filepath)

    if verbose:
        print(f"Data loaded.")

    # Participants
    unique_participants = data['Participant'].unique()
    if verbose:
        print(f"Number of unique participants: {len(unique_participants)}")

    # Create a mapping for renaming participants
    participant_mapping = {p: str(i + 1) for i, p in enumerate(unique_participants)}

    # Rename participants using the mapping
    data['Participant'] = data['Participant'].replace(participant_mapping)

    if verbose:
        print("Participants renamed.")

    # Iterate through unique participants
    time_column = []

    for p in unique_participants:
        if verbose:
            print(f"Calculating total time for participant {participant_mapping[p]}...")

        # Filter rows for the current participant
        participant_data = data[data['Participant'] == participant_mapping[p]]

        # Cumulative time over all paragraphs
        prev_time = 0

        for para in data['Paragraph'].unique():
            participant_data.loc[participant_data['Paragraph'] == para, 'time'] += prev_time

            prev_time = participant_data.loc[participant_data['Paragraph'] == para, 'time'].iloc[-1]

        # Add the values to the new columns
        time_column += list(participant_data['time'])

    data['time'] = list(map(lambda x: x * 1000, time_column))

    if verbose:
        print("Processing complete.")

    if to_file:
        data.to_csv(to_file, index=False, header=True)
        if verbose:
            print("Saving complete.")
        return data
    else:
        return data
