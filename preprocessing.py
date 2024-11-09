import os
import re

import numpy as np
import pandas as pd
from constants import log_message, WINDOW_MAPPING


def convert_cartesian_to_pixels(df):
    """
    Convert Cartesian coordinates in a DataFrame to pixel coordinates.

    :param df: DataFrame with 'x' and 'y' columns.
    :return: DataFrame with 'x_pixel' and 'y_pixel' columns.
    """
    # Apply the transformation formulas
    df['x_pixel'] = df['x'] * df['WinHeight'] + (df['WinWidth'] / 2)
    df['y_pixel'] = (df['WinHeight'] / 2) - (df['y'] * df['WinHeight'])

    return df


def rename_participants(df: pd.DataFrame, verbose: bool = True):
    # Participants
    unique_participants = df['Participant'].unique()
    log_message(f"Number of unique participants: {len(unique_participants)}", verbose)

    # Create a mapping for renaming participants
    participant_mapping = {p: str(i + 1) for i, p in enumerate(unique_participants)}

    # Rename participants using the mapping
    df['Participant'] = df['Participant'].replace(participant_mapping)

    log_message("Participants renamed.", verbose)

    return df, unique_participants


def preprocess_gaze_data(filepath: str, verbose: bool = True, to_file: str = None) -> pd.DataFrame:
    """
    Preprocesses gaze Data from a CSV file by renaming participants, converting x and y from Carthesian coordinates to
    pixels, and calculating cumulative time for each paragraph.

    :param filepath: Path to the CSV file containing gaze Data.
    :param verbose:  If True, logs progress messages. Defaults to True.
    :param to_file: If provided, the processed Data is saved to this filepath. Defaults to None.
# Check if any fixation duration exceeds 10 seconds
    if not fixation_df.empty and fixation_df['duration'].max() > 10:
        # If any fixation is longer than 10 seconds, drop the window
        return pd.DataFrame()  # Return an empty DataFrame to indicate no features are computed
    :return: If `to_file` is None, returns the processed DataFrame. Otherwise,
        it saves the processed DataFrame to the specified filepath and then returns it.
    """
    if to_file and os.path.isfile(to_file):
        log_message("That file already exists, reading from existing file...", verbose)
        return pd.read_csv(to_file)

    log_message(f"Loading Data from {filepath}...", verbose)

    data = pd.read_csv(filepath)

    log_message(f"Data loaded.", verbose)

    data, unique_participants = rename_participants(data)
    # Create a mapping for renaming participants
    participant_mapping = {p: str(i + 1) for i, p in enumerate(unique_participants)}

    # Iterate through unique participants
    time_column = []

    for p in unique_participants:
        log_message(f"Calculating total time for participant {participant_mapping[p]}...", verbose)

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

    log_message("Processing complete.", verbose)

    if to_file:
        data.to_csv(to_file, index=False, header=True)
        log_message("Saving complete.", verbose)
        return data
    else:
        return data


def compute_MW_score(df: pd.DataFrame) -> int:
    """
    Computes the average MW score.
    """
    pass


def preprocess_probe_data(filepath: str,
                          dropped_windows: dict[str, list[int]],
                          verbose: bool = True,
                          to_file: str = None) -> pd.DataFrame:
    """
    Preprocesses the probe data to rename the participants to legible names and drops the windows that had insufficient
    data.
    :param filepath: Path to the raw probe data.
    :param dropped_windows: Windows to be dropped from the probe data.
    :param verbose: Boolean indicating if process messages should be printed to the console.
    :param to_file: Path to which the processed probe data should be saved.
    :return: Processed probe data in a pandas dataframe.
    """
    if to_file and os.path.isfile(to_file):
        log_message("That file already exists, reading from existing file...", verbose)
        return pd.read_csv(to_file)

    data, unique_participants = rename_participants(pd.read_csv(filepath))

    drop_indices = []
    for participant, probe_list in dropped_windows.items():
        log_message(f"Processing participant {participant} with probes {probe_list}", verbose)
        for probe in probe_list:
            index = data.loc[(data['Participant'].astype(int) == int(participant)) & (
                        data['Probe'].astype(int) == int(WINDOW_MAPPING[probe]))].index
            drop_indices += list(index.values)

    print(drop_indices, len(drop_indices))

    # Drop collected indices in a single operation
    data.drop(index=drop_indices, inplace=True)
    log_message(f"Dropped {len(drop_indices)} windows.", verbose)

    # data["MW-score"] = compute_MW_score(data)

    if to_file:
        data.to_csv(to_file, index=False, header=True)
        log_message("Saving complete.", verbose)
        return data
    else:
        return data
