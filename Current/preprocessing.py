import operator
import os
import re

import pyreadr
import pandas as pd
from IPython.core.display_functions import display
from sklearn.preprocessing import OneHotEncoder


def filter_event(data: pd.DataFrame, participant: str, event: str):
    """
    Filter specific eye movement events from the given DataFrame.

    Args:
    data (pd.DataFrame): DataFrame containing eye movement events.
    participant (str): Participant identifier.
    event (str): Type of eye movement event to filter.

    Returns:
    pd.DataFrame: Filtered DataFrame containing only the specified eye movement events.
    """
    # Filter events based on event type
    match event:
        case "Fixations":
            pp_events = data.loc[data["v1"].isin(["Fixation L", "Fixation R"])].copy(deep=True)
            pp_events.insert(1, "v1.5", "")
            pp_events[['v1', 'v1.5']] = pp_events['v1'].str.split(' ', expand=True)
        case "Saccades":
            pp_events = data.loc[data["v1"].isin(["Saccade L", "Saccade R"])].copy(deep=True)
            pp_events.insert(1, "v1.5", "")
            pp_events[['v1', 'v1.5']] = pp_events['v1'].str.split(' ', expand=True)
        case "Blinks":
            pp_events = data.loc[data["v1"].isin(["Blink L", "Blink R"])].copy(deep=True)
            pp_events.insert(1, "v1.5", "")
            pp_events[['v1', 'v1.5']] = pp_events['v1'].str.split(' ', expand=True)
        case "Userevents":
            pp_events = data[data["v1"] == "UserEvent"].copy(deep=True)
            pp_events["v5"] = [re.sub(pattern=r'# Message: ', repl='', string=list(pp_events["v5"])[i]) for i in
                               range(len(pp_events["v5"]))]

    pp_events["Participant"] = participant

    return pp_events


def preprocess_gaze_data(filepath: str, verbose: bool = True, to_file: str = None) -> pd.DataFrame:
    """
    Preprocesses gaze data from a CSV file by renaming participants and calculating cumulative time for each paragraph.

    :param filepath: Path to the CSV file containing gaze data.
    :param verbose:  If True, prints progress messages. Defaults to True.
    :param to_file: If provided, the processed data will be saved to this file instead of returned. Defaults to None.

    :return: If `to_file` is None, returns the processed DataFrame. Otherwise,
        it saves the processed DataFrame to the specified filepath and then returns it.
    """
    if verbose:
        print(f"Loading data from {filepath}...")

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

        # Add the total time to the new time column
        time_column += list(participant_data['time'])

    data['time'] = list(map(lambda x: x*1000, time_column))

    if verbose:
        print("Processing complete.")

    if to_file:
        data.to_csv(to_file, index=False)
        if verbose:
            print("Saving complete.")
        return data
    else:
        return data
