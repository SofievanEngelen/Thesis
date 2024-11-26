import os
import pandas as pd
from typing import Tuple, Dict, List, Optional
from constants import log_message, WINDOW_MAPPING


def rename_participants(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Renames participants in the DataFrame to consecutive numeric IDs.

    :param df: The input DataFrame containing a 'Participant' column.
    :param verbose: A boolean flag for verbosity.
    :return: A tuple containing the updated DataFrame and a list of unique participants.
    """
    unique_participants = df['Participant'].unique()
    log_message(f"Number of unique participants: {len(unique_participants)}", verbose)

    participant_mapping = {p: str(i + 1) for i, p in enumerate(unique_participants)}
    df['Participant'] = df['Participant'].replace(participant_mapping)

    log_message("Participants renamed.", verbose)
    return df, unique_participants


def preprocess_gaze_data(filepath: str, verbose: bool = True, to_file: Optional[str] = None) -> pd.DataFrame:
    """
    Preprocesses gaze data from a CSV file by renaming participants, converting time to milliseconds,
    and calculating cumulative time for each paragraph.

    :param filepath: Path to the CSV file containing gaze data.
    :param verbose: Boolean flag for verbosity.
    :param to_file: Optional filepath to save the processed data. Defaults to None.
    :return: Processed DataFrame.
    """
    if to_file and os.path.isfile(to_file):
        log_message(f"File already exists at {to_file}, reading from existing file...", verbose)
        return pd.read_csv(to_file)

    log_message(f"Loading data from {filepath}...", verbose)
    data = pd.read_csv(filepath)
    log_message("Data loaded.", verbose)

    data, unique_participants = rename_participants(data, verbose)

    # Compute cumulative time for each participant and paragraph
    data['time'] = 0  # Initialize time column

    for participant in unique_participants:
        log_message(f"Calculating cumulative time for participant {participant}...", verbose)

        participant_data = data[data['Participant'] == participant]
        prev_time = 0

        for paragraph in participant_data['Paragraph'].unique():
            paragraph_indices = participant_data[participant_data['Paragraph'] == paragraph].index
            data.loc[paragraph_indices, 'time'] += prev_time
            prev_time = data.loc[paragraph_indices, 'time'].iloc[-1]

    # Convert time to milliseconds
    data['time'] = data['time'] * 1000

    log_message("Preprocessing complete.", verbose)

    if to_file:
        data.to_csv(to_file, index=False)
        log_message(f"Processed data saved to {to_file}.", verbose)

    return data


def preprocess_probe_data(filepath: str,
                          dropped_windows: Dict[str, List[int]],
                          verbose: bool = True,
                          to_file: Optional[str] = None) -> pd.DataFrame:
    """
    Preprocesses probe data by renaming participants and removing windows with insufficient data.

    :param filepath: Path to the raw probe data CSV file.
    :param dropped_windows: Dictionary mapping participants to the list of dropped probes.
    :param verbose: A boolean flag for verbosity.
    :param to_file: Optional filepath to save the processed data. Defaults to None.
    :return: Processed probe data as a DataFrame.
    """
    if to_file and os.path.isfile(to_file):
        log_message(f"File already exists at {to_file}, reading from existing file...", verbose)
        return pd.read_csv(to_file)

    log_message(f"Loading probe data from {filepath}...", verbose)
    data, _ = rename_participants(pd.read_csv(filepath), verbose)

    # Identify indices of rows to drop based on dropped_windows
    drop_indices = []
    for participant, probe_list in dropped_windows.items():
        log_message(f"Processing participant {participant} with dropped probes: {probe_list}", verbose)
        for probe in probe_list:
            probe_indices = data.loc[
                (data['Participant'].astype(int) == int(participant)) &
                (data['Probe'].astype(int) == int(WINDOW_MAPPING[probe]))
            ].index
            drop_indices.extend(probe_indices)

    log_message(f"Total windows to drop: {len(drop_indices)}", verbose)
    data.drop(index=drop_indices, inplace=True)
    log_message(f"Dropped {len(drop_indices)} windows.", verbose)

    # Placeholder for MW-score computation
    # data["MW-score"] = compute_MW_score(data)

    if to_file:
        data.to_csv(to_file, index=False)
        log_message(f"Processed probe data saved to {to_file}.", verbose)

    return data
