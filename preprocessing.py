import os
import pandas as pd
from constants import log_message


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

    # Participants
    unique_participants = data['Participant'].unique()
    log_message(f"Number of unique participants: {len(unique_participants)}", verbose)

    # Create a mapping for renaming participants
    participant_mapping = {p: str(i + 1) for i, p in enumerate(unique_participants)}

    # Rename participants using the mapping
    data['Participant'] = data['Participant'].replace(participant_mapping)

    log_message("Participants renamed.", verbose)

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
