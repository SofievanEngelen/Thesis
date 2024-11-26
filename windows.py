import os
import pandas as pd
from typing import Tuple, List, Optional
from feature_extraction import compute_features
from constants import log_message


def create_window(df: pd.DataFrame, start_time: int, end_time: int, verbose: bool = True) -> pd.DataFrame:
    """
    Create a data window for a specific time range and process it to detect fixations.

    :param df: The input DataFrame containing participant data.
    :param start_time: The start time for the window (in milliseconds).
    :param end_time: The end time for the window (in milliseconds).
    :param verbose: If True, log progress to the console. Defaults to True.
    :return: A DataFrame containing detected fixation events for the time window.
    """
    # Filter data within the given time window
    window_data = df[(df['time'] > start_time) & (df['time'] < end_time)]

    # Set trial to start_time in seconds
    window_data['trial'] = int(start_time / 1000)

    # Reset index for consistency
    window_data.reset_index(drop=True, inplace=True)

    # Optionally save window data for debugging
    if verbose:
        window_data.to_csv('window_data.csv', index=False)
        log_message(f"Prepared window data. Trial: {int(start_time / 1000)}", verbose)

    # Detect gaze events in the window data
    window_features = compute_features(window_data)

    if verbose:
        log_message(f"Computed window features. Trial: {int(start_time / 1000)}", verbose)

    return window_features


def training_windows(df: pd.DataFrame, window_size: int, to_file: Optional[str] = None, verbose: bool = True) \
        -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate training data using sliding windows from paragraphs of interest.

    :param df: DataFrame containing participant data.
    :param window_size: The size of the sliding window (in milliseconds).
    :param to_file: An optional path to save the resulting DataFrame. If the file exists, it will be read instead.
    :param verbose: If True, log progress to the console. Defaults to True.
    :return: A tuple containing the training DataFrame and a list of dropped windows.
    """
    if to_file and os.path.isfile(to_file):
        log_message(f"File already exists at {to_file}, reading existing file...", verbose)
        return pd.read_csv(to_file), []

    probe_paragraphs = [4, 10, 15, 20, 26, 30, 36]
    train_features_df = []
    dropped_windows = []

    for participant_id in range(1, len(df['Participant'].unique()) + 1):
        log_message(f"Processing Participant {participant_id}...", verbose)
        participant_data = df[df['Participant'] == participant_id]

        for paragraph in probe_paragraphs:
            try:
                # Get end time for the current paragraph
                paragraph_data = participant_data[participant_data['Paragraph'] == paragraph]
                if paragraph_data.empty:
                    log_message(f"No data for participant {participant_id}, paragraph {paragraph}.", verbose)
                    continue

                end_time = paragraph_data['time'].iloc[-1]
                start_time = end_time - window_size

                # Compute features for the window
                features_df = create_window(participant_data, start_time, end_time, verbose)

                if features_df.empty:
                    dropped_windows.append(f"Participant {participant_id}, Paragraph {paragraph}")
                    log_message(f"Empty features for Participant {participant_id}, Paragraph {paragraph}. Dropped.",
                                verbose)
                else:
                    train_features_df.append(features_df)

            except Exception as e:
                dropped_windows.append(f"Participant {participant_id}, Paragraph {paragraph}")
                log_message(f"Error processing Participant {participant_id}, Paragraph {paragraph}: {str(e)}", verbose)

    # Combine all feature DataFrames
    train_df = pd.concat(train_features_df, ignore_index=True)

    if to_file:
        train_df.to_csv(to_file, index=False)
        log_message(f"Training data saved to {to_file}.", verbose)

    return train_df, dropped_windows
