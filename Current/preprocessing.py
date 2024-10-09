import operator
import os
import re

import pyreadr
import pandas as pd
from IPython.core.display_functions import display
from sklearn.preprocessing import OneHotEncoder

# Define file paths
timing_data_dir = "training/training-data/Data Timings"
participant_data_dir = "training/training-data/Myrthe Faber/dat"
scores_path = "training/training-data/XML_data_cleaned.Rda"

# Define participants to be rejected
rejected_participants = {"p21", "p30", "p39", "p51", "p64", "p66", "p69", "p71", "p74", "p93"}


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


def process_participant_file(file_path: str):
    """
    Process a participant file containing eye movement data.

    Args:
    file_path (str): Path to the participant file.

    Returns:
    pd.DataFrame: Processed DataFrame containing eye movement data.
    str: Participant identifier.
    """
    # Read the participant file
    pp = pd.read_csv(file_path, header=None, names=[f"v{i}" for i in range(1, 18)], skiprows=4, sep='\t', dtype=str)

    # Extract participant and date info
    participant = pp.loc[1, 'v2'].lower()

    # Remove initial irrelevant lines
    pp_data_filtered = pp.iloc[20:].copy()

    return pp_data_filtered, participant


def process_train_participants(p_dir: str = None) -> None:
    """
    Process all participant files in the given directory.

    Args:
    p_dir (str): Directory containing participant files.

    Returns:
    pd.DataFrame: Concatenated DataFrame containing all fixations.
    pd.DataFrame: Concatenated DataFrame containing all saccades.
    pd.DataFrame: Concatenated DataFrame containing all blinks.
    pd.DataFrame: Concatenated DataFrame containing all user events.
    """
    if p_dir is None:
        p_dir = participant_data_dir

    fixations = []
    saccades = []
    blinks = []
    userevents = []

    # Process participant files
    for p_file in os.listdir(p_dir):
        if p_file.endswith("Events.txt"):
            pp_data_filtered, participant = process_participant_file(os.path.join(p_dir, p_file))

            # Filter fixations, saccades, blinks and user events
            fixations.append(filter_event(pp_data_filtered, participant, "Fixations"))
            saccades.append(filter_event(pp_data_filtered, participant, "Saccades"))
            blinks.append(filter_event(pp_data_filtered, participant, "Blinks"))
            userevents.append(filter_event(pp_data_filtered, participant, "Userevents"))

    # Concatenate dataframes
    fixations_df = pd.concat(fixations, ignore_index=True)
    saccades_df = pd.concat(saccades, ignore_index=True)
    blinks_df = pd.concat(blinks, ignore_index=True)
    userevents_df = pd.concat(userevents, ignore_index=True)

    # Reject participants and clean dataframes
    fixations_df = fixations_df[~fixations_df["Participant"].isin(rejected_participants)]
    fixations_df.rename(
        columns={"v1": "Event_Type", "v1.5": "Eye", "v2": "Trial", "v3": "Number", "v4": "Start", "v5": "End",
                 "v6": "Duration", "v7": "Location_X", "v8": "Location_Y", "v9": "Dispersion_X", "v10": "Dispersion_Y",
                 "v11": "Plane", "v12": "Avg_Pupil_Size_X", "v13": "Avg_Pupil_Size_Y"},
        inplace=True)
    fixations_df.drop(
        columns=['v14', 'v15', 'v16', 'v17'],
        inplace=True)
    fixations_df['Start'] = fixations_df['Start'].astype(int)
    fixations_df['End'] = fixations_df['End'].astype(int)

    saccades_df = saccades_df[~saccades_df["Participant"].isin(rejected_participants)]
    saccades_df.rename(
        columns={"v1": "Event_Type", "v1.5": "Eye", "v2": "Trial", "v3": "Number", "v4": "Start", "v5": "End",
                 "v6": "Duration", "v7": "Start_Loc_X", "v8": "Start_Loc_Y", "v9": "End_Loc_X", "v10": "End_Loc_Y",
                 "v11": "Amplitude", "v12": "Peak_Speed", "v13": "Avg_Pupil_Size_Y", "v14": "Average_Speed",
                 "v15": "Peak_Accel", "v16": "Peak_Decel", "v17": "Average_Accel"},
        inplace=True)
    saccades_df['Start'] = saccades_df['Start'].astype(int)
    saccades_df['End'] = saccades_df['End'].astype(int)

    blinks_df = blinks_df[~blinks_df["Participant"].isin(rejected_participants)]
    blinks_df.rename(
        columns={"v1": "Event_Type", "v1.5": "Eye", "v2": "Trial", "v3": "Number", "v4": "Start", "v5": "End",
                 "v6": "Duration"},
        inplace=True)
    blinks_df.drop(
        columns=['v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17'],
        inplace=True)
    blinks_df['Start'] = blinks_df['Start'].astype(int)
    blinks_df['End'] = blinks_df['End'].astype(int)

    userevents_df = userevents_df[~userevents_df["Participant"].isin(rejected_participants)]
    userevents_df.rename(
        columns={"v1": "Event_Type", "v2": "Trial", "v3": "Number", "v4": "Start", "v5": "Message"},
        inplace=True)
    userevents_df.drop(
        columns=['v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17'],
        inplace=True)
    userevents_df['Start'] = userevents_df['Start'].astype(int)

    # Export dataframe to csv
    fixations_df.to_csv("training/CSVs/input files/fixations.csv")
    saccades_df.to_csv("training/CSVs/input files/saccades.csv")
    blinks_df.to_csv("training/CSVs/input files/blinks.csv")
    userevents_df.to_csv("training/CSVs/input files/userevents.csv")


def process_scores_file(r_path: str = None) -> None:
    """
    Process the scores file containing thought type data.

    Args:
    r_path (str): Path to the scores file.

    Returns:
    pd.DataFrame: Processed DataFrame containing thought type scores.
    """
    if r_path is None:
        r_path = scores_path
    scores_file = pyreadr.read_r(r_path)

    scores_df = scores_file["XML_data_cleaned"]
    # display(scores_df)

    # scores = scores_df["Thought_Type"]
    # scores_df[["Thought_Type"]] = OneHotEncoder().fit_transform(scores.values.reshape(-1, 1)).toarray()

    # y = OneHotEncoder().fit_transform(y_class.values.reshape(-1, 1)).toarray()

    encoded_scores = OneHotEncoder().fit_transform(scores_df["Thought_Type"].values.reshape(-1, 1)).toarray()
    encoded_columns = pd.DataFrame(encoded_scores,
                                   columns=[f"MW-score-{i + 1}" for i in range(encoded_scores.shape[1])])
    scores_df = pd.concat([scores_df, encoded_columns], axis=1)
    scores_df.drop(columns=["Thought_Type"], inplace=True)
    display(scores_df.loc[:, ~scores_df.columns.isin(['Textual_Trigger', 'Personal_Connection', 'Absorption'])])

    (scores_df.loc[:,
     ~scores_df.columns.isin(['Thought_Description', 'Textual_Trigger', 'Personal_Connection', 'Absorption'])]
     .to_csv("training/CSVs/input files/scores.csv"))


def process_timings_file(file_path) -> pd.DataFrame:
    """
    Process a timing file containing event timestamps.

    Args:
    file_path (str): Path to the timing file.

    Returns:
    pd.DataFrame: Processed DataFrame containing event timings.
    """
    # Read the CSV file into a DataFrame
    timings = pd.read_csv(file_path, header=None, sep='\t', usecols=[1, 2, 4])

    # Rename the columns
    timings.columns = ["Participant", "Timing", "Message"]

    # Convert Participant to lowercase
    timings["Participant"] = timings["Participant"].str.lower()

    # Convert Timing into microseconds
    timings["Hours"] = timings["Timing"].str.slice(0, 2).astype(int)
    timings["Minutes"] = timings["Timing"].str.slice(3, 4).astype(int)
    timings["Seconds"] = timings["Timing"].str.slice(6, 7).astype(int)
    timings["Milliseconds"] = timings["Timing"].str.slice(9, 11).astype(int)
    timings["Start"] = (timings["Hours"] * 3600000000) + (timings["Minutes"] * 60000000) + \
                       (timings["Seconds"] * 1000000) + (timings["Milliseconds"] * 1000)

    # Define dictionaries to map messages to probe and question numbers
    probe_mapping = {
        "Question76": 1, "Question11": 1, "Question31": 1, "Question77": 1, "Question78": 1,
        "Question79": 2, "Question12": 2, "Question33": 2, "Question80": 2, "Question81": 2,
        "Question82": 3, "Question34": 3, "Question35": 3, "Question83": 3, "Question84": 3,
        "Question85": 4, "Question36": 4, "Question37": 4, "Question86": 4, "Question87": 4,
        "Question88": 5, "Question38": 5, "Question39": 5, "Question89": 5, "Question90": 5,
        "Question91": 6, "Question40": 6, "Question41": 6, "Question92": 6, "Question93": 6,
        "Question94": 7, "Question42": 7, "Question43": 7, "Question95": 7, "Question96": 7,
        "Question97": 8, "Question44": 8, "Question45": 8, "Question98": 8, "Question100": 8,
        "Question101": 9, "Question46": 9, "Question47": 9, "Question102": 9, "Question103": 9,
        "Question104": 10, "Question48": 10, "Question49": 10, "Question105": 10, "Question106": 10,
        "Question107": 11, "Question50": 11, "Question51": 11, "Question108": 11, "Question109": 11,
        "Question110": 12, "Question52": 12, "Question53": 12, "Question111": 12, "Question112": 12,
        "Question113": 13, "Question54": 13, "Question55": 13, "Question114": 13, "Question115": 13,
        "Question116": 14, "Question56": 14, "Question57": 14, "Question117": 14, "Question118": 14,
        "Question120": 15, "Question58": 15, "Question59": 15, "Question121": 15, "Question122": 15,
        "Question123": 16, "Question60": 16, "Question61": 16, "Question124": 16, "Question125": 16,
        "Question126": 17, "Question62": 17, "Question63": 17, "Question127": 17, "Question128": 17,
        "Question129": 18, "Question64": 18, "Question65": 18, "Question130": 18, "Question131": 18,
        "Question132": 19, "Question66": 19, "Question67": 19
    }
    question_mapping = {
        "Question76": 1, "Question79": 1, "Question82": 1, "Question85": 1, "Question88": 1,
        "Question91": 1, "Question94": 1, "Question97": 1, "Question101": 1, "Question104": 1,
        "Question107": 1, "Question110": 1, "Question113": 1, "Question116": 1, "Question120": 1,
        "Question123": 1, "Question126": 1, "Question129": 1, "Question132": 1,
        "Question11": 2, "Question12": 2, "Question34": 2, "Question36": 2, "Question38": 2,
        "Question40": 2, "Question42": 2, "Question44": 2, "Question46": 2, "Question48": 2,
        "Question50": 2, "Question52": 2, "Question54": 2, "Question56": 2, "Question58": 2,
        "Question60": 2, "Question62": 2, "Question64": 2, "Question66": 2,
        "Question31": 3, "Question33": 3, "Question35": 3, "Question37": 3, "Question39": 3,
        "Question41": 3, "Question43": 3, "Question45": 3, "Question47": 3, "Question49": 3,
        "Question51": 3, "Question53": 3, "Question55": 3, "Question57": 3, "Question59": 3,
        "Question61": 3, "Question63": 3, "Question65": 3, "Question67": 3,
        "Question77": 4, "Question80": 4, "Question83": 4, "Question86": 4, "Question89": 4,
        "Question92": 4, "Question95": 4, "Question98": 4, "Question102": 4, "Question105": 4,
        "Question108": 4, "Question111": 4, "Question114": 4, "Question117": 4, "Question121": 4,
        "Question124": 4, "Question127": 4, "Question130": 4, "Question133": 4,
        "Question78": 5, "Question81": 5, "Question84": 5, "Question87": 5, "Question90": 5,
        "Question93": 5, "Question96": 5, "Question100": 5, "Question103": 5, "Question106": 5,
        "Question109": 5, "Question112": 5, "Question115": 5, "Question118": 5, "Question122": 5,
        "Question125": 5, "Question128": 5, "Question131": 5, "Question134": 5
    }

    # Add Probe and Question variables
    timings['Probe'] = timings["Message"].map(probe_mapping)
    timings['Question'] = timings["Message"].map(question_mapping)

    # Filter out rows with empty Message
    timings = timings.dropna(subset=['Message'])

    # timings = timings[timings["Message"].notnull()]

    return timings


def process_train_timings(t_dir=None) -> None:
    """
    Process all timing files in the given directory.

    Args:
    t_dir (str): Directory containing timing files.

    Returns:
    pd.DataFrame: Concatenated DataFrame containing all timings.
    """
    if t_dir is None:
        t_dir = timing_data_dir

    timings = []

    # Process timing files
    for t_file in os.listdir(t_dir):
        if re.match(r"[pP][0-9]+\.txt$", t_file):
            timings.append(process_timings_file(os.path.join(t_dir, t_file)))

    # Concatenate timings dataframes
    timings_df = pd.concat(timings, ignore_index=True)

    timings_df.to_csv("training/CSVs/input files/timings.csv")


def preprocess_data(filepath: str, verbose: bool = True, to_file: str = None) -> pd.DataFrame | None:
    if verbose:
        print(f"Loading data from {filepath}...")

    data = pd.read_csv(filepath)

    ppt_data = data.drop(columns=['WinWidth', 'WinHeight', 'x', 'y'])

    if verbose:
        print(f"Data loaded.")

    # Participants
    unique_participants = ppt_data['Participant'].unique()
    if verbose:
        print(f"Number of unique participants: {len(unique_participants)}")

    # Create a mapping for renaming participants
    participant_mapping = {p: str(i + 1) for i, p in enumerate(unique_participants)}

    # Rename participants using the mapping
    ppt_data['Participant'] = ppt_data['Participant'].replace(participant_mapping)

    if verbose:
        print("Participants renamed.")

    # Iterate through unique participants
    time_column = []

    for p in unique_participants:
        if verbose:
            print(f"Calculating total time for participant {participant_mapping[p]}...")

        # Filter rows for the current participant
        participant_data = ppt_data[ppt_data['Participant'] == participant_mapping[p]]

        # Cumulative time over all paragraphs
        prev_time = 0

        for para in ppt_data['Paragraph'].unique():
            participant_data.loc[participant_data['Paragraph'] == para, 'time'] += prev_time

            prev_time = participant_data.loc[participant_data['Paragraph'] == para, 'time'].iloc[-1]

        # Add the total time to the new time column
        time_column += list(participant_data['time'])

    if verbose:
        print("Cumulative time calculated.")

    data['Participant'] = ppt_data['Participant']
    data['Paragraph'] = ppt_data['Paragraph']
    data['time'] = list(map(lambda x: x*1000, time_column))

    if verbose:
        print("Processing complete.")

    if to_file:
        data.to_csv(to_file, index=False)
        if verbose:
            print("Saving complete.")
    else:
        return data
