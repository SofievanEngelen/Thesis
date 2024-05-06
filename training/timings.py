import os
import pandas as pd
import warnings
import numpy as np
from IPython.core.display_functions import display

from preprocess_files import process_timings, process_participants, process_scores_file

warnings.simplefilter(action='ignore', category=Warning)


def calculate_mean_columns(df, columns):
    for p in df["Probe"].unique():
        left_eye_rows = df[(df["Probe"] == p) & (df['Eye'] == 'L')]
        right_eye_rows = df[(df["Probe"] == p) & (df['Eye'] == 'R')]
        for column in columns:
            left_eye_values = left_eye_rows[column].astype(float)
            right_eye_values = right_eye_rows[column].astype(float)
            mean_values = np.mean([left_eye_values, right_eye_values], axis=0)
            df.loc[(df["Probe"] == p) & (df['Eye'] == 'L'), column] = mean_values
    df = df[df['Eye'] != 'R']
    df = df.drop(columns=['Eye'], axis=1)
    return df.reset_index(drop=True)


def add_timings(fixations: pd.DataFrame, saccades: pd.DataFrame, blinks: pd.DataFrame, userevents: pd.DataFrame,
                timings: pd.DataFrame, window: int):
    # Define window in microseconds
    win = 1000000 * window

    # Define columns for aggregation
    fixation_columns = ['Participant', 'Eye', 'Probe', 'No_Fixations', 'Mean_Fixation_Duration',
                        'Median_Fixation_Duration', 'SD_Fixation_Duration', 'Skew_Fixation_Duration',
                        'Kurtosis_Fixation_Duration', 'Min_Fixation_Duration', 'Max_Fixation_Duration',
                        'Mean_Location_X', 'Median_Location_X', 'SD_Location_X', 'Skew_Location_X',
                        'Kurtosis_Location_X', 'Min_Location_X', 'Max_Location_X', 'Mean_Location_Y',
                        'Median_Location_Y', 'SD_Location_Y', 'Skew_Location_Y', 'Kurtosis_Location_Y',
                        'Min_Location_Y', 'Max_Location_Y', 'Mean_Dispersion_X', 'Median_Dispersion_X',
                        'SD_Dispersion_X', 'Skew_Dispersion_X', 'Kurtosis_Dispersion_X', 'Min_Dispersion_X',
                        'Max_Dispersion_X', 'Mean_Dispersion_Y', 'Median_Dispersion_Y', 'SD_Dispersion_Y',
                        'Skew_Dispersion_Y', 'Kurtosis_Dispersion_Y', 'Min_Dispersion_Y', 'Max_Dispersion_Y']

    saccade_columns = ['Participant', 'Eye', 'Probe', 'No_Saccades', 'Mean_Saccade_Duration',
                       'Median_Saccade_Duration', 'SD_Saccade_Duration', 'Skew_Saccade_Duration',
                       'Kurtosis_Saccade_Duration', 'Min_Saccade_Duration', 'Max_Saccade_Duration',
                       'Mean_Amplitude',
                       'Median_Amplitude', 'SD_Amplitude', 'Skew_Amplitude', 'Kurtosis_Amplitude', 'Min_Amplitude',
                       'Max_Amplitude', 'Mean_Peak_Speed', 'Median_Peak_Speed', 'SD_Peak_Speed', 'Skew_Peak_Speed',
                       'Kurtosis_Peak_Speed', 'Min_Peak_Speed', 'Max_Peak_Speed', 'Mean_Average_Speed',
                       'Median_Average_Speed', 'SD_Average_Speed', 'Skew_Average_Speed', 'Kurtosis_Average_Speed',
                       'Min_Average_Speed', 'Max_Average_Speed', 'Mean_Peak_Accel', 'Median_Peak_Accel',
                       'SD_Peak_Accel', 'Skew_Peak_Accel', 'Kurtosis_Peak_Accel', 'Min_Peak_Accel',
                       'Max_Peak_Accel',
                       'Mean_Peak_Decel', 'Median_Peak_Decel', 'SD_Peak_Decel', 'Skew_Peak_Decel',
                       'Kurtosis_Peak_Decel', 'Min_Peak_Decel', 'Max_Peak_Decel', 'Mean_Average_Accel',
                       'Median_Average_Accel', 'SD_Average_Accel', 'Skew_Average_Accel', 'Kurtosis_Average_Accel',
                       'Min_Average_Accel', 'Max_Average_Accel']

    blink_columns = ['Participant', 'Eye', 'Probe', 'No_Blinks', 'Mean_Blink_Duration', 'Median_Blink_Duration',
                     'SD_Blink_Duration', 'Skew_Blink_Duration', 'Kurtosis_Blink_Duration', 'Min_Blink_Duration',
                     'Max_Blink_Duration']

    # Create dataframes to store results

    fixations_by_event = []
    saccades_by_event = []
    blinks_by_event = []

    # fixations_by_event = pd.DataFrame(columns=fixation_columns)
    # saccades_by_event = pd.DataFrame(columns=saccade_columns)
    # blinks_by_event = pd.DataFrame(columns=blink_columns)

    # display(userevents)
    # Loop through participants and events
    for p in fixations['Participant'].unique():
        print(f"Adding timings to participant file: {p}")
        p_events_all = timings[timings['Participant'] == p]
        p_events = timings[(timings['Participant'] == p) & (timings['Question'] == 1)]

        for e in p_events['Probe'].unique():
            timing = int(p_events.loc[p_events['Probe'] == e, 'Start'].iloc[0])
            timing -= p_events_all.loc[p_events_all['Message'] == "RichText.rtf", 'Start'].iloc[0]
            timing += userevents.loc[(userevents['Participant'] == p) & (userevents['Message'] == "RichText.rtf"),
            'Start'].iloc[0]

            # Fixations
            p_fix = fixations[(fixations['Participant'] == p) & (fixations['End'] <= timing) &
                              (fixations['Start'] >= (timing - 1000 * win))]
            p_fix['Probe'] = e
            p_fix['No_Fixations'] = np.nan

            p_fix_grouped = p_fix.groupby(['Participant', 'Eye', 'Probe']).agg(
                No_Fixations=('Duration', 'count'),
                Mean_Fixation_Duration=('Duration', 'mean'),
                Median_Fixation_Duration=('Duration', 'median'),
                SD_Fixation_Duration=('Duration', 'std'),
                Skew_Fixation_Duration=('Duration', 'skew'),
                Kurtosis_Fixation_Duration=('Duration', lambda x: x.kurt()),
                Min_Fixation_Duration=('Duration', 'min'),
                Max_Fixation_Duration=('Duration', 'max'),
                Mean_Location_X=('Location_X', 'mean'),
                Median_Location_X=('Location_X', 'median'),
                SD_Location_X=('Location_X', 'std'),
                Skew_Location_X=('Location_X', 'skew'),
                Kurtosis_Location_X=('Location_X', lambda x: x.kurt()),
                Min_Location_X=('Location_X', 'min'),
                Max_Location_X=('Location_X', 'max'),
                Mean_Location_Y=('Location_Y', 'mean'),
                Median_Location_Y=('Location_Y', 'median'),
                SD_Location_Y=('Location_Y', 'std'),
                Skew_Location_Y=('Location_Y', 'skew'),
                Kurtosis_Location_Y=('Location_Y', lambda x: x.kurt()),
                Min_Location_Y=('Location_Y', 'min'),
                Max_Location_Y=('Location_Y', 'max'),
                Mean_Dispersion_X=('Dispersion_X', 'mean'),
                Median_Dispersion_X=('Dispersion_X', 'median'),
                SD_Dispersion_X=('Dispersion_X', 'std'),
                Skew_Dispersion_X=('Dispersion_X', 'skew'),
                Kurtosis_Dispersion_X=('Dispersion_X', lambda x: x.kurt()),
                Min_Dispersion_X=('Dispersion_X', 'min'),
                Max_Dispersion_X=('Dispersion_X', 'max'),
                Mean_Dispersion_Y=('Dispersion_Y', 'mean'),
                Median_Dispersion_Y=('Dispersion_Y', 'median'),
                SD_Dispersion_Y=('Dispersion_Y', 'std'),
                Skew_Dispersion_Y=('Dispersion_Y', 'skew'),
                Kurtosis_Dispersion_Y=('Dispersion_Y', lambda x: x.kurt()),
                Min_Dispersion_Y=('Dispersion_Y', 'min'),
                Max_Dispersion_Y=('Dispersion_Y', 'max')
            )

            fixations_by_event.append(p_fix_grouped.reset_index())

            # Saccades
            p_sacc = saccades[(saccades['Participant'] == p) & (saccades['End'] <= timing) &
                              (saccades['Start'] >= (timing - 1000 * win))]
            p_sacc['Probe'] = e
            p_sacc['No_Saccades'] = np.nan

            p_sacc_grouped = p_sacc.groupby(['Participant', 'Eye', 'Probe']).agg(
                No_Saccades=('Duration', 'count'),
                Mean_Saccade_Duration=('Duration', 'mean'),
                Median_Saccade_Duration=('Duration', 'median'),
                SD_Saccade_Duration=('Duration', 'std'),
                Skew_Saccade_Duration=('Duration', 'skew'),
                Kurtosis_Saccade_Duration=('Duration', lambda x: x.kurt()),
                Min_Saccade_Duration=('Duration', 'min'),
                Max_Saccade_Duration=('Duration', 'max'),
                Mean_Amplitude=('Amplitude', 'mean'),
                Median_Amplitude=('Amplitude', 'median'),
                SD_Amplitude=('Amplitude', 'std'),
                Skew_Amplitude=('Amplitude', 'skew'),
                Kurtosis_Amplitude=('Amplitude', lambda x: x.kurt()),
                Min_Amplitude=('Amplitude', 'min'),
                Max_Amplitude=('Amplitude', 'max'),
                Mean_Peak_Speed=('Peak_Speed', 'mean'),
                Median_Peak_Speed=('Peak_Speed', 'median'),
                SD_Peak_Speed=('Peak_Speed', 'std'),
                Skew_Peak_Speed=('Peak_Speed', 'skew'),
                Kurtosis_Peak_Speed=('Peak_Speed', lambda x: x.kurt()),
                Min_Peak_Speed=('Peak_Speed', 'min'),
                Max_Peak_Speed=('Peak_Speed', 'max'),
                Mean_Average_Speed=('Average_Speed', 'mean'),
                Median_Average_Speed=('Average_Speed', 'median'),
                SD_Average_Speed=('Average_Speed', 'std'),
                Skew_Average_Speed=('Average_Speed', 'skew'),
                Kurtosis_Average_Speed=('Average_Speed', lambda x: x.kurt()),
                Min_Average_Speed=('Average_Speed', 'min'),
                Max_Average_Speed=('Average_Speed', 'max'),
                Mean_Peak_Accel=('Peak_Accel', 'mean'),
                Median_Peak_Accel=('Peak_Accel', 'median'),
                SD_Peak_Accel=('Peak_Accel', 'std'),
                Skew_Peak_Accel=('Peak_Accel', 'skew'),
                Kurtosis_Peak_Accel=('Peak_Accel', lambda x: x.kurt()),
                Min_Peak_Accel=('Peak_Accel', 'min'),
                Max_Peak_Accel=('Peak_Accel', 'max'),
                Mean_Peak_Decel=('Peak_Decel', 'mean'),
                Median_Peak_Decel=('Peak_Decel', 'median'),
                SD_Peak_Decel=('Peak_Decel', 'std'),
                Skew_Peak_Decel=('Peak_Decel', 'skew'),
                Kurtosis_Peak_Decel=('Peak_Decel', lambda x: x.kurt()),
                Min_Peak_Decel=('Peak_Decel', 'min'),
                Max_Peak_Decel=('Peak_Decel', 'max'),
                Mean_Average_Accel=('Average_Accel', 'mean'),
                Median_Average_Accel=('Average_Accel', 'median'),
                SD_Average_Accel=('Average_Accel', 'std'),
                Skew_Average_Accel=('Average_Accel', 'skew'),
                Kurtosis_Average_Accel=('Average_Accel', lambda x: x.kurt()),
                Min_Average_Accel=('Average_Accel', 'min'),
                Max_Average_Accel=('Average_Accel', 'max')
            )

            saccades_by_event.append(p_sacc_grouped.reset_index())

            # Blinks
            p_blink = blinks[(blinks['Participant'] == p) & (blinks['End'] <= timing) &
                             (blinks['Start'] >= (timing - 1000 * win))]
            p_blink['Probe'] = e
            p_blink['No_Blinks'] = np.nan

            p_blink_grouped = p_blink.groupby(['Participant', 'Eye', 'Probe']).agg(
                No_Blinks=('Duration', 'count'),
                Mean_Blink_Duration=('Duration', 'mean'),
                Median_Blink_Duration=('Duration', 'median'),
                SD_Blink_Duration=('Duration', 'std'),
                Skew_Blink_Duration=('Duration', 'skew'),
                Kurtosis_Blink_Duration=('Duration', lambda x: x.kurt()),
                Min_Blink_Duration=('Duration', 'min'),
                Max_Blink_Duration=('Duration', 'max')
            )

            blinks_by_event.append(p_blink_grouped.reset_index())

    fixations_by_event = pd.concat(fixations_by_event, ignore_index=True)
    fixations_by_event.to_csv('by_events/Fixations_by_event.csv', index=False)

    saccades_by_event = pd.concat(saccades_by_event, ignore_index=True)
    saccades_by_event.to_csv('by_events/Saccades_by_event.csv', index=False)

    blinks_by_event = pd.concat(blinks_by_event, ignore_index=True)
    blinks_by_event.to_csv('by_events/Blinks_by_event.csv', index=False)


def agg_by_window(window: int = 20):
    # Check if t
    if (os.path.isfile("CSVs/input files/fixations.csv")
            and os.path.isfile("CSVs/input files/saccades.csv")
            and os.path.isfile("CSVs/input files/blinks.csv")
            and os.path.isfile("CSVs/input files/userevents.csv")):

        print("Pre-loading event csv files...")

        fixations = pd.read_csv("CSVs/input files/fixations.csv")
        saccades = pd.read_csv("CSVs/input files/saccades.csv")
        blinks = pd.read_csv("CSVs/input files/blinks.csv")
        userevents = pd.read_csv("CSVs/input files/userevents.csv")
    else:
        print("Processing participants files...")

        process_participants()

        fixations = pd.read_csv("CSVs/input files/fixations.csv")
        saccades = pd.read_csv("CSVs/input files/saccades.csv")
        blinks = pd.read_csv("CSVs/input files/blinks.csv")
        userevents = pd.read_csv("CSVs/input files/userevents.csv")

    if os.path.isfile("CSVs/input files/timings.csv"):

        print("Pre-loading timings csv file...")

        timings = pd.read_csv("CSVs/input files/timings.csv")
    else:
        print("Processing timings file...")

        process_timings()

        timings = pd.read_csv("CSVs/input files/timings.csv")

    if os.path.isfile("CSVs/input files/scores.csv"):

        print("Pre-loading MW scores csv file...")

        scores = pd.read_csv("CSVs/input files/scores.csv")
    else:
        print("Processing scores file...")

        process_scores_file()

        scores = pd.read_csv("CSVs/input files/scores.csv")

    # Call function with preloaded CSV's
    if (os.path.isfile("CSVs/by_events/Fixations_by_event.csv")
            and os.path.isfile("CSVs/by_events/Saccades_by_event.csv")
            and os.path.isfile("CSVs/by_events/Blinks_by_event.csv")):

        print("Pre-loading timed_events files...")

        fixations_by_event = pd.read_csv("CSVs/by_events/Fixations_by_event.csv")
        saccades_by_event = pd.read_csv("CSVs/by_events/Saccades_by_event.csv")
        blinks_by_event = pd.read_csv("CSVs/by_events/Blinks_by_event.csv")
    else:
        print("Processing timed_events files...")

        add_timings(fixations, saccades, blinks, userevents,
                    timings, window)

        fixations_by_event = pd.read_csv("CSVs/by_events/Fixations_by_event.csv")
        saccades_by_event = pd.read_csv("CSVs/by_events/Saccades_by_event.csv")
        blinks_by_event = pd.read_csv("CSVs/by_events/Blinks_by_event.csv")

    mean_columns = list(set(fixations_by_event.columns) | set(blinks_by_event.columns) | set(saccades_by_event.columns))
    for x in ["Participant", "Probe", "Eye"]:
        mean_columns.remove(x)

    eyetracking_by_event = pd.merge(fixations_by_event, saccades_by_event, on=["Participant", "Probe", "Eye"])

    eyetracking_by_event = pd.merge(eyetracking_by_event, blinks_by_event, on=["Participant", "Probe", "Eye"])
    eyetracking_by_event = calculate_mean_columns(eyetracking_by_event, mean_columns)

    eyetracking_by_event = pd.merge(eyetracking_by_event, scores, on=["Participant", "Probe"], how="left")

    eyetracking_by_event.reset_index(inplace=True)

    # Save results to CSV
    eyetracking_by_event.to_csv('eyetracking_by_event.csv', index=False)
