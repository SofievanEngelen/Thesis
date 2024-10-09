import time
from IPython.core.display_functions import display
from pandas import DataFrame
from saccades import detect_fixations


def start_sliding_window(participant: int, data: DataFrame, window_size: int):
    start_time = 0

    data['Participant'] = data['Participant'].astype(int)
    p_data = data[data['Participant'] == participant]

    if p_data.empty:
        raise Exception(f'No participant {participant} found')

    while True:
        end_time = start_time + window_size

        window_data = p_data[(p_data['time'] > start_time) & (p_data['time'] < end_time)]
        window_data.loc[:, 'trial'] = start_time / 1000

        fixation_df = detect_fixations(window_data)
        # fixation_df.to_csv(f'fixations_p{participant}')

        start_time += 1000

        if start_time > (int(p_data['time'].max()) - window_size + 1000):
            break
