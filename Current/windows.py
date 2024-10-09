import time

from IPython.core.display_functions import display
from pandas import DataFrame
from saccades import call_detect_fixations


def sliding_window(participant: int, data: DataFrame, window_size: int):
    start_time = 0

    data['Participant'] = data['Participant'].astype(int)
    p_data = data[data['Participant'] == participant]

    if p_data.empty:
        raise Exception(f'No participant {participant} found')

    while True:
        end_time = start_time + window_size

        window_data = p_data[(p_data['cumulative_time'] > start_time) & (p_data['cumulative_time'] < end_time)]
        window_data.loc[:, 'trial'] = start_time / 1000

        # print(call_detect_fixations(window_data))
        window_data.to_csv('test.csv', index=False)
        display(window_data)

        time.sleep(3)

        start_time += 1000
        if start_time > (int(p_data['cumulative_time'].max()) - window_size + 1000):
            break
