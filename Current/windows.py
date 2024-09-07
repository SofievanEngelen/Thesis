import json
import os
import time
import random

import pandas as pd
from IPython.core.display_functions import display
from pandas import DataFrame


# OLD CODE
# def test_windows(data, window_size):
#     moving_windows_dict = {}
#
#     # Simulate continuous monitoring and processing of eyetracking data
#     for i in range(window_size - 1, len(data)):
#         # Initialize an empty list to hold the moving windows
#         windows_list = []
#
#         # Read eyetracking data from the JSON file
#         eyetracking_data = data[:i + 1]
#
#         # Implement moving windows
#         if len(eyetracking_data) >= window_size:
#             windows_list += eyetracking_data[-window_size:]
#
#         # Print the moving windows
#         # print(f"Moving Windows at Point {i + 1}:")
#         # print(moving_windows_list)
#         # for window in windows_list:
#         #     window['window'] = int(i+2-window_size)
#         # print(window)
#         if int(i + 2 - window_size) > 0:
#             moving_windows_dict[int(i + 2 - window_size)] = windows_list
#         print(moving_windows_dict)
#
#         # Simulate a delay to mimic real-time processing
#         time.sleep(1)
#
#
# # Function to implement moving windows
# def create_moving_window(starttime, data, window_size):
#     endtime = starttime + window_size
#
#     window_data = data.loc[(data["time"] >= starttime) & (data["time"] <= endtime)]
#
#     print(f"Start time = {starttime}: ")
#     display(window_data)
#
#     return window_data
#
#     # for i in range(1000):
#     #     while starttime < data[i]["timestamp"] <= endtime:
#     #         window.append({"x": data[i]["x"], "y": data[i]["y"]})
#     #     if data[i]["timestamp"] > endtime:
#     #         return window
#
#
# def windows(data, window_size):
#     starttime = 0
#     while True:
#         endtime = starttime + window_size
#         window_data = data.loc[(data["time"] >= starttime) & (data["time"] <= endtime)]
#         display(window_data)
#         time.sleep(1)
#         starttime += 1
#
#
# # Path to the JSON file
# json_file_path = "eyetracking_data.json"
#
# # Define the window size
# window_size = 20  # Adjust according to your requirement
#
# # Initialize an empty list to hold the moving windows
# moving_windows_list = []
#
# last_modified_time = float(0)

# while True:
# Check if the JSON file has been modified
# if os.path.exists(json_file_path):
#     modified_time = os.path.getmtime(json_file_path)
#     if modified_time > last_modified_time:
#         # Read eyetracking data from the JSON file
#         eyetracking_data = read_eyetracking_data(json_file_path)
#
#         # Implement moving windows
#         moving_windows_list = moving_windows(eyetracking_data, window_size)
#
#         # Update last modified time
#         last_modified_time = modified_time
#
#         # Optionally, visualize the moving windows
#         # Plotting code can be added here
#
#         print("Moving windows updated.")
#
# # Wait for some time before checking for updates again
# time.sleep(1)

def sliding_window(participant: str, data: DataFrame, window_size: int):
    start_time = 0
    p_data = data[data['Participant'] == participant]
    print(p_data)

    while True:
        end_time = start_time + window_size
        window_data = p_data[(p_data['time'] > start_time) & (p_data['time'] < end_time)]
        print(len(window_data), window_data, '\n')
        # time.sleep(2)
        start_time += 1
        if start_time > (int(p_data['time'].max()) - window_size + 1):
            break
