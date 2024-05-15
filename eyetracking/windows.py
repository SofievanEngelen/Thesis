import json
import os
import time
import random

import pandas as pd


def test_windows(file_path, window_size):
    moving_windows_dict = {}
    data = pd.read_csv(file_path)

    # Simulate continuous monitoring and processing of eyetracking data
    for i in range(window_size - 1, len(data)):
        # Initialize an empty list to hold the moving windows
        windows_list = []

        # Read eyetracking data from the JSON file
        eyetracking_data = data[:i + 1]

        # Implement moving windows
        if len(eyetracking_data) >= window_size:
            windows_list += eyetracking_data[-window_size:]

        # Print the moving windows
        # print(f"Moving Windows at Point {i + 1}:")
        # print(moving_windows_list)
        # for window in windows_list:
        #     window['window'] = int(i+2-window_size)
        # print(window)
        if int(i + 2 - window_size) > 0:
            moving_windows_dict[int(i + 2 - window_size)] = windows_list
        print(moving_windows_dict)

        # Simulate a delay to mimic real-time processing
        time.sleep(1)


# Function to implement moving windows
def create_moving_window(starttime, eyetracking_data, window_size):
    window = []

    endtime = starttime + window_size * 1000

    for i in range(1000):
        while starttime < eyetracking_data[i]["timestamp"] <= endtime:
            window.append({"x": eyetracking_data[i]["x"], "y": eyetracking_data[i]["y"]})
        if eyetracking_data[i]["timestamp"] > endtime:
            return window


def windows(file_path, window_size):
    data = pd.read_csv(file_path)




# Path to the JSON file
json_file_path = "eyetracking_data.json"

# Define the window size
window_size = 20  # Adjust according to your requirement

# Initialize an empty list to hold the moving windows
moving_windows_list = []

last_modified_time = float(0)

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
