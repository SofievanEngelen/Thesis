import json
import os
import time
import random


def test_windows(mock_data, window_size):

    # Simulate continuous monitoring and processing of eyetracking data
    for i in range(len(mock_data)):
        # Initialize an empty list to hold the moving windows
        moving_windows_list = []

        # Read eyetracking data from the JSON file
        eyetracking_data = mock_data[:i + 1]

        # Implement moving windows
        if len(eyetracking_data) >= window_size:
            moving_windows_list += eyetracking_data[-window_size:]

        # Print the moving windows
        print(f"Moving Windows at Point {i + 1}:")
        for window in moving_windows_list:
            print(window)

        # Simulate a delay to mimic real-time processing
        time.sleep(1)


# Function to read eyetracking data from JSON file
def read_eyetracking_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data


# Function to implement moving windows
def create_moving_window(starttime, eyetracking_data, window_size):

    window = []

    endtime = starttime + window_size*1000

    for i in range(1000):
        while starttime < eyetracking_data[i]["timestamp"] <= endtime:
            window.append({"x":eyetracking_data[i]["x"], "y":eyetracking_data[i]["y"]})
        if eyetracking_data[i]["timestamp"] > endtime:
            return window


# Path to the JSON file
json_file_path = "eyetracking_data.json"

# Define the window size
window_size = 20  # Adjust according to your requirement

# Initialize an empty list to hold the moving windows
moving_windows_list = []

last_modified_time = float(0)

while True:
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
