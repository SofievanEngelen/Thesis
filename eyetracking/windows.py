import json
import os
import time
import random


# Function to read eyetracking data from JSON file
def read_eyetracking_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data


# Function to implement moving windows
def moving_windows(eyetracking_data, window_size):
    windows = []
    for i in range(len(eyetracking_data) - window_size + 1):
        window = eyetracking_data[i:i + window_size]
        windows.append(window)
    return windows


# Path to the JSON file
json_file_path = "eyetracking_data.json"

# Define the window size
window_size = 20  # Adjust according to your requirement

# Initialize an empty list to hold the moving windows
moving_windows_list = []

last_modified_time = float(0)

# while True:
#     # Check if the JSON file has been modified
#     if os.path.exists(json_file_path):
#         modified_time = os.path.getmtime(json_file_path)
#         if modified_time > last_modified_time:
#             # Read eyetracking data from the JSON file
#             eyetracking_data = read_eyetracking_data(json_file_path)
#
#             # Implement moving windows
#             moving_windows_list = moving_windows(eyetracking_data, window_size)
#
#             # Update last modified time
#             last_modified_time = modified_time
#
#             # Optionally, visualize the moving windows
#             # Plotting code can be added here
#
#             print("Moving windows updated.")
#
#     # Wait for some time before checking for updates again
#     time.sleep(1)


# Function to generate mock eyetracking data with timestamps
def generate_mock_eyetracking_data(num_points):
    eyetracking_data = []
    current_time = int(time.time() * 1000)  # Current time in milliseconds
    for i in range(num_points):
        # Generate random eyetracking coordinates
        x = random.uniform(0, 1920)  # Assuming screen width of 1920 pixels
        y = random.uniform(0, 1080)  # Assuming screen height of 1080 pixels

        # Generate timestamp for each point
        timestamp = current_time + i * 10  # Incrementing by 10 milliseconds
        eyetracking_data.append({"x": x, "y": y, "timestamp": timestamp})
    return eyetracking_data


# Function to write mock eyetracking data to JSON file
def write_mock_eyetracking_data_to_json(eyetracking_data, json_file):
    with open(json_file, 'w') as file:
        json.dump(eyetracking_data, file)


def test_windows():
    # Path to the mock JSON file
    mock_json_file_path = "mock_eyetracking_data.json"

    # Generate and write mock eyetracking data to JSON file
    mock_eyetracking_data = generate_mock_eyetracking_data(1000)  # Generate 1000 points
    write_mock_eyetracking_data_to_json(mock_eyetracking_data, mock_json_file_path)

    # Define the window size
    window_size = 20  # Adjust according to your requirement

    # Initialize an empty list to hold the moving windows
    moving_windows_list = []

    # Simulate the continuous monitoring and processing of eyetracking data
    for i in range(len(mock_eyetracking_data)):
        # Read eyetracking data from the JSON file
        eyetracking_data = mock_eyetracking_data[:i + 1]

        # Implement moving windows
        if len(eyetracking_data) >= window_size:
            moving_windows_list = eyetracking_data[-window_size:]

        # Print the moving windows
        print(f"Moving Windows at Point {i + 1}:")
        for window in moving_windows_list:
            print(window)

        # Simulate a delay to mimic real-time processing
        time.sleep(0.5)  # Adjust as needed
