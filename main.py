import json
import random
import time

import pandas as pd
from IPython.core.display_functions import display

from eyetracking.event_detection import detect_fixations
from eyetracking.windows import test_windows
from training.ML import ml
from preprocessing import process_test_data


# Function to generate mock eyetracking data with timestamps
def generate_mock_eyetracking_data(file_path, num_points):
    eyetracking_data = []
    current_time = 0  # Current time in seconds

    for i in range(num_points):
        # Generate random eyetracking coordinates
        x = random.uniform(0, 1920)  # Assuming screen width of 1920 pixels
        y = random.uniform(0, 1080)  # Assuming screen height of 1080 pixels

        # Generate timestamp for each point
        timestamp = current_time + i  # Incrementing by 10 milliseconds
        eyetracking_data.append({"x": x, "y": y, "timestamp": timestamp})

    with open(file_path, 'w') as file:
        json.dump(eyetracking_data, file)

    return eyetracking_data


mock_file_path = "./mock_eyetracking_data.json"

train_data = pd.read_csv("training/CSVs/eyetracking_by_event.csv")
test_data_path = "eyetracking/test-data/gaze_data.csv"


def main():
    test_windows(test_data_path, 5)
    # print((198220010446-194350979093)/1000000/60)


if __name__ == "__main__":
    main()
