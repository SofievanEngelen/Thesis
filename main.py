import json
import random
import time

import pandas as pd
from IPython.core.display_functions import display

from eyetracking.event_detection import detect_fixations
from eyetracking.windows import windows
from training.ML import train_model
from preprocessing import process_test_data, process_scores_file
from training.timings import agg_by_window


# Function to generate mock eyetracking data with timestamps
def generate_mock_eyetracking_data(file_path, num_points):
    eyetracking_data = []
    current_time = 0  # Current time in seconds

    for i in range(num_points):
        # Generate random eyetracking coordinates
        x = random.uniform(0, 1920)  # Assuming screen width of 1920 pixels
        y = random.uniform(0, 1080)  # Assuming screen height of 1080 pixels

        # Generate timestamp for each point
        timestamp = current_time + 0.001 * i  # Incrementing by 10 milliseconds
        eyetracking_data.append({"x": x, "y": y, "timestamp": timestamp})

    data = pd.DataFrame(eyetracking_data)
    data.to_csv("mock_data.csv")

    return eyetracking_data


mock_file_path = "./mock_eyetracking_data.json"

# train_data = pd.read_csv("training/CSVs/eyetracking_by_event.csv")
test_data_path = "eye_tracking_test.csv"


def main():
    # print((667/2) - 0.6379310344827587 * 667)

    # Windows / event detection
    # process_test_data()
    # data = pd.read_csv(test_data_path)
    # data = data.loc[(data["Participant"] == "p1") & (data["Paragraph"] == 1)]
    # windows(data, 5)

    # Data preprocessing
    # process_scores_file()
    # agg_by_window()

    # ML
    data = pd.read_csv('training/eyetracking_by_event.csv')
    train_model(data)


if __name__ == "__main__":
    main()
