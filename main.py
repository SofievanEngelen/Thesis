import json
import random
import time

from eyetracking.windows import test_windows


# Function to generate mock eyetracking data with timestamps
def generate_mock_eyetracking_data(file_path, num_points):
    eyetracking_data = []
    current_time = int(time.time() * 1000)  # Current time in milliseconds

    for i in range(num_points):
        # Generate random eyetracking coordinates
        x = random.uniform(0, 1920)  # Assuming screen width of 1920 pixels
        y = random.uniform(0, 1080)  # Assuming screen height of 1080 pixels

        # Generate timestamp for each point
        timestamp = current_time + i * 10  # Incrementing by 10 milliseconds
        eyetracking_data.append({"x": x, "y": y, "timestamp": timestamp})

    with open(file_path, 'w') as file:
        json.dump(eyetracking_data, file)

    return eyetracking_data


mock_file_path = "./mock_eyetracking_data.json"


def main():
    mock_data = generate_mock_eyetracking_data(mock_file_path, 1000)  # Generate 1000 points
    test_windows(mock_data, 20)


if __name__ == "__main__":
    main()
