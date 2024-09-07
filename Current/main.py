import json
import random
import time

import pandas as pd
from IPython.core.display_functions import display

# from eyetracking.event2 import detect_fixations
# from eyetracking.windows import windows
# from training.ML import train_model
# from preprocessing import process_test_data, process_scores_file
# from training.timings import agg_by_window
from windows import sliding_window


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
# test_data_path = "eye_tracking_test.csv"

event_samples = pd.DataFrame(
    [{'x': 1543.3067419981896, 'y': 378.49673204121257, 'trial': 1, 'time': 0.05},
     {'x': 1693.4370973211091, 'y': 255.85506473596905, 'trial': 1, 'time': 0.25},
     {'x': 1557.5812269343166, 'y': 929.7994129012582, 'trial': 1, 'time': 0.5},
     {'x': 731.3764514438205, 'y': 740.185914327995, 'trial': 1, 'time': 0.75},
     {'x': 1543.3067419981896, 'y': 378.49673204121257, 'trial': 1, 'time': 1},
     {'x': 1693.4370973211091, 'y': 255.85506473596905, 'trial': 1, 'time': 1.25},
     {'x': 1557.5812269343166, 'y': 929.7994129012582, 'trial': 1, 'time': 1.5},
     {'x': 731.3764514438205, 'y': 740.185914327995, 'trial': 1, 'time': 1.75},
     {'x': 1808.4911085479405, 'y': 641.0438741694295, 'trial': 1, 'time': 2},
     {'x': 1416.5773884783425, 'y': 931.0667972610821, 'trial': 1, 'time': 2.25},
     {'x': 1530.2698692177953, 'y': 441.86808389815917, 'trial': 1, 'time': 2.5},
     {'x': 257.3629102666456, 'y': 193.51804121939384, 'trial': 1, 'time': 2.75},
     {'x': 1589.2189308677841, 'y': 889.528490335354, 'trial': 1, 'time': 3},
     {'x': 190.12416148839696, 'y': 474.34859551553575, 'trial': 1, 'time': 3.25},
     {'x': 1270.800559520843, 'y': 943.6779305099109, 'trial': 1, 'time': 3.5},
     {'x': 1233.4476596960662, 'y': 305.10818700944344, 'trial': 1, 'time': 3.75},
     {'x': 1405.4391518482757, 'y': 724.5182594926556, 'trial': 1, 'time': 4},
     {'x': 403.33156848536674, 'y': 826.3063828782667, 'trial': 1, 'time': 4.25},
     {'x': 1771.0281376105122, 'y': 453.2032275919212, 'trial': 1, 'time': 4.5},
     {'x': 179.9783004139153, 'y': 219.52344440407552, 'trial': 1, 'time': 4.75},
     {'x': 1535.7707767543598, 'y': 1032.8589788496215, 'trial': 1, 'time': 5},
     {'x': 949.212902199926, 'y': 607.8865457839801, 'trial': 1, 'time': 5.25},
     {'x': 870.5354888882188, 'y': 521.9416777623795, 'trial': 1, 'time': 5.5},
     {'x': 602.560848841942, 'y': 427.29371693550934, 'trial': 1, 'time': 5.75},
     {'x': 1543.3067419981896, 'y': 378.49673204121257, 'trial': 1, 'time': 6.05},
     {'x': 1693.4370973211091, 'y': 255.85506473596905, 'trial': 1, 'time': 6.25},
     {'x': 1557.5812269343166, 'y': 929.7994129012582, 'trial': 1, 'time': 6.5},
     {'x': 731.3764514438205, 'y': 740.185914327995, 'trial': 1, 'time': 6.75},
     {'x': 1543.3067419981896, 'y': 378.49673204121257, 'trial': 1, 'time': 7},
     {'x': 1693.4370973211091, 'y': 255.85506473596905, 'trial': 1, 'time': 7.25},
     {'x': 1557.5812269343166, 'y': 929.7994129012582, 'trial': 1, 'time': 7.5},
     {'x': 731.3764514438205, 'y': 740.185914327995, 'trial': 1, 'time': 7.75},
     {'x': 1808.4911085479405, 'y': 641.0438741694295, 'trial': 1, 'time': 8},
     {'x': 1416.5773884783425, 'y': 931.0667972610821, 'trial': 1, 'time': 8.25},
     {'x': 1530.2698692177953, 'y': 441.86808389815917, 'trial': 1, 'time': 8.5},
     {'x': 257.3629102666456, 'y': 193.51804121939384, 'trial': 1, 'time': 8.75},
     {'x': 1589.2189308677841, 'y': 889.528490335354, 'trial': 1, 'time': 9},
     {'x': 190.12416148839696, 'y': 474.34859551553575, 'trial': 1, 'time': 9.25},
     {'x': 1270.800559520843, 'y': 943.6779305099109, 'trial': 1, 'time': 9.5},
     {'x': 1233.4476596960662, 'y': 305.10818700944344, 'trial': 1, 'time': 9.75},
     {'x': 1405.4391518482757, 'y': 724.5182594926556, 'trial': 1, 'time': 10},
     {'x': 403.33156848536674, 'y': 826.3063828782667, 'trial': 1, 'time': 10.25},
     {'x': 1771.0281376105122, 'y': 453.2032275919212, 'trial': 1, 'time': 10.5},
     {'x': 179.9783004139153, 'y': 219.52344440407552, 'trial': 1, 'time': 10.75},
     {'x': 1535.7707767543598, 'y': 1032.8589788496215, 'trial': 1, 'time': 11},
     {'x': 949.212902199926, 'y': 607.8865457839801, 'trial': 1, 'time': 11.25},
     {'x': 870.5354888882188, 'y': 521.9416777623795, 'trial': 1, 'time': 11.5},
     {'x': 1543.3067419981896, 'y': 378.49673204121257, 'trial': 1, 'time': 12.05},
     {'x': 1693.4370973211091, 'y': 255.85506473596905, 'trial': 1, 'time': 12.25},
     {'x': 1557.5812269343166, 'y': 929.7994129012582, 'trial': 1, 'time': 12.5},
     {'x': 731.3764514438205, 'y': 740.185914327995, 'trial': 1, 'time': 12.75},
     {'x': 1543.3067419981896, 'y': 378.49673204121257, 'trial': 1, 'time': 13},
     {'x': 1693.4370973211091, 'y': 255.85506473596905, 'trial': 1, 'time': 13.25},
     {'x': 1557.5812269343166, 'y': 929.7994129012582, 'trial': 1, 'time': 13.5},
     {'x': 731.3764514438205, 'y': 740.185914327995, 'trial': 1, 'time': 13.75},
     {'x': 1808.4911085479405, 'y': 641.0438741694295, 'trial': 1, 'time': 14},
     {'x': 1416.5773884783425, 'y': 931.0667972610821, 'trial': 1, 'time': 14.25},
     {'x': 1530.2698692177953, 'y': 441.86808389815917, 'trial': 1, 'time': 14.5},
     {'x': 257.3629102666456, 'y': 193.51804121939384, 'trial': 1, 'time': 14.75},
     {'x': 1589.2189308677841, 'y': 889.528490335354, 'trial': 1, 'time': 15},
     {'x': 190.12416148839696, 'y': 474.34859551553575, 'trial': 1, 'time': 15.25},
     {'x': 1270.800559520843, 'y': 943.6779305099109, 'trial': 1, 'time': 15.5},
     {'x': 1233.4476596960662, 'y': 305.10818700944344, 'trial': 1, 'time': 15.75},
     {'x': 1405.4391518482757, 'y': 724.5182594926556, 'trial': 1, 'time': 16},
     {'x': 403.33156848536674, 'y': 826.3063828782667, 'trial': 1, 'time': 16.25},
     {'x': 1771.0281376105122, 'y': 453.2032275919212, 'trial': 1, 'time': 16.5},
     {'x': 179.9783004139153, 'y': 219.52344440407552, 'trial': 1, 'time': 16.75},
     {'x': 1535.7707767543598, 'y': 1032.8589788496215, 'trial': 1, 'time': 17},
     {'x': 949.212902199926, 'y': 607.8865457839801, 'trial': 1, 'time': 17.25},
     {'x': 870.5354888882188, 'y': 521.9416777623795, 'trial': 1, 'time': 17.5}])


def main():
    # Event detection
    # fixations = detect_fixations(event_samples)
    # print(fixations.head())

    # Windows
    # process_test_data()
    # data = pd.read_csv(test_data_path)
    # data = data.loc[(data["Participant"] == "p1") & (data["Paragraph"] == 1)]
    # windows(data, 5)

    elements = pd.read_csv("./eye_tracking_test.csv")
    sliding_window('p1', data=elements, window_size=20)

    # Data preprocessing
    # process_scores_file()
    # agg_by_window()

    # ML
    # data = pd.read_csv('training/eyetracking_by_event.csv')
    # train_model(data)


if __name__ == "__main__":
    main()
