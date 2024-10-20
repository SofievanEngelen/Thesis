import time
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from saccades import detect_events

AOI_df = pd.read_csv('./original-data/aoi-boxes.csv', delimiter=';', decimal=',')


# Helper function that computes statistics for a measure
def compute_statistics(series, variable_name):
    """
        Computes various statistical measures for a given pandas Series.

        :param series: A pandas Series containing numerical data.
        :param variable_name: The name of the variable to compute statistics for, for dictionary naming purposes.
        :return: A dictionary with statistical measures.
    """
    return {
        f'{variable_name}_mean': series.mean(),
        f'{variable_name}_median': series.median(),
        f'{variable_name}_std': series.std(),
        f'{variable_name}_max': series.max(),
        f'{variable_name}_min': series.min(),
        f'{variable_name}_range': series.max() - series.min(),
        f'{variable_name}_kurtosis': kurtosis(series),
        f'{variable_name}_skewness': skew(series)
    }


def compute_features(gaze_data: pd.DataFrame) -> pd.DataFrame:
    """
        Computes global and local features from gaze data by detecting fixations, blinks, and saccades.

        :param gaze_data: A pandas DataFrame containing gaze data.
        :return: A pandas DataFrame with computed features.
    """
    # Detect events from gaze data
    fixation_df, saccade_df = detect_events(gaze_data)
    # print("Blinks: ", blink_df)

    participant = gaze_data['Participant'][0]
    paragraph = gaze_data['Paragraph'][0]

    fixation_df['participant'] = participant
    # blink_df['participant'] = participant
    saccade_df['participant'] = participant

    fixation_df['paragraph'] = paragraph
    # blink_df['paragraph'] = paragraph
    saccade_df['paragraph'] = paragraph

    # print("Fixations: ", fixation_df)
    # print("Saccades: ", saccade_df)

    # Filter fixations that fall within the AOI
    fixation_AOI_df = fixation_df[fixation_df.apply(lambda row: checkAOI(row['x_CC'], row['y_CC'], row['paragraph']),
                                                    axis=1)]

    # print("AOI fixations: ", fixation_AOI_df)

    # Define features to avoid repeated lookups
    num_fixations = len(fixation_df)
    num_saccades = len(saccade_df)
    saccade_angles = saccade_df['angle']
    # blink_duration = blink_df['duration'] if not blink_df.empty else pd.Series([0])

    # Fixation features
    fixation_duration_stats = compute_statistics(fixation_df['duration'], 'fixation_duration')
    fixation_dispersion_stats = compute_statistics(fixation_df['dispersion'], 'fixation_dispersion')

    # Saccade features
    saccade_duration_stats = compute_statistics(saccade_df['duration'], 'saccade_duration')
    saccade_amplitude_stats = compute_statistics(saccade_df['amplitude'], 'saccade_amplitude')
    saccade_angle_stats = compute_statistics(saccade_angles, 'saccade_angle')

    # Blink features
    # blink_duration_stats = compute_statistics(blink_duration, 'blink_duration')

    # Fixation within AOI features
    if not fixation_AOI_df.empty:
        fixation_AOI_duration_stats = compute_statistics(fixation_AOI_df['duration'], 'fixation_AOI_duration_')
        fixation_AOI_dispersion_stats = compute_statistics(fixation_AOI_df['dispersion'], 'fixation_AOI_dispersion')
    else:
        fixation_AOI_duration_stats = compute_statistics(pd.Series(0), 'fixation_AOI_duration_')
        fixation_AOI_dispersion_stats = compute_statistics(pd.Series(0), 'fixation_AOI_dispersion')

    feature_dict = {
        'num_fixations': num_fixations,
        **fixation_duration_stats,
        **fixation_dispersion_stats,

        'num_saccades': num_saccades,
        **saccade_duration_stats,
        **saccade_amplitude_stats,
        **saccade_angle_stats,
        'horizontal_saccade_ratio': (saccade_angles.abs() < (np.pi / 4)).sum() / num_saccades if num_saccades else np.nan,
        'fixation_saccade_ratio': num_fixations / num_saccades if num_saccades else np.nan,

        # 'num_blinks': len(blink_df),
        # **blink_duration_stats,
        'num_fixations_AOI': len(fixation_AOI_df),
        **fixation_AOI_duration_stats,
        **fixation_AOI_dispersion_stats
    }

    return pd.DataFrame.from_dict(feature_dict, orient='index').T


def checkAOI(x: float, y: float, paragraph: int) -> bool:
    """
    Checks if a given gaze point (x, y) is in the AOI (area of interest) of the given paragraph. Coordinates in
    Cartesian coordinate system.

    :param x: X coordinate of the gaze point.
    :param y: Y coordinate of the gaze point.
    :param paragraph: Paragraph in which the gaze point occurs.
    :return: Boolean indicating if the gaze point is within the AOI of the given paragraph.
    """
    aoi_tl_x = float(AOI_df.loc[AOI_df['Paragraph'] == paragraph, 'x1'].iloc[0])
    aoi_tl_y = float(AOI_df.loc[AOI_df['Paragraph'] == paragraph, 'y1'].iloc[0])
    aoi_br_x = float(AOI_df.loc[AOI_df['Paragraph'] == paragraph, 'x2'].iloc[0])
    aoi_br_y = float(AOI_df.loc[AOI_df['Paragraph'] == paragraph, 'y2'].iloc[0])

    return aoi_tl_x <= x <= aoi_br_x and aoi_br_y <= y <= aoi_tl_y

