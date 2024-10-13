import constants
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew


def compute_global_features(events_df: pd.DataFrame, saccade_df: pd.DataFrame) -> dict:
    fixation_df = events_df[events_df['event'] == 'fixation']
    print(fixation_df.head())
    blinks_df = events_df[events_df['event'] == 'blink']
    saccade_amplitude = np.sqrt(np.square(saccade_df['x'].diff()) + np.square(saccade_df['y'].diff()))
    saccade_angles = np.arctan2(saccade_df['y'].diff(), saccade_df['x'].diff())
    num_saccades = len(saccade_df)
    num_fixations = len(fixation_df)

    global_features = {'num_fixations': num_fixations,
                       'mean_fixation_duration': fixation_df['dur'].mean(),
                       'median_fixation_duration': fixation_df['dur'].median(),
                       'std_fixation_duration': fixation_df['dur'].std(),
                       'max_fixation_duration': fixation_df['dur'].max(),
                       'min_fixation_duration': fixation_df['dur'].min(),
                       'range_fixation_duration': fixation_df['dur'].max() - fixation_df['dur'].min(),
                       'kurtosis_fixation_duration': kurtosis(fixation_df['dur']),
                       'skewness_fixation_duration': skew(fixation_df['dur']),
                       'fixation_dispersion': np.sqrt(
                           np.square(fixation_df[fixation_df['event'] == 'fixation']['x'].diff()).sum() +
                           np.square(fixation_df[fixation_df['event'] == 'fixation']['y'].diff()).sum()),

                       'num_saccades': num_saccades,
                       # 'mean_saccade_duration': saccade_df['dur'].mean(),
                       # 'median_saccade_duration': saccade_df['dur'].median(),
                       # 'std_saccade_duration': saccade_df['dur'].std(),
                       # 'max_saccade_duration': saccade_df['dur'].max(),
                       # 'min_saccade_duration': saccade_df['dur'].min(),
                       # 'range_saccade_duration': saccade_df['dur'].max() - saccade_df['dur'].min(),
                       # 'kurtosis_saccade_duration': kurtosis(saccade_df['dur']),
                       # 'skewness_saccade_duration': skew(saccade_df['dur']),

                       'mean_saccade_amplitude': saccade_amplitude.mean(),
                       'median_saccade_amplitude': saccade_amplitude.median(),
                       'std_saccade_amplitude': saccade_amplitude.std(),
                       'max_saccade_amplitude': saccade_amplitude.max(),
                       'min_saccade_amplitude': saccade_amplitude.min(),
                       'range_saccade_amplitude': saccade_amplitude.max() - saccade_amplitude.min(),
                       'kurtosis_saccade_amplitude': kurtosis(saccade_amplitude),
                       'skewness_saccade_amplitude': skew(saccade_amplitude),

                       'mean_saccade_angle': saccade_angles.mean(),
                       'median_saccade_angle': saccade_angles.median(),
                       'std_saccade_angle': saccade_angles.std(),
                       'max_saccade_angle': saccade_angles.max(),
                       'min_saccade_angle': saccade_angles.min(),
                       'range_saccade_angle': saccade_angles.max() - saccade_angles.min(),
                       'kurtosis_saccade_angle': kurtosis(saccade_angles),
                       'skewness_saccade_angle': skew(saccade_angles),
                       'horizontal_saccade_ratio': sum(np.abs(saccade_angles) < (np.pi / 4)) / num_saccades,
                       'fixation_saccade_ratio': num_fixations / num_saccades if num_saccades != 0 else np.nan,

                       'num_blinks': len(blinks_df),
                       'mean_blink_duration': blinks_df['dur'].mean(),
                       'median_blink_duration': blinks_df['dur'].median(),
                       'std_blink_duration': blinks_df['dur'].std(),
                       'max_blink_duration': blinks_df['dur'].max(),
                       'min_blink_duration': blinks_df['dur'].min(),
                       'range_blink_duration': blinks_df['dur'].max() - blinks_df['dur'].min(),
                       'kurtosis_blink_duration': kurtosis(blinks_df['dur']),
                       'skewness_blink_duration': skew(blinks_df['dur'])}

    return global_features


def checkAOI(x: float, y: float, paragraph: int) -> bool:
    """
    Checks if a given gaze point (x, y) is in the AOI (area of interest) of the given paragraph. Coordinates in
    Carthesian coordinate system.

    :param x: X coordinate of the gaze point.
    :param y: Y coordinate of the gaze point.
    :param paragraph: Paragraph in which the gaze point occurs.
    :return: Boolean indicating if the gaze point is within the AOI of the given paragraph.
    """
    aoi_df = constants.AOI_DF
    print(aoi_df.head())

    aoi_tl_x = float(aoi_df.loc[aoi_df['Paragraph'] == paragraph, 'x1'].iloc[0])
    aoi_tl_y = float(aoi_df.loc[aoi_df['Paragraph'] == paragraph, 'y1'].iloc[0])
    aoi_br_x = float(aoi_df.loc[aoi_df['Paragraph'] == paragraph, 'x2'].iloc[0])
    aoi_br_y = float(aoi_df.loc[aoi_df['Paragraph'] == paragraph, 'y2'].iloc[0])

    return aoi_tl_x <= x <= aoi_br_x and aoi_br_y <= y <= aoi_tl_y
