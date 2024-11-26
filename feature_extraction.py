import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from typing import Dict
from saccades import detect_events
from constants import AOI_DF_PATH, MIN_FIXATION_DURATION

# Global list to track dropped windows
dropped_windows = []


def compute_statistics(series: pd.Series, variable_name: str) -> Dict[str, float]:
    """
    Compute statistical measures for a pandas Series.

    :param series: A pandas Series containing numerical data.
    :param variable_name: The name of the variable for naming the computed statistics.
    :return: A dictionary of computed statistics.
    """
    if series.empty:
        return {f'{variable_name}_{stat}': np.nan for stat in
                ['mean', 'median', 'std', 'max', 'min', 'range', 'kurtosis', 'skewness']}

    return {
        f'{variable_name}_mean': series.mean(),
        f'{variable_name}_median': series.median(),
        f'{variable_name}_std': series.std(),
        f'{variable_name}_max': series.max(),
        f'{variable_name}_min': series.min(),
        f'{variable_name}_range': series.max() - series.min(),
        f'{variable_name}_kurtosis': kurtosis(series),
        f'{variable_name}_skewness': skew(np.array(series))
    }


def compute_features(gaze_data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes global and local features from gaze data by detecting fixations, blinks and saccades.

    :param gaze_data: A pandas DataFrame containing gaze data with required columns.
    :return: A pandas DataFrame containing computed features. If a window is dropped, returns an empty DataFrame.
    """
    # Detect fixations and saccades from gaze data
    fixation_df, saccade_df = detect_events(gaze_data)

    # Extract metadata
    participant = gaze_data['Participant'].iloc[0]
    paragraph = gaze_data['Paragraph'].iloc[0]
    trial = gaze_data['trial'].iloc[0]

    # Drop the window if any fixation lasts more than MIN_FIXATION_DURATION or no saccades are detected.
    if (not fixation_df.empty and fixation_df['duration'].max() > MIN_FIXATION_DURATION) or saccade_df.empty:
        dropped_windows.append({'participant': participant, 'paragraph': paragraph, 'trial': trial})
        return pd.DataFrame()

    # Add paragraph information to fixations
    fixation_df['paragraph'] = paragraph

    # Load AOI coordinates for the paragraph
    AOI_df = pd.read_csv(AOI_DF_PATH, delimiter=';', decimal=',')
    aoi_coords = AOI_df.loc[AOI_df['Paragraph'] == paragraph, ['x1', 'y1', 'x2', 'y2']].iloc[0]
    aoi_tl_x, aoi_tl_y, aoi_br_x, aoi_br_y = aoi_coords

    # Filter fixations within the AOI
    fixation_AOI_df = fixation_df[
        fixation_df['x'].between(aoi_tl_x, aoi_br_x) &
        fixation_df['y'].between(aoi_br_y, aoi_tl_y)
        ]

    # Compute counts
    num_fixations = len(fixation_df)
    num_saccades = len(saccade_df)

    # Compute fixation statistics
    fixation_duration_stats = compute_statistics(fixation_df['duration'], 'fixation_duration')
    fixation_dispersion_stats = compute_statistics(fixation_df['dispersion'], 'fixation_dispersion')

    # Compute saccade statistics
    saccade_angles = saccade_df['angle']
    saccade_duration_stats = compute_statistics(saccade_df['duration'], 'saccade_duration')
    saccade_amplitude_stats = compute_statistics(saccade_df['amplitude'], 'saccade_amplitude')
    saccade_angle_stats = compute_statistics(saccade_angles, 'saccade_angle')

    # Compute AOI fixation statistics
    fixation_AOI_duration_stats = compute_statistics(fixation_AOI_df['duration'], 'fixation_AOI_duration')
    fixation_AOI_dispersion_stats = compute_statistics(fixation_AOI_df['dispersion'], 'fixation_AOI_dispersion')

    # Compute feature dictionary
    feature_dict = {
        'participant': participant,
        'paragraph': paragraph,
        'trial': trial,

        'num_fixations': num_fixations,
        **fixation_duration_stats,
        **fixation_dispersion_stats,

        'num_saccades': num_saccades,
        **saccade_duration_stats,
        **saccade_amplitude_stats,
        **saccade_angle_stats,

        'horizontal_saccade_ratio': (saccade_angles.abs() < (np.pi / 4)).mean() if num_saccades > 0 else np.nan,
        'fixation_saccade_ratio': num_fixations / num_saccades if num_saccades > 0 else np.nan,

        'num_fixations_AOI': len(fixation_AOI_df),
        **fixation_AOI_duration_stats,
        **fixation_AOI_dispersion_stats
    }

    return pd.DataFrame([feature_dict])
