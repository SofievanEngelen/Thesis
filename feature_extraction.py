import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from saccades import detect_events
from constants import AOI_DF_PATH, MIN_FIXATION_DURATION

dropped_windows = []


# Helper function that computes statistics for a measure
def compute_statistics(series, variable_name):
    """Compute statistical measures for a pandas Series."""
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
        f'{variable_name}_skewness': skew(series)
    }


def compute_features(gaze_data: pd.DataFrame) -> pd.DataFrame:
    """
        Computes global and local features from gaze data by detecting fixations, blinks, and saccades.

        :param gaze_data: A pandas DataFrame containing gaze Data.
        :return: A pandas DataFrame with computed features, and a boolean indicating if the window was dropped and an
        empty dataframe was returned.
    """
    # Detect events from gaze Data
    fixation_df, saccade_df = detect_events(gaze_data)

    participant = gaze_data['Participant'].iloc[0]
    paragraph = gaze_data['Paragraph'].iloc[0]
    trial = gaze_data['trial'].iloc[0]

    # Drop the window if any fixation lasts more than 10 seconds
    if (not fixation_df.empty and fixation_df['duration'].max() > MIN_FIXATION_DURATION) or saccade_df.empty:
        return pd.DataFrame()

    fixation_df['paragraph'] = paragraph

    # Extract AOI coordinates for paragraph
    AOI_df = pd.read_csv(AOI_DF_PATH, delimiter=';', decimal=',')
    aoi_coords = AOI_df.loc[AOI_df['Paragraph'] == paragraph, ['x1', 'y1', 'x2', 'y2']].iloc[0]
    aoi_tl_x, aoi_tl_y, aoi_br_x, aoi_br_y = aoi_coords

    # Filter fixations that fall within the AOI
    fixation_AOI_df = fixation_df[
        (fixation_df['x'].between(aoi_tl_x, aoi_br_x)) &
        (fixation_df['y'].between(aoi_br_y, aoi_tl_y))
        ]

    num_fixations = len(fixation_df)
    num_saccades = len(saccade_df)

    # Compute fixation features
    fixation_duration_stats = compute_statistics(fixation_df['duration'], 'fixation_duration')
    fixation_dispersion_stats = compute_statistics(fixation_df['dispersion'], 'fixation_dispersion')

    # Compute saccade features
    saccade_angles = saccade_df['angle']
    saccade_duration_stats = compute_statistics(saccade_df['duration'], 'saccade_duration')
    saccade_amplitude_stats = compute_statistics(saccade_df['amplitude'], 'saccade_amplitude')
    saccade_angle_stats = compute_statistics(saccade_angles, 'saccade_angle')

    # Compute fixation AOI features
    fixation_AOI_duration_stats = compute_statistics(fixation_AOI_df['duration'], 'fixation_AOI_duration')
    fixation_AOI_dispersion_stats = compute_statistics(fixation_AOI_df['dispersion'], 'fixation_AOI_dispersion')

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
        'horizontal_saccade_ratio': (saccade_angles.abs() < (np.pi / 4)).mean() if num_saccades else np.nan,
        'fixation_saccade_ratio': num_fixations / num_saccades if num_saccades else np.nan,

        'num_fixations_AOI': len(fixation_AOI_df),
        **fixation_AOI_duration_stats,
        **fixation_AOI_dispersion_stats
    }

    return pd.DataFrame([feature_dict])
