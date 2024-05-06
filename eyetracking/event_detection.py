import numpy as np
from scipy.signal import medfilt


def label_blinks_artifacts(fixations):
    lsdx = np.log10(fixations['mad.x'])
    lsdy = np.log10(fixations['mad.y'])
    median_lsdx = np.median(lsdx)
    median_lsdy = np.median(lsdy)
    mad_lsdx = np.median(np.abs(lsdx - median_lsdx))
    mad_lsdy = np.median(np.abs(lsdy - median_lsdy))

    threshold_lsdx = median_lsdx - 4 * mad_lsdx
    threshold_lsdy = median_lsdy - 4 * mad_lsdy
    event = np.where(
        (lsdx < threshold_lsdx) & (lsdy < threshold_lsdy),
        'blink', 'fixation'
    )

    threshold_lsdx = median_lsdx + 4 * mad_lsdx
    threshold_lsdy = median_lsdy + 4 * mad_lsdy
    event = np.where(
        (lsdx > threshold_lsdx) & (lsdy > threshold_lsdy),
        'too dispersed', event
    )

    dur = 1 / fixations['dur']
    median_dur = np.median(dur)
    mad_dur = np.median(np.abs(dur - median_dur))

    threshold_dur = median_dur + mad_dur * 5
    event = np.where(
        (event != 'blink') & (dur > threshold_dur),
        'too short', event
    )

    return event


def aggregate_fixations(samples):
    saccade_events = np.sign(np.concatenate(([0], np.diff(samples['saccade']))))
    trial_numeric = samples['trial'].astype('category').cat.codes + 1
    trial_events = np.sign(np.concatenate(([0], np.diff(trial_numeric))))

    samples['fixation.id'] = np.cumsum((saccade_events == -1) | (trial_events == 1))
    samples['t2'] = samples['timestamp'].shift(-1)
    samples.loc[trial_events == 1, 't2'] = np.nan
    samples['t2'] = samples['t2'].fillna(method='ffill')

    samples = samples[~samples['saccade']]

    fixations = samples.groupby('fixation.id').agg(
        trial=('trial', 'first'),
        start=('timestamp', 'min'),
        end=('t2', lambda x: x.max(skipna=True)),
        x=('x', 'median'),
        y=('y', 'median'),
        mad_x=('x', 'mad'),
        mad_y=('y', 'mad'),
        peak_vx=('vx', lambda x: x[np.argmax(np.abs(x))]),
        peak_vy=('vy', lambda x: x[np.argmax(np.abs(x))])
    ).reset_index()

    fixations['dur'] = fixations['end'] - fixations['start']
    return fixations


def detect_saccades(samples, lam, smooth_saccades=True):
    vx = medfilt(samples['x'], kernel_size=3)
    vy = medfilt(samples['y'], kernel_size=3)

    msdx = np.sqrt(np.median(vx ** 2) - np.median(vx) ** 2)
    msdy = np.sqrt(np.median(vy ** 2) - np.median(vy) ** 2)

    radiusx = msdx * lam
    radiusy = msdy * lam

    sacc = ((vx / radiusx) ** 2 + (vy / radiusy) ** 2) > 1
    if smooth_saccades:
        sacc = medfilt(sacc, kernel_size=3)
        sacc = np.round(sacc).astype(bool)

    samples['saccade'] = sacc
    samples['vx'] = vx
    samples['vy'] = vy

    return samples


def detect_fixations(samples, lam=6, smooth_coordinates=False, smooth_saccades=True):
    if {'x', 'y', 'trial', 'timestamp'}.issubset(samples.columns):
        raise ValueError("Input data frame needs columns 'x', 'y', 'trial', and 'timestamp'.")

    if not all(samples.groupby('trial')['timestamp'].apply(lambda x: all(x.diff().dropna() > 0))):
        raise ValueError("Samples need to be in chronological order within trial.")

    samples = samples[['x', 'y', 'trial', 'timestamp']]

    if smooth_coordinates:
        x = samples['x'].iloc[[0, -1]]
        y = samples['y'].iloc[[0, -1]]
        samples['x'] = medfilt(samples['x'], kernel_size=3)
        samples['y'] = medfilt(samples['y'], kernel_size=3)
        samples.loc[[0, -1], ['x', 'y']] = [x, y]

    samples = detect_saccades(samples, lam, smooth_saccades)

    if not samples['saccade'].any():
        raise ValueError("No saccades were detected. Something went wrong.")

    fixations = aggregate_fixations(samples)

    fixations['event'] = label_blinks_artifacts(fixations)

    return fixations

# Example usage:
# samples = pd.read_csv('path/to/samples.csv')
# fixations = detect_fixations(samples)
# print(fixations.head())
