import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from scipy.signal import medfilt


def mad(data):
    median = np.median(data)
    return np.median(np.abs(data - median))


def label_blinks_artifacts(fixations):
    # Blink and artifact detection based on dispersion:
    lsdx = np.log10(fixations['mad_x'])
    lsdy = np.log10(fixations['mad_y'])
    median_lsdx = np.median(lsdx)
    median_lsdy = np.median(lsdy)
    mad_lsdx = np.median(np.abs(lsdx - median_lsdx))
    mad_lsdy = np.median(np.abs(lsdy - median_lsdy))

    # Dispersion too low -> blink:
    threshold_lsdx = median_lsdx - 4 * mad_lsdx
    threshold_lsdy = median_lsdy - 4 * mad_lsdy
    event = np.where((~np.isnan(lsdx) & (lsdx < threshold_lsdx)) &
                     (~np.isnan(lsdy) & (lsdy < threshold_lsdy)),
                     "blink", "fixation")

    # Dispersion too high -> artifact:
    threshold_lsdx = median_lsdx + 4 * mad_lsdx
    threshold_lsdy = median_lsdy + 4 * mad_lsdy
    event = np.where((~np.isnan(lsdx) & (lsdx > threshold_lsdx)) &
                     (~np.isnan(lsdy) & (lsdy > threshold_lsdy)),
                     "too dispersed", event)

    # Artifact detection based on duration:
    dur = 1 / fixations['dur']
    median_dur = np.median(dur)
    mad_dur = np.median(np.abs(dur - median_dur))

    # Duration too short -> artifact:
    threshold_dur = median_dur + mad_dur * 5
    event = np.where((event != "blink") & (dur > threshold_dur), "too short", event)

    # Convert event to categorical variable
    event = pd.Categorical(event, categories=["fixation", "blink", "too dispersed", "too short"])

    return event


def aggregate_fixations(samples):
    # In saccade.events a 1 marks the start of a saccade and a -1 the
    # start of a fixation.
    saccade_events = np.sign(np.concatenate(([0], np.diff(samples['saccade']))))

    window_numeric = pd.factorize(samples['window'])[0]
    window_events = np.sign(np.concatenate(([0], np.diff(window_numeric))))

    # New fixations start either when a saccade ends or when a trial ends:
    samples['fixation.id'] = np.cumsum((saccade_events == -1) | (window_events == 1))
    samples['t2'] = samples['timestamp']
    samples['t2'] = np.where(window_events == 1, np.nan, samples['t2'])
    samples['t2'] = samples['t2'].shift(-1)
    samples['t2'].iloc[-1] = samples['timestamp'].iloc[-1]  # Set last t2 value to last time value

    # Discard samples that occurred during saccades:
    samples = samples[~samples['saccade']]

    fixations = samples.groupby('fixation.id').agg(
        window=('window', 'first'),
        start=('timestamp', 'min'),
        end=('t2', lambda x: x.max(skipna=True)),
        x=('x', 'median'),
        y=('y', 'median'),
        mad_x=('x', lambda x: mad(x)),
        mad_y=('y', lambda x: mad(x)),
        peak_vx=('vx', lambda x: x[np.argmax(np.abs(x))]),
        peak_vy=('vy', lambda x: x[np.argmax(np.abs(x))])
    ).reset_index()

    fixations['dur'] = fixations['end'] - fixations['start']

    return fixations


def detect_saccades(samples, lam, smooth_saccades=False):
    # Calculate horizontal and vertical velocities:
    vx = medfilt(samples['x'], kernel_size=3)
    vy = medfilt(samples['y'], kernel_size=3)

    # Fill in missing values
    vx[0] = vx[1]
    vy[0] = vy[1]
    vx[-1] = vx[-2]
    vy[-1] = vy[-2]

    msdx = np.sqrt(np.median(vx ** 2) - np.median(vx) ** 2)
    msdy = np.sqrt(np.median(vy ** 2) - np.median(vy) ** 2)

    radiusx = msdx * lam
    radiusy = msdy * lam

    sacc = ((vx / radiusx) ** 2 + (vy / radiusy) ** 2) > 1
    if smooth_saccades:
        sacc = np.convolve(sacc, np.ones(3) / 3, mode='same')  # Apply a moving average filter
        sacc = np.round(sacc).astype(bool)  # Convert to logical values

    samples['saccade'] = ~np.isnan(sacc) & sacc
    samples['vx'] = vx
    samples['vy'] = vy

    return samples


def detect_fixations(samples, lam=6, smooth_coordinates=True, smooth_saccades=True):
    if not {"x", "y", "window", "timestamp"}.issubset(samples.columns):
        raise ValueError("Input data frame needs columns 'x', 'y', 'window', and 'timestamp'.")

    if not all(samples.groupby('window')['timestamp'].apply(lambda x: all(x.diff().dropna() > 0))):
        raise ValueError("Samples need to be in chronological order within trial.")

    samples = samples[["x", "y", "window", "timestamp"]]

    # display(samples)

    if smooth_coordinates:
        x = samples['x'].iloc[[0, -1]]
        y = samples['y'].iloc[[0, -1]]
        samples['x'] = medfilt(samples['x'], kernel_size=3)
        samples['y'] = medfilt(samples['y'], kernel_size=3)
        # samples.loc[[0, -1], ["x", "y"]] = [x, y]
        print("test: ",samples.loc[[-1], "x"])
        # samples.loc[[0, -1], ["x", "y"]] = [x, y]
        # samples.loc[[0, -1], ["x", "y"]] = [x, y]
        samples['y'].iloc[[0, -1]] = y

    display(samples)

    samples = detect_saccades(samples, lam, smooth_saccades)

    display(samples)

    if not samples['saccade'].any():
        raise ValueError("No saccades were detected. Something went wrong.")

    fixations = aggregate_fixations(samples)

    fixations['event'] = label_blinks_artifacts(fixations)

    return fixations

# Example usage:
# samples = pd.DataFrame([{'x': 1543.3067419981896, 'y': 378.49673204121257, 'timestamp': 1715358250626},
#                         {'x': 1693.4370973211091, 'y': 255.85506473596905, 'timestamp': 1715358250636},
#                         {'x': 1557.5812269343166, 'y': 929.7994129012582, 'timestamp': 1715358250646},
#                         {'x': 731.3764514438205, 'y': 740.185914327995, 'timestamp': 1715358250656},
#                         {'x': 1808.4911085479405, 'y': 641.0438741694295, 'timestamp': 1715358250666},
#                         {'x': 1416.5773884783425, 'y': 931.0667972610821, 'timestamp': 1715358250676},
#                         {'x': 1530.2698692177953, 'y': 441.86808389815917, 'timestamp': 1715358250686},
#                         {'x': 257.3629102666456, 'y': 193.51804121939384, 'timestamp': 1715358250696},
#                         {'x': 1589.2189308677841, 'y': 889.528490335354, 'timestamp': 1715358250706},
#                         {'x': 190.12416148839696, 'y': 474.34859551553575, 'timestamp': 1715358250716},
#                         {'x': 1270.800559520843, 'y': 943.6779305099109, 'timestamp': 1715358250726},
#                         {'x': 1233.4476596960662, 'y': 305.10818700944344, 'timestamp': 1715358250736},
#                         {'x': 1405.4391518482757, 'y': 724.5182594926556, 'timestamp': 1715358250746},
#                         {'x': 403.33156848536674, 'y': 826.3063828782667, 'timestamp': 1715358250756},
#                         {'x': 1771.0281376105122, 'y': 453.2032275919212, 'timestamp': 1715358250766},
#                         {'x': 179.9783004139153, 'y': 219.52344440407552, 'timestamp': 1715358250776},
#                         {'x': 1535.7707767543598, 'y': 1032.8589788496215, 'timestamp': 1715358250786},
#                         {'x': 949.212902199926, 'y': 607.8865457839801, 'timestamp': 1715358250796},
#                         {'x': 870.5354888882188, 'y': 521.9416777623795, 'timestamp': 1715358250806},
#                         {'x': 602.560848841942, 'y': 427.29371693550934, 'timestamp': 1715358250816}])
# fixations = detect_fixations(samples)
# print(fixations.head())
