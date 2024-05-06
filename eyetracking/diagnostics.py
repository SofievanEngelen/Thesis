import matplotlib.pyplot as plt
import pandas as pd


def diagnostic_plot(samples, fixations, start_event=1, start_time=None, duration=2000, interactive=True):
    assert start_event >= 1
    assert start_event <= fixations.shape[0]
    assert start_time >= samples['time'].min()
    assert start_time <= samples['time'].max()

    if start_time is None:
        start_time = fixations.loc[start_event - 1, 'start']

    if interactive:
        plt.figure()

    plt.plot(samples['time'], samples['x'], 'ro-', markersize=3, label='x-coordinate')
    plt.plot(samples['time'], samples['y'], 'bo-', markersize=3, label='y-coordinate')

    for _, row in fixations.iterrows():
        plt.axvline(row['start'], color='lightgrey')
        plt.axvline(row['end'], color='lightgrey')
        plt.axhline(row['x'], linestyle='--', color='grey')
        plt.axhline(row['y'], linestyle='--', color='grey')

    plt.xlim(start_time, start_time + duration)
    plt.ylim(min(fixations[['x', 'y']].values.flatten()), max(fixations[['x', 'y']].values.flatten()))

    plt.xlabel('Time (ms)')
    plt.ylabel('Position (pixels)')
    plt.title('Diagnostic Plot of Samples and Fixations')
    plt.legend()

    if interactive:
        plt.show()

def diagnostic_plot_event_types(fixations):
    event_colors = {'fixation': 'black', 'blink': 'red', 'too short': 'blue', 'too dispersed': 'green'}
    fixations['event_color'] = fixations['event'].map(event_colors)

    pd.plotting.scatter_matrix(
        fixations[['mad.x', 'mad.y', 'dur']],
        c=fixations['event_color'],
        alpha=0.5,
        figsize=(8, 8)
    )

def calculate_summary(fixations):
    stats = pd.DataFrame(index=["Number of trials", "Duration of trials", "No. of fixations per trial",
                                "Duration of fixations", "Dispersion horizontal", "Dispersion vertical",
                                "Peak velocity horizontal", "Peak velocity vertical"],
                         columns=["mean", "sd"])

    stats.loc["Number of trials", 'mean'] = len(fixations['trial'].unique())
    stats.loc["Duration of trials", 'mean'] = fixations.groupby('trial')['dur'].sum().mean()
    stats.loc["Duration of trials", 'sd'] = fixations.groupby('trial')['dur'].sum().std()

    n_fixations_per_trial = fixations.groupby('trial').size()
    stats.loc["No. of fixations per trial", 'mean'] = n_fixations_per_trial.mean()
    stats.loc["No. of fixations per trial", 'sd'] = n_fixations_per_trial.std()

    stats.loc["Duration of fixations", 'mean'] = fixations['dur'].mean()
    stats.loc["Duration of fixations", 'sd'] = fixations['dur'].std()

    stats.loc["Dispersion horizontal", 'mean'] = fixations['mad.x'].mean()
    stats.loc["Dispersion horizontal", 'sd'] = fixations['mad.x'].std()
    stats.loc["Dispersion vertical", 'mean'] = fixations['mad.y'].mean()
    stats.loc["Dispersion vertical", 'sd'] = fixations['mad.y'].std()

    stats.loc["Peak velocity horizontal", 'mean'] = fixations['peak.vx'].mean()
    stats.loc["Peak velocity horizontal", 'sd'] = fixations['peak.vx'].std()
    stats.loc["Peak velocity vertical", 'mean'] = fixations['peak.vy'].mean()
    stats.loc["Peak velocity vertical", 'sd'] = fixations['peak.vy'].std()

    return stats
