import numpy as np
import matplotlib.pyplot as plt

from misc_helpers import display_secs_as_time


def plot_session_log(nwbfile):
    """
    Plot various values (like target posiion, cursor position, etc.) over time
    throughout a session.

    :param NWBFile nwbfile: An NWBFile, as returned by `NWBHDF5IO.read`.
    """
    trial_start_times = np.array(nwbfile.trials["start_time"])
    target_x_positions = np.array([p[0] for p in nwbfile.trials["target_pos"]])
    target_y_positions = np.array([p[1] for p in nwbfile.trials["target_pos"]])

    fig, ax = plt.subplots()

    ax.step(trial_start_times, target_x_positions, label="target position (x)")
    ax.step(trial_start_times, target_y_positions, label="target position (y)")

    ax.set_xlabel("time (hh:mm)")
    ax.set_ylabel("distance (m)")
    ax.set_title("Session log")
    ax.xaxis.set_major_formatter(lambda secs, _: display_secs_as_time(secs))
    ax.legend()

    plt.show()


def plot_target_positions(nwbfile):
    """
    Plot all target positions, in space.

    :param NWBFile nwbfile: An NWBFile, as returned by `NWBHDF5IO.read`.
    """
    unique_target_points = set((x, y) for x, y, _ in nwbfile.trials["target_pos"])
    x_vals, y_vals = zip(*unique_target_points)

    fig, ax = plt.subplots()

    plt.scatter(x_vals, y_vals)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Target positions")

    plt.show()


def inspect_trials_columns(nwbfile):
    """
    Print information about each column available in trials.

    :param NWBFile nwbfile: An NWBFile, as returned by `NWBHDF5IO.read`.
    """
    for colname in nwbfile.trials.colnames:
        if colname == "timeseries":
            continue
        print("=====")
        print(f"{colname}: {len(nwbfile.trials[colname])}")
        print(nwbfile.trials[colname][:10])
        print(nwbfile.trials[colname][-10:])
