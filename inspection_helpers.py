import time
import os
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle


from misc_helpers import (
    cached_func,
    display_secs_as_time,
    get_behavior_idxs_by_trial_idx,
)


## Constants.


REPORTING_DIR = "reporting"


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

    ax.scatter(x_vals, y_vals)

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


def display_event_timeline(sorted_events):
    """
    Display trial events in a pretty columnar printout, with times and time differences.

    :param dict[] sorted_events: Each dict in the list has fields "timestamp" and
        "field". The timestamp is seconds into the session, and the field is one of
        "start_time", "go_cue_time", etc. as they appear in the trials structure of the
        nwb file.
    """
    prev_time = 0
    for event_dict in sorted_events[:50]:
        t = event_dict["timestamp"]
        f = event_dict["field"]
        print("{:>10f} {:>20s} (+{:>})".format(t, f, round(t - prev_time, 6)))
        prev_time = t


def plot_predicted_channel_firing_rates(firing_rate_model):
    """
    For each channel, show a plot of its predicted firing rate as a function of the
    angle of the intended velocity vector. I.e. As the user attempts or performs
    movements in each direction, what do we think each channel's firing rate will be?
    These plots look like tuning curves (though they are predictions, not raw data).

    :param sklearn.linear_model.LinearRegression firing_rate_model: sklearn's
        `LinearRegression` class is probably what this should be. If using something
        else, it should work if it has a `predict` method with the same signature as
        that of `LinearRegression`, as well as a `coef_` property whose first axis has
        length equal to the number of channels.
    """

    num_channels = firing_rate_model.coef_.shape[0]
    for channel_idx in range(num_channels):
        fig, ax = plt.subplots()

        velocity_magnitude = 70
        angles_to_test = np.arange(0, 2.25 * math.pi, math.pi / 4)
        predictions = []
        for theta in angles_to_test:
            velocity_vector = (
                np.array([math.cos(theta), math.sin(theta)]) * velocity_magnitude
            )
            channel_firing_rate = firing_rate_model.predict([velocity_vector])[
                0, channel_idx
            ]
            predictions.append(channel_firing_rate)

        ax.plot(angles_to_test, predictions)

        ax.set_xlabel("angle (radians)")
        ax.set_ylabel("firing rate (spikes / sec)")
        ax.set_xticks(angles_to_test)
        ax.xaxis.set_major_formatter(lambda angle, _: f"{round(angle / math.pi, 2)} π")
        ax.set_ylim(0)
        ax.set_title(f"Channel {channel_idx} firing rate by angle")

        plt.show()


def plot_trial_hand_trajectories(
    nwbfile,
    bin_size,
    trial_idxs=(3, 4, 5, 6),
    predicted_velocities=None,
    slow_factor=1,
    show_plot=False,
):
    """
    Create gifs of hand trajectories overlaid on the target configuration. Also put
    things such as

    :param NWBFile nwbfile: An NWBFile, as returned by `NWBHDF5IO.read`.
    :param float bin_size: Length (in seconds) of each time bin.
    :param int[] trial_idxs: List of trials to generate trajectory animations for.
    :param numpy.ndarray predicted_velocities: 2D array where axis 0 is which time bin
        and each element is length 2, the x and y velocity predicted by our decoder.
    :param float slow_factor: Multiplier for how slow to run the animation. A value of 2
        makes the animation interval twice as long, so the same animation runs but it is
        visually twice as slow.
    :param bool show_plot: If False, don't pause to show the plot of a trial when it is
        made, just save the video file. If True, pause to show the plot for each trial.
    """

    hand_positions = nwbfile.processing["behavior"]["Position"]["Hand"]

    behavior_idxs_by_trial_idx = cached_func(
        "behavior_idxs_by_trial_idx.npy",
        lambda: get_behavior_idxs_by_trial_idx(nwbfile),
    )

    unique_target_points = set((x, y) for x, y, _ in nwbfile.trials["target_pos"])
    target_radius = nwbfile.trials["target_size"][0][0] / 2

    trial_target_positions = nwbfile.trials["target_pos"]
    trial_start_times = nwbfile.trials["start_time"]
    trial_go_cue_times = nwbfile.trials["go_cue_time"]
    for trial_idx in trial_idxs:
        trial_target_pos = trial_target_positions[trial_idx]
        trial_start_time = trial_start_times[trial_idx]
        trial_go_cue_time = trial_go_cue_times[trial_idx]
        behavior_idxs = behavior_idxs_by_trial_idx[trial_idx]
        start_idx = behavior_idxs[0]
        stop_idx = behavior_idxs[-1]
        trial_hand_positions = hand_positions.data[start_idx:stop_idx]
        trial_hand_x = trial_hand_positions[:, 0]
        trial_hand_y = trial_hand_positions[:, 1]

        # Make an animation of the trial.

        fig, ax = plt.subplots(figsize=(10, 10))

        target_circles = [
            Circle((x, y), radius=target_radius, linewidth=0, color="gray")
            for x, y in unique_target_points
        ]
        target_collection = PatchCollection(target_circles)
        ax.add_collection(target_collection)
        cued_target = Circle(
            trial_target_pos, radius=target_radius, linewidth=0, color="red"
        )
        ax.add_patch(cued_target)
        (hand_trajectory,) = ax.plot([], [], color="blue")
        (predicted_velocity_line,) = ax.plot([], [], color="red")

        ax.set_xlim([-200, 200])
        ax.set_ylim([-200, 200])
        ax.set_aspect("equal", adjustable="box")

        trial_num_ms = stop_idx - start_idx
        ms_per_frame = 50
        anim_num_frames = math.floor(trial_num_ms / ms_per_frame)
        anim_interval = ms_per_frame * slow_factor

        def init_func():
            hand_trajectory.set_data([], [])
            predicted_velocity_line.set_data([], [])
            cued_target.set_color("red")

        def animate(frame):
            num_ms_so_far = frame * ms_per_frame
            num_secs_so_far = num_ms_so_far / 1000
            overall_time = trial_start_time + num_secs_so_far

            hand_trajectory.set_data(
                trial_hand_x[:num_ms_so_far], trial_hand_y[:num_ms_so_far]
            )

            if overall_time >= trial_go_cue_time:
                cued_target.set(color="green")

            if predicted_velocities is not None:
                bin_idx = math.floor(overall_time / bin_size)
                predicted_velocity = predicted_velocities[bin_idx]
                start_point = np.array(
                    [trial_hand_x[num_ms_so_far], trial_hand_y[num_ms_so_far]]
                )
                end_point = start_point + predicted_velocity
                predicted_velocity_points = np.array([start_point, end_point])
                predicted_velocity_line.set_data(
                    predicted_velocity_points[:, 0],
                    predicted_velocity_points[:, 1],
                )

        anim = FuncAnimation(
            fig,
            animate,
            init_func=init_func,
            frames=anim_num_frames,
            interval=anim_interval,
        )

        if show_plot:
            plt.show()
        else:
            fps = 1000 / anim_interval
            anim_filename = f"trial_{trial_idx}_trajectory.gif"
            anim_filepath = os.path.join(REPORTING_DIR, anim_filename)
            anim.save(anim_filepath, writer="pillow", fps=fps)

        plt.close()
