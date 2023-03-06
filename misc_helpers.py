import time
import math
from datetime import timedelta
import bisect

import numpy as np


def display_secs_as_time(secs):
    """
    Given a number of seconds, get a string in a human-readable time format, similar to
    hh:mm:ss.

    :param number secs:

    :return str:
    """
    # Get a number of seconds in h:mm:ss format.
    time_string = str(timedelta(seconds=secs))
    # If the precision is sub-second (i.e. there's a decimal), strip trailing 0's.
    if "." in time_string:
        time_string = time_string.rstrip("0")

    return time_string


def polar_coords(x, y):
    """
    From Euclidean x and y, calculate polar coordinates
        : r (distance from origin), and
        : theta (angle from x-axis, between -pi and pi).

    :return (r, theta):
    """
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    return r, theta


def get_event_timeline(nwbfile):
    """
    Make a list of all trial-related events, like go cue, hitting the target, etc.
    sorted by time.

    :param NWBFile nwbfile: An NWBFile, as returned by `NWBHDF5IO.read`.

    :return dict[]: Each dict in the list has fields "timestamp" and "field". The
        timestamp is seconds into the session, and the field is one of "start_time",
        "go_cue_time", etc. as they appear in the trials structure of the nwb file.
    """
    time_fields = [
        "start_time",
        "stop_time",
        "fail_time",
        "go_cue_time",
        "reach_time",
        "target_hold_time",
        "target_held_time",
        "target_shown_time",
        "target_acquire_time",
    ]

    all_events = []
    for field in time_fields:
        for val in nwbfile.trials[field]:
            timestamps = val if isinstance(val, np.ndarray) else [val]
            for t in timestamps:
                if not np.isnan(t):
                    all_events.append(
                        {
                            "timestamp": t,
                            "field": field,
                        }
                    )

    sorted_events = sorted(all_events, key=lambda d: d["timestamp"])

    return sorted_events


def get_binned_firing_rates(nwbfile, bin_size):
    """
    Make a matrix of all time bins in the session, counting how many spikes each channel
    has within each bin, and scaling to represent firing rate in spikes per second.

    :param NWBFile nwbfile: An NWBFile, as returned by `NWBHDF5IO.read`.
    :param float bin_size: Length (in seconds) of each time bin.

    :return np.ndarray binned_firing_rates: Matrix with dimensions
        (num_bins x num_channels)
    """
    # Each element is a channel's spiking activity, in the form of a list of times
    # (in seconds from session start) that spikes occurred.
    spike_times_by_channel = nwbfile.units["spike_times"][:]

    # Get the dimensions of the binned spike counts matrix.
    num_channels = len(spike_times_by_channel)
    latest_spike_time = max(t for l in spike_times_by_channel for t in l)
    num_bins = math.ceil(latest_spike_time / bin_size)

    # Initialize the binned spike counts as 0.
    binned_spike_counts = np.zeros((num_bins, num_channels))

    # Iterate over all spikes in all channels and add each one to the appropriate bin.
    for channel_idx, spike_times in enumerate(spike_times_by_channel):
        for spike_time in spike_times:
            bin_idx = math.floor(spike_time / bin_size)
            binned_spike_counts[bin_idx, channel_idx] += 1

    binned_firing_rates = binned_spike_counts / bin_size

    return binned_firing_rates


def get_binned_intended_velocities(nwbfile, bin_size):
    """
    Make a matrix of all time bins in the session, calculating the average vector from
    hand position to target within each bin.

    :param NWBFile nwbfile: An NWBFile, as returned by `NWBHDF5IO.read`.
    :param float bin_size: Length (in seconds) of each time bin.

    :return np.ndarray binned_intended_velocities: Matrix with dimensions (num_bins x 2)
    """
    hand_positions = nwbfile.processing["behavior"]["Position"]["Hand"]
    hand_position_timestamps = hand_positions.timestamps
    hand_position_vectors = hand_positions.data
    trial_target_positions = nwbfile.trials["target_pos"]
    trial_go_cue_times = nwbfile.trials["go_cue_time"]
    trial_stop_times = nwbfile.trials["stop_time"]

    latest_trial_stop = trial_stop_times[-1]
    num_bins = math.ceil(latest_trial_stop / bin_size)

    binned_intended_velocities = np.empty((num_bins, 2))
    binned_intended_velocities.fill(np.nan)

    cur_trial_idx = 0
    cur_behavior_idx = 0
    for bin_idx in range(num_bins):
        bin_start = bin_idx * bin_size
        bin_end = (bin_idx + 1) * bin_size
        bin_midpoint = (bin_end + bin_start) / 2

        # Get all the hand positions in this time bin.
        bin_hand_positions = []
        while (
            cur_behavior_idx < len(hand_position_timestamps)
            and hand_position_timestamps[cur_behavior_idx] < bin_end
        ):
            if hand_position_timestamps[cur_behavior_idx] >= bin_start:
                bin_hand_positions.append(hand_position_vectors[cur_behavior_idx])
            cur_behavior_idx += 1
        # If there are no hand positions given in this time bin, skip this bin.
        if len(bin_hand_positions) == 0:
            continue
        # Get the average hand position in this time bin.
        bin_hand_positions = np.array(bin_hand_positions)
        bin_avg_hand_position = np.mean(bin_hand_positions, axis=0)

        # Get the target position in this time bin. (Use the midpoint of the bin, so we
        # only use a target if it was active for more than half the bin.)
        bin_target_position = None
        while (
            cur_trial_idx < len(trial_stop_times)
            and trial_stop_times[cur_trial_idx] < bin_start
        ):
            cur_trial_idx += 1
        trial_go_cue_time = trial_go_cue_times[cur_trial_idx]
        trial_stop_time = trial_stop_times[cur_trial_idx]
        if trial_go_cue_time < bin_midpoint < trial_stop_time:
            bin_target_position = trial_target_positions[cur_trial_idx]

        if bin_target_position is not None:
            # Assume intended velocity is the vector from the hand to the target.
            bin_intended_velocity = bin_target_position - bin_avg_hand_position
            # Take just x and y. No z.
            bin_intended_velocity = bin_intended_velocity[:2]

            binned_intended_velocities[bin_idx] = bin_intended_velocity

    return binned_intended_velocities


def get_behavior_idxs_by_trial_idx(nwbfile):
    """
    Get the indices in the behavior data that map to each trial.

    :param NWBFile nwbfile: An NWBFile, as returned by `NWBHDF5IO.read`.

    :return dict behavior_idxs_by_trial_idx: Keys are trial idx. Values are dicts with
        keys "start_idx" and "stop_idx" which are the indices in the behavior array that
        correspond to the start and stop timestamps for that trial.
    """

    behavior_idxs_by_trial_idx = {}

    # Loop through trials and behavior stream in parallel, accruing the mapping from
    # trial start times to indices of the behavior stream.
    trial_start_times = nwbfile.trials["start_time"]
    trial_go_cue_times = nwbfile.trials["go_cue_time"]
    trial_stop_times = nwbfile.trials["stop_time"]
    hand_positions = nwbfile.processing["behavior"]["Position"]["Hand"]
    behavior_timestamps = hand_positions.timestamps
    num_trials = len(trial_start_times)
    num_behavior_idxs = len(behavior_timestamps)
    cur_trial_idx = 0
    cur_behavior_idx = 0
    while cur_trial_idx < num_trials:
        trial_start_time = trial_start_times[cur_trial_idx]
        trial_go_cue_time = trial_go_cue_times[cur_trial_idx]
        trial_stop_time = trial_stop_times[cur_trial_idx]
        start_idx = None
        stop_idx = None

        while cur_behavior_idx < num_behavior_idxs:
            behavior_timestamp = behavior_timestamps[cur_behavior_idx]
            if behavior_timestamp == trial_start_time:
                start_idx = cur_behavior_idx
            elif behavior_timestamp == trial_go_cue_time:
                go_cue_idx = behavior_timestamp
            elif behavior_timestamp == trial_stop_time:
                stop_idx = cur_behavior_idx

            cur_behavior_idx += 1

            if stop_idx is not None:
                break

        behavior_idxs_by_trial_idx[cur_trial_idx] = {
            "start_idx": start_idx,
            "go_cue_idx": go_cue_idx,
            "stop_idx": stop_idx,
        }

        cur_trial_idx += 1

    return behavior_idxs_by_trial_idx
