import os
import time
import math
from datetime import timedelta
import bisect

import numpy as np
from sklearn.linear_model import LinearRegression
from filterpy.kalman import KalmanFilter


## Constants.


CACHE_DIR = "cache"


def cache_result(cache_filename, result):
    """
    Save a numpy object to a cache file in our cache directory.

    :param str cache_filename: E.g. Name of the cache file to load from or save to,
        preferably with the .npy file extension to make it clear what it is, e.g.
        "binned_firing_rates.npy"
    :param numpy.ndarray result: Really anything that `numpy.save` can handle.
    """
    cache_filepath = os.path.join(CACHE_DIR, cache_filename)

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    with open(cache_filepath, "wb") as f:
        np.save(f, result, allow_pickle=True)


def cached_func(cache_filename, func):
    """
    Run a function which generates a numpy serializable object. However, first check if
    the cached result exists, in which case don't run the function again, and instead
    just load from the cached file. If the cache does not exist, do run the function,
    and also save the result as a cache file.

    :param str cache_filename: E.g. Name of the cache file to load from or save to,
        preferably with the .npy file extension to make it clear what it is, e.g.
        "binned_firing_rates.npy"
    :param function func: Function (of no arguments) to run, which generates the object
        we want. If the function you want to run takes arguments, you can simply define
        a lambda function in-line, like `lambda: my_func(arg1, arg2)`, to make an
        equivalent function that takes no arguments.

    :return result: Numpy savable/loadable object, e.g. a numpy.ndarray.
    """
    cache_filepath = os.path.join(CACHE_DIR, cache_filename)

    # If cache file exists, just load from that.
    if os.path.exists(cache_filepath):
        with open(cache_filepath, "rb") as f:
            result = np.load(f, allow_pickle=True)
            print(f"Loaded {cache_filename} from cache.")
    else:
        # Otherwise, calculate the binned intended velocities.
        print(f"Calculating {cache_filename}...")
        result = func()
        # And cache them for future runs.
        cache_result(cache_filename, result)
        print(f"Calculated and saved {cache_filename}.")

    return result


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

    :return numpy.ndarray binned_firing_rates: Matrix with dimensions
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

    :return numpy.ndarray binned_intended_velocities: Matrix with dimensions
        (num_bins x 2)
    """
    hand_positions = nwbfile.processing["behavior"]["Position"]["Hand"]
    hand_position_timestamps = hand_positions.timestamps
    hand_position_vectors = hand_positions.data
    trial_target_positions = nwbfile.trials["target_pos"]
    trial_go_cue_times = nwbfile.trials["go_cue_time"]
    trial_target_acquire_times = nwbfile.trials["target_acquire_time"]
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
        trial_target_acquire_time = trial_target_acquire_times[cur_trial_idx][-1]
        trial_stop_time = trial_stop_times[cur_trial_idx]
        trial_target_end_time = (
            trial_target_acquire_time
            if not np.isnan(trial_target_acquire_time)
            and trial_target_acquire_time < trial_stop_time
            else trial_stop_time
        )
        # Only consider the target active if we are between a go cue and the last
        # acquisition of the target (i.e. ignoring the dwell phase).
        if trial_go_cue_time < bin_midpoint < trial_target_end_time:
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

    :return dict behavior_idxs_by_trial_idx: Axis 0 is trial idx. Values in each column
        are the indices into the behavior array that correspond to certain timestamps
        for that trial.
        - Column 0 is start_idx.
        - Column 1 is go_cue_idx.
        - Column 2 is stop_idx.
    """
    trial_start_times = nwbfile.trials["start_time"]
    num_trials = len(trial_start_times)

    behavior_idxs_by_trial_idx = np.empty((num_trials,), dtype=object)
    behavior_idxs_by_trial_idx.fill(np.nan)

    # Loop through trials and behavior stream in parallel, accruing the array of trial
    # rows, where each row has important indices of the behavior stream.
    trial_go_cue_times = nwbfile.trials["go_cue_time"]
    trial_target_acquire_times = nwbfile.trials["target_acquire_time"]
    trial_stop_times = nwbfile.trials["stop_time"]
    hand_positions = nwbfile.processing["behavior"]["Position"]["Hand"]
    behavior_timestamps = hand_positions.timestamps
    num_behavior_idxs = len(behavior_timestamps)
    cur_behavior_idx = 0
    for trial_idx in range(num_trials):
        trial_start_time = trial_start_times[trial_idx]
        trial_go_cue_time = trial_go_cue_times[trial_idx]
        trial_target_acquire_time = trial_target_acquire_times[trial_idx]
        trial_stop_time = trial_stop_times[trial_idx]
        start_idx = None
        go_cue_idx = None
        target_acquire_idx = None
        stop_idx = None

        while cur_behavior_idx < num_behavior_idxs:
            behavior_timestamp = behavior_timestamps[cur_behavior_idx]
            if behavior_timestamp == trial_start_time:
                start_idx = cur_behavior_idx
            elif behavior_timestamp == trial_go_cue_time:
                go_cue_idx = cur_behavior_idx
            elif behavior_timestamp == trial_target_acquire_time[-1]:
                target_acquire_idx = cur_behavior_idx
            elif behavior_timestamp == trial_stop_time:
                stop_idx = cur_behavior_idx

            cur_behavior_idx += 1

            if stop_idx is not None:
                break

        # Take the important timestamps for this trial, and store their behavior indices
        # in the row for this trial.
        behavior_idxs_dict = {
            "start_idx": start_idx,
            "go_cue_idx": go_cue_idx,
            "target_acquire_idx": target_acquire_idx,
            "stop_idx": stop_idx,
        }
        behavior_idxs_by_trial_idx[trial_idx] = behavior_idxs_dict

    return behavior_idxs_by_trial_idx


def get_kalman_model(binned_firing_rates, binned_intended_velocities):
    """
    Not doc'ing this since I'm not using it.
    """

    # Constant term to account for baseline firing rates, v_x, v_y.
    state_dimension = 3
    # Measurements from two 96-channel electrode arrays.
    measurement_dimension = 192

    kalman_model = KalmanFilter(dim_x=state_dimension, dim_z=measurement_dimension)

    # Calculate measurement matrix (mapping intended velocities to firing rates).
    # First, get samples to put into linear regression (skip bins with no target).
    regression_input_X = []
    regression_input_Y = []
    num_bins = min([binned_firing_rates.shape[0], binned_intended_velocities.shape[0]])
    for bin_idx in range(num_bins):
        bin_intended_velocity = binned_intended_velocities[bin_idx]
        bin_firing_rate = binned_firing_rates[bin_idx]
        # If there is no target during this bin, intended velocity will contain nan.
        bin_is_valid = not np.isnan(np.sum(bin_intended_velocity))
        if bin_is_valid:
            regression_input_X.append(bin_intended_velocity)
            regression_input_Y.append(bin_firing_rate)
    regression_input_X = np.array(regression_input_X)
    regression_input_Y = np.array(regression_input_Y)
    # Run the regression to get the measurement matrix.
    measurement_model = LinearRegression().fit(regression_input_X, regression_input_Y)
    # Since we have a constant term in addition to x and y velocity, add on the
    # intercept terms as the left column.
    measurement_matrix = np.hstack(
        (measurement_model.intercept_[:, np.newaxis], measurement_model.coef_)
    )

    # Initial state. 1 for the constant term, and 0 velocity.
    kalman_model.x = np.array([1, 0, 0])

    # Measurement function. Mapping from state to expected measurements. (192 x 3)
    kalman_model.H = measurement_matrix

    # State transition matrix. Map previous state to next state based on kinematics
    # alone. We assume x and y are independent, and that they merely have a slight
    # damping factor.
    velocity_damping_factor = 0.9
    kalman_model.F = np.array(
        [
            [1, 0, 0],
            [0, velocity_damping_factor, 0],
            [0, 0, velocity_damping_factor],
        ]
    )

    # Estimate uncertainty matrix. Assume the only error is in the velocity (the
    # constant term should always be 1), and the error in x and y do not covary.
    kalman_model.P = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    # Measurement uncertainty. Not sure how to get the uncertainty/noise in
    # measuring firing rates (typically measurement uncertainty comes from the
    # manufacturer of a device, but besides not having that, that would refer to the
    # voltages themselves, as opposed to the firing rates which are derived from
    # them). For now, trying small random gaussian noise.
    kalman_model.R = np.random.normal(scale=1, size=(192, 192))

    return kalman_model


def get_binned_predicted_velocities_kalman(binned_firing_rates, kalman_model):
    """
    Use a Kalman filter model to predict velocity vectors based on neural firing rates.

    :param numpy.ndarray binned_firing_rates: Matrix with dimensions
        (num_bins x num_channels)
    :param filterpy.kalman.KalmanFilter kalman_model: Our Kalman filter model, with
        parameters already fit and defined and everything. Important is that it has a
        `predict` method, as well as an `update` method which takes in a vector of
        firing rates for a single time bin. It should also expose the current state
        estimate (a 2D velocity vector) as the property `x`.

    :return numpy.ndarray binned_predicted_velocities: Matrix with dimensions
        (num_bins x 2), holding the model's predicted x- and y- velocity at every time
        step.
    """
    num_bins = binned_firing_rates.shape[0]

    binned_predicted_velocities = np.empty((num_bins, 2))
    binned_predicted_velocities.fill(np.nan)
    # Feed in binned firing rates (measurements) one time step at a time, and
    # see what the Kalman model predicts for that time step.
    for bin_idx in range(num_bins):
        if bin_idx % 10000 == 0:
            print(f"predicted: {bin_idx}")

        bin_firing_rates = binned_firing_rates[bin_idx]
        kalman_model.predict()
        kalman_model.update(bin_firing_rates)
        estimated_state = kalman_model.x[1:]

        binned_predicted_velocities[bin_idx] = estimated_state

    return binned_predicted_velocities


def get_signed_angle_between(true_vector, other_vector):
    """
    For two 2D vectors, get the angle between them, but take the direction into account.
    In other words, if you would need to rotate true_vector counter-clockwise to line up
    with other_vector, then return a positive number (between 0 and pi). If clockwise,
    then return a negative number (between 0 and -pi).

    :param numpy.ndarray true_vector: Length 2 vector (x and y), relative to which
        another vector's angle will be determined.
    :param numpy.ndarray other_vector: Length 2 vector (x and y), a vector to determine
        its angle, relative to true_vector.

    :return float relative_other_angle: Radians. Between -pi and pi. Angle you would
        have to rotate true_vector to align it with other_vector.
    """
    # If either vector has nan, return nan.
    if np.isnan(true_vector).any() or np.isnan(other_vector).any():
        return np.nan

    # Get each vector direction as an angle between -pi and pi.
    true_angle = math.atan2(true_vector[1], true_vector[0])
    other_angle = math.atan2(other_vector[1], other_vector[0])

    # Imagine rotating the picture to make true_vector pointing straight right (an angle
    # of 0).
    relative_other_angle = other_angle - true_angle

    if relative_other_angle > math.pi:
        relative_other_angle -= 2 * math.pi
    elif relative_other_angle <= -math.pi:
        relative_other_angle += 2 * math.pi

    return relative_other_angle


class LinearDecoderWithSmoothing:
    """
    Implements a decoder obeying the equation

        v_t = alpha * v_{t-1} + beta * M * z_t

    The first term takes into account the previous velocity prediction (v_{t-1}).
    The second term takes into account the new neural measurements (z_t).
    The full equation takes both of those into account, weighted by alpha and beta.

    This class maintains some state, because the prediction for the current time step
    depends on the previous prediction and possibly previous neural measurements.
    """

    def __init__(self, alpha, beta, M, delta_t, neural_time_lag):
        """
        :param float alpha: Weight of the previous predicted velocity for the next
            predicted velocity.
        :param float beta: Weight of the neurally-predicted velocity.
        :param sklearn.linear_model.LinearRegression M: Model with `predict` method
            which takes in a vector of firing rates and outputs a velocity vector.
        :param float delta_t: Length (in seconds) between each prediction.
        :param float neural_time_lag: Length (in seconds) we are assuming the lag is
            between neural firing and intended velocity. We fit the linear mapping from
            firing rates to velocities based with this time lag in mind, so we should
            use the same lag in this model's predictions.
        """
        self.alpha = alpha
        self.beta = beta
        self.M = M
        self.delta_t = delta_t
        self.neural_time_lag = neural_time_lag
        self.neural_bin_lag = int(neural_time_lag / delta_t)

        self.predicted_velocities = []
        self.measured_firing_rates = []

    def predict(self, firing_rate_vector):
        """
        Given the neural data at the current time step, and previous predicted
            velocities, predict the velocity for the current time step.

        :param np.ndarray firing_rate_vector: Single-axis length-192 firing rate vector
            at the current time step (for the 192 channels).

        :param np.ndarray v_t: Predicted x- and y- velocity for the current time step.
        """
        # Store the measured firing rate vector.
        self.measured_firing_rates.append(firing_rate_vector)

        # Assuming there is enough history (1 prediction), get the previous prediction.
        prev_velocity = (
            self.predicted_velocities[-1]
            if len(self.predicted_velocities) > 0
            else np.zeros(2)
        )
        # Assuming there is enough history (a few measurements), get the predicted
        # velocity from the neural activity a few measurements ago (according to the
        # specified time lag).
        lagged_firing_rate_vector = (
            self.measured_firing_rates[-1 - self.neural_bin_lag]
            if len(self.measured_firing_rates) > self.neural_bin_lag
            else np.zeros(192)
        )
        neural_velocity = self.M.predict(lagged_firing_rate_vector.reshape(1, -1))[0]

        # Apply the equation (simple linear mapping, with smoothing).
        predicted_velocity = self.alpha * prev_velocity + self.beta * neural_velocity

        # Store the result for the future.
        self.predicted_velocities.append(predicted_velocity)

        return predicted_velocity
