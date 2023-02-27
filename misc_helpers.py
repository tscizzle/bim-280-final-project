import math
from datetime import timedelta
import bisect


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
    r = math.sqrt(x ** 2 + y ** 2)
    theta = math.atan2(y, x)
    return r, theta


def create_relative_response_matrix(data, event_name, bin_size):
    """
    Create a relative response matrix.

    Axis 0 is trial, so the length is the number of trials for the given event. Axis 1
    is binned spike counts for all neurons, so the length is the number of neurons times
    the number of bins in our response window.

    Each value is how many spikes a certain neuron had within that time bin relative to
    the trial start. So for a given trial (row), if there are 20 bins (say, window of
    200ms and bins of 10ms), columns 0-19 are the first neuron's bins, 20-39 are the
    next neuron's bins, etc. The order of neurons is sorted alphabetically by name.

    :param dict data: Dictionary loaded from the input JSON, with "events" and "neurons"
        fields saying the timing of the events and when the neurons spiked.
    :param string event_name: Name of the event we are getting the neural response for.
        This will be a key in the data's "events" dict.
    :param float bin_size: In seconds, time width of each bin within which to count
        spikes.

    :return np.ndarray relative_response_matrix: Matrix with dimensions
        num_trials x (num_bins * num_neurons)
    """

    num_trials = len(data["events"][event_name])
    num_neurons = len(data["neurons"])
    num_bins = int((RESPONSE_END - RESPONSE_START) / bin_size)

    relative_response_matrix = np.zeros((num_trials, num_neurons * num_bins), dtype=int)

    # For each trial of this event...
    for row_idx, event_time in enumerate(data["events"][event_name]):

        # For each neuron...
        for neuron_idx, spike_times in enumerate(sorted(data["neurons"].values())):

            # Since all neurons' bins are horizontally conctenated in each trial's row,
            # appropriately offset this neuron's bins. (i.e. If there's 20 bins, for the
            # first neuron use indices 0-19, for the next neuron use indices 20-39, ...)
            neuron_column_offset = neuron_idx * num_bins

            # Get when this trial started, since spikes will be counted in time bins
            # relative to that.
            window_start = event_time + RESPONSE_START
            window_end = event_time + RESPONSE_END

            # Find the first and last spike time within the response time range.
            # (Instead of iterating through all the spike times in linear time, we can
            # in log time first filter to the relevant small range of spike times.)
            first_spike_idx = bisect.bisect_left(spike_times, window_start)
            last_spike_idx = bisect.bisect_right(spike_times, window_end)

            # For each neuron spike...
            for spike_time in spike_times[first_spike_idx:last_spike_idx]:

                # Calculate the bin this spike belongs in relative to the beginning of
                # the window for this specific trial.
                bin_idx = math.floor((spike_time - window_start) / bin_size)

                # If the bin falls within our response window...
                if 0 <= bin_idx < num_bins:

                    # Get the column to add this spike to, given the neuron's offset in
                    # the row as well as the bin this spike falls in.
                    col_idx = neuron_column_offset + bin_idx

                    # Increment the running spike count for the calculated spot.
                    relative_response_matrix[row_idx, col_idx] += 1

    return relative_response_matrix
