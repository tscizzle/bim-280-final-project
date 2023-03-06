import time
import math
import os
import glob

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO

from misc_helpers import (
    get_event_timeline,
    get_binned_firing_rates,
    get_binned_intended_velocities,
    get_behavior_idxs_by_trial_idx,
)
from inspection_helpers import (
    display_event_timeline,
    inspect_trials_columns,
    plot_trial_hand_trajectories,
)


## Constants.


BIN_SIZE = 0.05


## Main function.


def main():
    ## Get the relevant data files. (Monkey J. 3-rings task.)

    data_dir = "data"
    nwb_extenstion = ".nwb"
    nwb_filenames = glob.glob(f"{data_dir}/*{nwb_extenstion}")

    days_with_3rings_task = ["2016-01-27", "2016-01-28"]
    filenames_with_3rings_task = sorted(
        [
            filename
            for filename in nwb_filenames
            if any(day.replace("-", "") in filename for day in days_with_3rings_task)
        ]
    )

    if filenames_with_3rings_task:
        print("\nFound relevant sessions:\n")
        for filename in filenames_with_3rings_task:
            print(f"\t{filename}")
    else:
        print("No relevant sessions found.")
        return

    ## Read in data from a session.

    first_session = filenames_with_3rings_task[0]
    print(f"\nLoading data from {first_session}...\n")
    with NWBHDF5IO(first_session, "r") as io:
        nwbfile = io.read()
        print("Loaded data.\n")

        ## Get intermediate data (calculate and save it if it doesn't exist, but load it
        ## if it already exists).

        cache_dir = "cache"

        intended_velocities_cache = os.path.join(
            cache_dir, "binned_intended_velocities.npy"
        )
        if os.path.exists(intended_velocities_cache):
            with open(intended_velocities_cache, "rb") as f:
                binned_intended_velocities = np.load(f)
                print("\nLoaded binned intended velocities from cache.\n")
        else:
            print("\nBinning intended velocities...\n")
            binned_intended_velocities = get_binned_intended_velocities(
                nwbfile, BIN_SIZE
            )
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open(intended_velocities_cache, "wb") as f:
                np.save(f, binned_intended_velocities)
            print("Binned intended velocities.\n")
        print(f"binned_intended_velocities.shape: {binned_intended_velocities.shape}")

        firing_rates_cache = os.path.join(cache_dir, "binned_firing_rates.npy")
        if os.path.exists(firing_rates_cache):
            with open(firing_rates_cache, "rb") as f:
                binned_firing_rates = np.load(f)
                print("\nLoaded binned firing rates from cache.\n")
        else:
            print("\nBinning firing rates...\n")
            binned_firing_rates = get_binned_firing_rates(nwbfile, BIN_SIZE)
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open(firing_rates_cache, "wb") as f:
                np.save(f, binned_firing_rates)
            print("Binned firing rates.\n")
        print(f"binned_firing_rates.shape: {binned_firing_rates.shape}")

        ## Calculate measurement matrix (mapping intended velocities to firing rates).

        # Get samples to put into linear regression (skip bins with no target).
        regression_input_X = []
        regression_input_Y = []
        num_bins = min(
            [binned_firing_rates.shape[0], binned_intended_velocities.shape[0]]
        )
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
        model = LinearRegression().fit(regression_input_X, regression_input_Y)
        measurement_matrix = model.coef_

        ## Plot predicted firing rates of individual channels as direction varies.

        num_channels = binned_firing_rates.shape[1]
        for channel_idx in range(num_channels):
            fig, ax = plt.subplots()

            velocity_magnitude = 70
            angles_to_test = np.arange(0, 2.25 * math.pi, math.pi / 4)
            predictions = []
            for theta in angles_to_test:
                velocity_vector = (
                    np.array([math.cos(theta), math.sin(theta)]) * velocity_magnitude
                )
                channel_firing_rate = model.predict([velocity_vector])[0, channel_idx]
                predictions.append(channel_firing_rate)

            ax.plot(angles_to_test, predictions)

            ax.set_xlabel("angle (radians)")
            ax.set_ylabel("firing rate (spikes / sec)")
            ax.set_xticks(angles_to_test)
            ax.xaxis.set_major_formatter(
                lambda angle, _: f"{round(angle / math.pi, 2)} π"
            )
            ax.set_ylim(0)
            ax.set_title(f"Channel {channel_idx} firing rate by angle")

            plt.show()


if __name__ == "__main__":
    main()
