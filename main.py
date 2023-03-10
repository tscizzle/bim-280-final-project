import time
import math
import os
import glob

import numpy as np
from sklearn.linear_model import LinearRegression
from pynwb import NWBHDF5IO

from misc_helpers import (
    cached_func,
    get_event_timeline,
    get_binned_firing_rates,
    get_binned_intended_velocities,
    get_behavior_idxs_by_trial_idx,
    get_kalman_model,
    get_binned_predicted_velocities_kalman,
    LinearDecoderWithSmoothing,
)
from inspection_helpers import (
    display_event_timeline,
    inspect_trials_columns,
    plot_trial_hand_trajectories,
    plot_predicted_channel_firing_rates,
)


## Constants.


BIN_SIZE = 0.02


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
        print("Loaded data.")

        ## Get intermediate data (like binned firing rates and binned intended
        ## velocities).

        print("\nGetting binned intended velocities...\n")
        binned_intended_velocities = cached_func(
            "binned_intended_velocities.npy",
            lambda: get_binned_intended_velocities(nwbfile, BIN_SIZE),
        )
        print("Got binned intended velocities.")
        print(f"binned_intended_velocities.shape: {binned_intended_velocities.shape}")

        print("\nGetting binned firing rates...\n")
        binned_firing_rates = cached_func(
            "binned_firing_rates.npy",
            lambda: get_binned_firing_rates(nwbfile, BIN_SIZE),
        )
        print("Got binned firing rates.")
        print(f"binned_firing_rates.shape: {binned_firing_rates.shape}")

        num_bins_total = min(
            [binned_firing_rates.shape[0], binned_intended_velocities.shape[0]]
        )
        num_bins_train = int(num_bins_total * 4 / 5)
        num_bins_validate = int(num_bins_train * 9 / 10)

        ## TODO: 10 times, hold out validation set, train on the rest, and evaluate
        ##  angle error
        ## TODO: for each of these folds, grid search time lag 0 - 3, alpha 0.1 to 0.9,
        ##  and beta 0.1 to 0.9
        ## TODO: end up with hyperparameters, and a guess for how good performance
        ##  should be on the test set.
        ## TODO: run the decoder on the test set and get a performance for angle error.

        ## Create a linear velocity decoder (v_t = (a * v_{t-1}) + b * M * z_t).

        print("\nCreating linear decoder...\n")
        # Assume there is a lag between firing and intention, so map intended velocity
        # to firing rates from a few bins prior.
        neural_fitting_time_lag = 0.06
        neural_fitting_bin_lag = int(neural_fitting_time_lag / BIN_SIZE)
        # Get samples to put into linear regression.
        #   - Skip bins with no target.
        #   - Use firing rates from a slight time lag earlier.
        regression_input_X = []
        regression_input_Y = []
        for bin_idx in range(neural_fitting_bin_lag, num_bins_train):
            bin_intended_velocity = binned_intended_velocities[bin_idx]
            bin_firing_rate = binned_firing_rates[bin_idx - neural_fitting_bin_lag]
            # If there is no target during this bin, intended velocity will contain nan.
            bin_is_valid = not np.isnan(np.sum(bin_intended_velocity))
            if bin_is_valid:
                regression_input_X.append(bin_firing_rate)
                regression_input_Y.append(bin_intended_velocity)
        regression_input_X = np.array(regression_input_X)
        regression_input_Y = np.array(regression_input_Y)
        # Run the regression to get the preferred directions matrix.
        M = LinearRegression().fit(regression_input_X, regression_input_Y)
        # Define alpha and beta arbitrarily.
        alpha, beta = 0.9, 0.1
        linear_decoder = LinearDecoderWithSmoothing(
            alpha, beta, M, BIN_SIZE, neural_fitting_time_lag
        )
        print("Created linear decoder.")

        ## Use the decoder to predict intended velocities.

        print("\nUsing linear decoder to predict velocities...\n")
        binned_predicted_velocities = np.array(
            [
                linear_decoder.predict(bin_firing_rates)
                for bin_firing_rates in binned_firing_rates
            ]
        )
        print("Used linear decoder to predict velocities.")
        print(f"binned_predicted_velocities.shape: {binned_predicted_velocities.shape}")

        ## Visualize trial trajectory and predicted velocities.

        plot_trial_hand_trajectories(
            nwbfile,
            BIN_SIZE,
            trial_idxs=range(200),
            predicted_velocities=binned_predicted_velocities,
            slow_factor=10,
            # show_plot=True,
        )


if __name__ == "__main__":
    main()
