import time
import math
import os
import glob

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from filterpy.kalman import KalmanFilter

from misc_helpers import (
    cached_func,
    get_event_timeline,
    get_binned_firing_rates,
    get_binned_intended_velocities,
    get_behavior_idxs_by_trial_idx,
    get_kalman_model,
    get_binned_predicted_velocities_kalman,
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

        binned_intended_velocities = cached_func(
            "binned_intended_velocities.npy",
            lambda: get_binned_intended_velocities(nwbfile, BIN_SIZE),
        )
        print(f"binned_intended_velocities.shape: {binned_intended_velocities.shape}")

        binned_firing_rates = cached_func(
            "binned_firing_rates.npy",
            lambda: get_binned_firing_rates(nwbfile, BIN_SIZE),
        )
        print(f"binned_firing_rates.shape: {binned_firing_rates.shape}")

        ## Create a linear velocity decoder (v_t = (a * v_{t-1}) + b * M * z_t).

        # # Get samples to put into linear regression (skip bins with no target).
        # regression_input_X = []
        # regression_input_Y = []
        # num_bins = min(
        #     [binned_firing_rates.shape[0], binned_intended_velocities.shape[0]]
        # )
        # for bin_idx in range(num_bins):
        #     bin_intended_velocity = binned_intended_velocities[bin_idx]
        #     bin_firing_rate = binned_firing_rates[bin_idx]
        #     # If there is no target during this bin, intended velocity will contain nan.
        #     bin_is_valid = not np.isnan(np.sum(bin_intended_velocity))
        #     if bin_is_valid:
        #         regression_input_X.append(bin_firing_rate)
        #         regression_input_Y.append(bin_intended_velocity)
        # regression_input_X = np.array(regression_input_X)
        # regression_input_Y = np.array(regression_input_Y)
        # # Run the regression to get the preferred directions matrix.
        # M = LinearRegression().fit(regression_input_X, regression_input_Y).coef_

        ## Use the decoder to predict intended velocities.

        # binned_predicted_velocities = cached_func(
        #     "binned_predicted_velocities.npy",
        #     lambda: get_binned_predicted_velocities_kalman(
        #         binned_firing_rates,
        #         kalman_model,
        #     ),
        # )
        # print(f"binned_predicted_velocities.shape: {binned_predicted_velocities.shape}")

        ## Visualize trial trajectory and predicted velocities.

        # plot_trial_hand_trajectories(
        #     nwbfile,
        #     BIN_SIZE,
        #     trial_idxs=range(20, 30),
        #     predicted_velocities=binned_predicted_velocities,
        #     slow_factor=10,
        #     show_plot=True,
        # )


if __name__ == "__main__":
    main()
