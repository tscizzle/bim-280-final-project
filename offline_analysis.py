import time
import math
import os
import glob

import numpy as np
from sklearn.linear_model import LinearRegression
from pynwb import NWBHDF5IO
from matplotlib import pyplot as plt

from misc_helpers import (
    cache_result,
    cached_func,
    get_event_timeline,
    get_binned_firing_rates,
    get_binned_intended_velocities,
    get_behavior_idxs_by_trial_idx,
    get_kalman_model,
    get_binned_predicted_velocities_kalman,
    get_signed_angle_between,
    LinearDecoderWithSmoothing,
)
from inspection_helpers import (
    display_event_timeline,
    inspect_trials_columns,
    plot_predicted_channel_firing_rates,
    plot_prediction_angle_errors,
    plot_trial_hand_trajectories,
)

plt.style.use("seaborn-v0_8-dark")


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
        print("Found relevant sessions:")
        for filename in filenames_with_3rings_task:
            print(f"\t{filename}")
    else:
        print("No relevant sessions found.")
        return

    ## Read in data from a session.

    first_session = filenames_with_3rings_task[0]
    print(f"Loading data from {first_session}...")
    with NWBHDF5IO(first_session, "r") as io:
        nwbfile = io.read()
        print("Loaded data.")

        ## Get intermediate data (like binned firing rates and binned intended
        ## velocities).

        print("Getting binned intended velocities...")
        binned_intended_velocities = cached_func(
            "binned_intended_velocities.npy",
            lambda: get_binned_intended_velocities(nwbfile, BIN_SIZE),
        )
        print("Got binned intended velocities.")
        print(f"\tbinned_intended_velocities.shape: {binned_intended_velocities.shape}")

        print("Getting binned firing rates...")
        binned_firing_rates = cached_func(
            "binned_firing_rates.npy",
            lambda: get_binned_firing_rates(nwbfile, BIN_SIZE),
        )
        print("Got binned firing rates.")
        print(f"\tbinned_firing_rates.shape: {binned_firing_rates.shape}")

        ## Set parameters of the train, validate, test process.

        num_bins_total = min(
            [binned_firing_rates.shape[0], binned_intended_velocities.shape[0]]
        )

        holdout_portion = 0.2
        num_bins_holdout = int(num_bins_total * holdout_portion)
        num_bins_train = num_bins_total - num_bins_holdout
        holdout_bins = np.array(range(num_bins_train, num_bins_total))
        train_bins = np.array(range(num_bins_train))

        print(f"num_bins_total: {num_bins_total}")
        print(f"num_bins_holdout: {num_bins_holdout}")
        print(f"num_bins_train: {num_bins_train}")

        ## Train the decoder.

        print("Training...")

        train_start = time.time()

        # These parameters were optimized using grid search.
        neural_fitting_time_lag = 0.0
        alpha = 0.5
        beta = 0.5

        # Assume there is a lag between firing and intention, so map intended
        # velocity to firing rates from a few bins prior.
        neural_fitting_bin_lag = int(neural_fitting_time_lag / BIN_SIZE)
        # Get samples to put into linear regression.
        #   - Skip bins with no target.
        #   - Use firing rates from a slight time lag earlier.
        regression_input_X = []
        regression_input_Y = []
        for bin_idx in train_bins:
            bin_intended_velocity = binned_intended_velocities[bin_idx]
            bin_firing_rate = binned_firing_rates[
                bin_idx - neural_fitting_bin_lag
                if bin_idx >= neural_fitting_bin_lag
                else 0
            ]
            # If there is no target during this bin, intended velocity will contain
            # nan.
            bin_is_valid = not np.isnan(np.sum(bin_intended_velocity))
            if bin_is_valid:
                regression_input_X.append(bin_firing_rate)
                regression_input_Y.append(bin_intended_velocity)
        regression_input_X = np.array(regression_input_X)
        regression_input_Y = np.array(regression_input_Y)
        # Run the regression (essentially, gets the preferred directions matrix, with an
        # offset term).
        M = LinearRegression().fit(regression_input_X, regression_input_Y)

        train_end = time.time()
        train_time = train_end - train_start
        train_time_per_sample = train_time / len(regression_input_X)
        print(
            f"Trained on {len(regression_input_X)} samples in {train_time} seconds "
            f"({train_time_per_sample} seconds per sample)."
        )

        ## Create the decoder, and evaluate it on both the training set and test set.

        data_subsets = [
            {"bins": train_bins, "name": "train"},
            {"bins": holdout_bins, "name": "test"},
        ]
        for data_subset in data_subsets:
            subset_bins = data_subset["bins"]
            subset_name = data_subset["name"]

            # Create the decoder, given the values of M, alpha, beta, and
            # time_lag we've chosen or fit. This decoder implements this
            # equation:
            #   v_t = (a * v_{t-1}) + b * M * z_t
            linear_decoder = LinearDecoderWithSmoothing(
                alpha=alpha,
                beta=beta,
                M=M,
                delta_t=BIN_SIZE,
                neural_time_lag=neural_fitting_time_lag,
            )

            # Use the decoder to predict velocities.

            print("Using decoder to predict velocities...")

            prediction_start = time.time()

            predicted_velocities = np.array(
                [
                    linear_decoder.predict(bin_firing_rates)
                    for bin_firing_rates in binned_firing_rates[subset_bins]
                ]
            )

            prediction_end = time.time()
            prediction_time = prediction_end - prediction_start
            prediction_time_per_sample = prediction_time / len(predicted_velocities)
            print(
                f"Predicted {len(predicted_velocities)} samples in {prediction_time} "
                f"seconds ({prediction_time_per_sample} seconds per sample)."
            )

            print(f"predicted_velocities.shape: {predicted_velocities.shape}")

            # Evaluate by calculating angle errors of the predictions.
            true_velocities = binned_intended_velocities[subset_bins]
            relative_angles = np.array(
                [
                    get_signed_angle_between(true_velocity, predicted_velocity)
                    for true_velocity, predicted_velocity in zip(
                        true_velocities, predicted_velocities
                    )
                ]
            )
            relative_angles = relative_angles[~np.isnan(relative_angles)]

            # Plot the angle error distribution.
            plot_prediction_angle_errors(relative_angles, subset_name)

            ## Visualize trial trajectory and predicted velocities.

            if subset_name == "test":
                full_length_predicted_velocities = np.zeros((num_bins_total, 2))
                full_length_predicted_velocities[subset_bins] = predicted_velocities
                plot_trial_hand_trajectories(
                    nwbfile,
                    BIN_SIZE,
                    trial_idxs=[
                        4007,
                        4019,
                        4021,
                        4031,
                        4038,
                    ],
                    predicted_velocities=full_length_predicted_velocities,
                    slow_factor=10,
                    # show_plot=True,
                )


if __name__ == "__main__":
    main()
