import os

import numpy as np
from sklearn.linear_model import LinearRegression

from misc_helpers import LinearDecoderWithSmoothing


## Constants.

# Files and directories.

CACHE_DIR = "cache"


class ClosedLoopDecoder:
    def __init__(self, calibration_filename):
        """
        TODO: doc this
        """
        self.calibration_filename = calibration_filename

        self.alpha = 0.5
        self.beta = 0.5
        self.M = None

        self.predicted_velocities = []

        self.training_data = {
            "timestamps": [],
            "firing_rates": [],
            "intended_velocities": [],
        }
        if self.calibration_filename is not None:
            cache_filepath = os.path.join(CACHE_DIR, self.calibration_filename)
            with open(cache_filepath, "rb") as f:
                npzfile = np.load(f)

                self.training_data["timestamps"] = npzfile["timestamps"]
                self.training_data["firing_rates"] = npzfile["firing_rates"]
                self.training_data["intended_velocities"] = npzfile[
                    "intended_velocities"
                ]
            self.train_model()

    def train_model(self):
        """
        TODO: doc this
        """
        regression_input_X = []
        regression_input_Y = []
        for firing_rates, intended_velocity in zip(
            self.training_data["firing_rates"],
            self.training_data["intended_velocities"],
        ):
            # If there is no target during this bin, intended velocity will contain
            # nan.
            bin_is_valid = not np.isnan(np.sum(intended_velocity))
            if bin_is_valid:
                regression_input_X.append(firing_rates)
                regression_input_Y.append(intended_velocity)
        regression_input_X = np.array(regression_input_X)
        regression_input_Y = np.array(regression_input_Y)
        # Run the regression (essentially, gets the preferred directions matrix and an
        # offset term).
        self.M = LinearRegression().fit(regression_input_X, regression_input_Y)

    def predict(self, firing_rate_vector):
        """
        Given the neural data at the current time step, and previous predicted
            velocities, predict the velocity for the current time step.

        :param np.ndarray firing_rate_vector: Single-axis length-192 firing rate vector
            at the current time step (for the 192 channels).

        :param np.ndarray v_t: Predicted x- and y- velocity for the current time step.
        """
        # Assuming there is enough history (1 prediction), get the previous prediction.
        prev_velocity = (
            self.predicted_velocities[-1]
            if len(self.predicted_velocities) > 0
            else np.zeros(2)
        )
        # Get the predicted velocity from the neural activity.
        neural_velocity = self.M.predict(firing_rate_vector.reshape(1, -1))[0]

        # Apply the equation (simple linear mapping, with smoothing).
        predicted_velocity = self.alpha * prev_velocity + self.beta * neural_velocity

        # Store the result for the future.
        self.predicted_velocities.append(predicted_velocity)

        return predicted_velocity
