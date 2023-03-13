import os
import math

import numpy as np


## Constants.

# Files and directories.

CACHE_DIR = "cache"


class FiringRatesSimulator:
    def __init__(self, calibration_filename):
        """
        TODO: doc this
        """
        self.calibration_filename = calibration_filename

        # We want the numbers to look familiar, in the range of 50-100.
        self.FIRING_RATE_SCALE = 100
        self.FIRING_RATE_NOISE_STDDEV = 30

        self.preferred_directions = None

        if self.calibration_filename is None:
            self.randomize_preferred_directions()
        else:
            self.load_preferred_directions()

    def randomize_preferred_directions(self):
        """
        TODO: doc this
        """
        preferred_angles = np.random.random(192) * 2 * math.pi
        preferred_directions = np.array(
            [[math.cos(angle), math.sin(angle)] for angle in preferred_angles]
        )
        self.preferred_directions = preferred_directions

    def load_preferred_directions(self):
        """
        TODO: doc this
        """
        cache_filepath = os.path.join(CACHE_DIR, self.calibration_filename)
        with open(cache_filepath, "rb") as f:
            npzfile = np.load(f)
            self.preferred_directions = npzfile["preferred_directions"]

    def encode_velocity_as_firing_rates(self, velocity_vector):
        """
        TODO: doc this
        """
        firing_rates = []

        for preferred_direction in self.preferred_directions:
            # Get firing rate of this channel as if it's cosine tuned to velocity.
            firing_rate = np.dot(velocity_vector, preferred_direction)
            # Scale it so the numbers look familiar.
            firing_rate *= self.FIRING_RATE_SCALE
            # Add Gaussian noise.
            firing_rate += np.random.normal(scale=self.FIRING_RATE_NOISE_STDDEV)
            # But firing rate below 0 doesn't make sense, so bound it.
            firing_rate = max([firing_rate, 0])

            firing_rates.append(firing_rate)

        firing_rates = np.array(firing_rates)

        return firing_rates
