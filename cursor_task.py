import os
import sys
import math
import time
import random
import queue

import numpy as np
import pygame
from pygame.math import Vector2

from firing_rates_simulator import FiringRatesSimulator
from closed_loop_decoder import ClosedLoopDecoder


## Constants.

# Colors.

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_GRAY = (220, 220, 220)
DARK_GRAY = (150, 150, 150)
DARKER_GRAY = (70, 70, 70)
DARKEST_GRAY = (30, 30, 30)
RED = (255, 0, 0)
LIGHT_RED = (255, 130, 130)
LIGHTEST_RED = (255, 230, 230)
LIGHT_BLUE = (150, 150, 255)
LIGHT_GREEN = (150, 255, 150)

# Files and directories.

CACHE_DIR = "cache"


## Game.


class Targets2dGame:
    def __init__(
        self, num_targets=8, dwell_requirement_secs=1, calibration_filename=None
    ):
        """
        Construct a Targets2dGame object.

        :param int num_targets: Number of targets placed in a circle (with an additional
            target placed in the center)
        :param float dwell_requirement_secs: Seconds the cursor must remain continuously
            on the cued target to select it and complete the trial.
        :param str calibration_filename: If not supplied, we are in open-loop, where we
            are only storing data (neural firing rates and intended velocities), not
            using it to control the cursor. If supplied, we are in closed-loop, where we
            train the decoder first, then use the decoder to control the cursor.
        """

        ## Specifiable game parameters.

        self.num_targets = num_targets
        self.dwell_requirement_secs = dwell_requirement_secs
        self.calibration_filename = calibration_filename
        self.is_open_loop = self.calibration_filename is None
        self.is_closed_loop = not self.is_open_loop

        ## Constant game parameters (can be made specifiable if desired).

        self.FPS = 50
        # Distance from edge of screen to edge distal target on the shortest screen
        # dimension.
        self.PADDING_AROUND_TARGET_RING_px = 50
        self.TARGET_RADIUS_px = 50
        self.CURSOR_RADIUS_px = 30
        self.TARGET_TOUCHING_DISTANCE_px = self.TARGET_RADIUS_px + self.CURSOR_RADIUS_px
        self.TRIAL_TIMEOUT_sec = 10
        self.CENTER_TARGET_ID = "center"
        self.SIMULATOR_SIZE = 400

        ## Misc game-wide references (many are set during game initialization).

        self.game_window = None
        self.window_width_px = None
        self.window_height_px = None
        # The shorter of width and height.
        self.shortest_dim_px = None
        self.window_center = None
        # Distance from center of center target to center of distal target.
        self.target_ring_radius_px = None
        # Queue for putting cursor movement events into. For example, the BRAND node
        # which uses this game takes redis stream entries which represent velocities and
        # puts them here to control the cursor.
        self.cursor_movements_queue = queue.Queue()
        self.frame_update_timestamp = None
        self.frame_idx = 0
        # Simulator area info.
        self.simulator_left = None
        self.simulator_right = None
        self.simulator_top = None
        self.simulator_bottom = None
        self.simulator_center = None
        # Neural data simulation.
        self.neural_simulator = FiringRatesSimulator(self.calibration_filename)
        # Neural decoding.
        self.neural_decoder = ClosedLoopDecoder(self.calibration_filename)
        # Storing calibration data during open-loop.
        self.open_loop_samples = {
            "timestamps": [],
            "firing_rates": [],
            "intended_velocities": [],
        }

        ## Task state.

        self.trial_idx = -1
        # Id of target currently cued.
        self.target_cued = None
        # When the current target cue began.
        self.target_cued_timestamp = None
        # When the cursor last began touching a target (helps calculate dwell timing).
        self.target_last_entered_timestamp = None

        ## Game state for drawing objects.

        # Circle being controlled by user.
        self.cursor_position = Vector2(0, 0)
        # Targets for user to aim for when cued.
        self.target_objs = {}
        # The target that the cursor is touching (if any).
        self.target_cursor_touching = False
        # Mouse position within in the simulator.
        self.simulator_mouse_position = None
        # Simulated firing rate vector.
        self.current_firing_rates = np.zeros(192)
        # Electrodes for visualizing firing rates.
        self.electrode_objs = {}

    ####################################################################################
    # Lifecycle methods
    ####################################################################################

    def init_game(self):
        """
        Initialize the game state, open the game window, preparing the game to be
        played.
        """

        ## Misc initialization.

        pygame.init()

        # Create a window for the game.
        self.game_window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("Targets Game")

        ## Calculate some things before setting all the game state.

        self.window_width_px, self.window_height_px = self.game_window.get_size()
        self.shortest_dim_px = min([self.window_width_px, self.window_height_px])
        self.window_center = Vector2(
            self.window_width_px / 2, self.window_height_px / 2
        )
        self.target_ring_radius_px = (
            (self.shortest_dim_px / 2)
            - self.PADDING_AROUND_TARGET_RING_px
            - self.TARGET_RADIUS_px
        )
        # Angle between the directions to each distal target.
        angle_between_targets = (math.pi * 2) / self.num_targets
        # Simulator area info.
        self.simulator_left = self.window_width_px - self.SIMULATOR_SIZE
        self.simulator_right = self.window_width_px
        self.simulator_top = self.window_height_px - self.SIMULATOR_SIZE
        self.simulator_bottom = self.window_height_px
        self.simulator_center = Vector2(
            self.simulator_left + self.SIMULATOR_SIZE / 2,
            self.simulator_top + self.SIMULATOR_SIZE / 2,
        )

        ## Game state for drawing objects.

        # Cursor.
        self.cursor_position = self.window_center.copy()

        # Targets.
        # (One center target, and the rest equally spread in a circle around it.)
        self.target_objs = {}

        self.target_objs[self.CENTER_TARGET_ID] = {
            "id": self.CENTER_TARGET_ID,
            "position": self.window_center.copy(),
        }

        for idx in range(self.num_targets):
            target_id = f"outer_{idx}"
            target_position = self.window_center + Vector2(
                math.cos(angle_between_targets * idx) * self.target_ring_radius_px,
                math.sin(angle_between_targets * idx) * self.target_ring_radius_px,
            )
            self.target_objs[target_id] = {
                "id": target_id,
                "position": target_position,
            }

        # Electrodes
        # (Two 96-channel arrays.)
        self.electrode_objs = {}

        electrode_spacing = 25
        array_spacing = electrode_spacing * 11
        for array_idx in range(2):
            for electrode_idx in range(96 * array_idx, 96 * (array_idx + 1)):
                array_top = electrode_spacing + array_idx * array_spacing
                array_left = self.window_width_px - array_spacing
                electrode_idx_in_array = electrode_idx - 96 * array_idx
                electrode_row_in_array = electrode_idx_in_array // 10
                electrode_col_in_array = electrode_idx_in_array % 10
                electrode_position = Vector2(
                    array_left + electrode_col_in_array * electrode_spacing,
                    array_top + electrode_row_in_array * electrode_spacing,
                )
                self.electrode_objs[electrode_idx] = {
                    "electrode_idx": electrode_idx,
                    "position": electrode_position,
                }

    def run_game_loop(self):
        """
        Run the game loop at a consistent frame rate, repeatedly receiving input,
        updating state, and updating visuals to reflect that state.
        """

        # Manage the rate at which the visuals refresh.
        fps_clock = pygame.time.Clock()

        ## Run the game loop where things actually happen.

        while True:
            # If the game is quit, tear things down.
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.quit_game()

            # Get keyboard presses.
            key_input = pygame.key.get_pressed()

            # Quit the game if escape is pressed.
            if key_input[pygame.K_ESCAPE]:
                self.quit_game()

            ## Update the game state.

            self.update_game_state_based_on_inputs()
            self.update_game_state_based_on_task()

            ## Update the visuals.

            self.draw_game()

            ## Pause appropriately to maintain the specified frame rate.

            fps_clock.tick(self.FPS)

    def quit_game(self):
        """
        End the game, including closing the game window, etc.
        """
        # If this was open-loop, save the data to be used as training data. Also save
        # the neural encoding the simulator used, so it can continue to simulate data
        # according to that encoding.
        if self.is_open_loop:
            timestamps = self.open_loop_samples["timestamps"]
            firing_rates = self.open_loop_samples["firing_rates"]
            intended_velocities = self.open_loop_samples["intended_velocities"]
            preferred_directions = self.neural_simulator.preferred_directions
            block_start = int(timestamps[0])
            cache_filename = f"calibration_data_{block_start}.npz"
            cache_filepath = os.path.join(CACHE_DIR, cache_filename)
            with open(cache_filepath, "wb") as f:
                np.savez(
                    f,
                    timestamps=timestamps,
                    firing_rates=firing_rates,
                    intended_velocities=intended_velocities,
                    preferred_directions=preferred_directions,
                )

        pygame.quit()

        sys.exit()

    ####################################################################################
    # Additional public methods.
    ####################################################################################

    def add_to_cursor_movement_queue(self, x, y):
        """
        Tell the cursor to move a certain amount. This does not technically move the
        cursor, but rather prepares it to be moved, and next time the game loop gets to
        receiving inputs, it will apply this movement (along with any others on the
        queue at that time).

        This is how code outside this class should interface with this game in order to
        move the cursor.

        :param float x: Number of pixels to move in the x direction (positive to the
            right, negative to the left).
        :param float y: Number of pixels to move in the y direction (positive down,
            negative up).
        """
        self.cursor_movements_queue.put(Vector2(x, y))

    ####################################################################################
    # Internal helper methods.
    ####################################################################################

    def update_game_state_based_on_inputs(self):
        """
        Given any inputs received currently or since last game loop cycle, update the
        variables which define the state of the game (cursor position, etc.). This does
        not visually change things. These state changes will be visually reflected once
        the game objects are redrawn, which is done elsewhere.
        """

        ## Based on mouse position, set simulator state.

        mouse_pos_x, mouse_pos_y = pygame.mouse.get_pos()
        is_mouse_within_simulator = (
            self.simulator_left <= mouse_pos_x <= self.simulator_right
        ) and (self.simulator_top <= mouse_pos_y <= self.simulator_bottom)
        self.simulator_mouse_position = (
            Vector2(mouse_pos_x, mouse_pos_y) - self.simulator_center
            if is_mouse_within_simulator
            else Vector2(0, 0)
        )
        # Take the mouse position in the simulator window and scale it so that intended
        # velocity is roughly magnitude 1 near the edge of the simulator window.
        simulator_velocity = np.array(self.simulator_mouse_position) / (
            self.SIMULATOR_SIZE / 2
        )
        self.current_firing_rates = (
            self.neural_simulator.encode_velocity_as_firing_rates(simulator_velocity)
        )
        if self.is_open_loop:
            # Build up samples of firing rates and intended velocities.
            timestamp = time.time()
            firing_rates = self.current_firing_rates
            intended_velocity = [np.nan, np.nan]
            if self.target_cued:
                target_position = self.target_objs[self.target_cued]["position"]
                cursor_position = self.cursor_position
                intended_velocity_vector = target_position - cursor_position
                intended_velocity = np.array(
                    [intended_velocity_vector.x, intended_velocity_vector.y]
                )
            self.open_loop_samples["timestamps"].append(timestamp)
            self.open_loop_samples["firing_rates"].append(firing_rates)
            self.open_loop_samples["intended_velocities"].append(intended_velocity)
            # Instead of having the cursor stand still during open-loop, have it
            # automatically move toward the cued target.
            if self.target_cued:
                target_position = self.target_objs[self.target_cued]["position"]
                cursor_position = self.cursor_position
                toward_target = np.array(target_position - cursor_position)
                toward_target = toward_target / np.linalg.norm(toward_target)
                toward_target *= 3
                self.add_to_cursor_movement_queue(toward_target[0], toward_target[1])
        else:
            # Use the decoder to predict velocity based on the simulated firing rates.
            predicted_velocity = self.neural_decoder.predict(self.current_firing_rates)
            predicted_x, predicted_y = predicted_velocity / self.FPS
            self.add_to_cursor_movement_queue(predicted_x, predicted_y)

        ## Read various inputs, to get how much to move the cursor.

        cursor_delta = Vector2(0, 0)

        # Read events from `self.cursor_movements_queue`.
        # (This is how code outside this class controls the cursor, for example a BRAND
        # node running this game, controlling the cursor based on velocities it reads
        # from a redis stream).
        num_queued_entries = self.cursor_movements_queue.qsize()
        # Technically, more entries can be added while this loop happens, but to
        # guarantee no infinite loop / extra delay, only grab the amount of entries
        # present when we looked initially.
        for _ in range(num_queued_entries):
            cursor_delta += self.cursor_movements_queue.get(block=False)

        # Also accept keyboard arrow keys as input.

        key_input = pygame.key.get_pressed()
        ARROW_KEY_GAIN = 5

        if key_input[pygame.K_LEFT]:
            cursor_delta += Vector2(-1, 0) * ARROW_KEY_GAIN

        if key_input[pygame.K_RIGHT]:
            cursor_delta += Vector2(1, 0) * ARROW_KEY_GAIN

        if key_input[pygame.K_UP]:
            cursor_delta += Vector2(0, -1) * ARROW_KEY_GAIN

        if key_input[pygame.K_DOWN]:
            cursor_delta += Vector2(0, 1) * ARROW_KEY_GAIN

        ## Based on the inputs, update the cursor position state.

        # Apply the cursor movement.
        new_cursor_position = self.cursor_position + cursor_delta

        # Bound the cursor within the window walls.
        new_cursor_position.x = max([new_cursor_position.x, 0])
        new_cursor_position.x = min([new_cursor_position.x, self.window_width_px])
        new_cursor_position.y = max([new_cursor_position.y, 0])
        new_cursor_position.y = min([new_cursor_position.y, self.window_height_px])

        self.cursor_position = new_cursor_position

        ## Update the target being touched by the cursor.

        prev_target_cursor_touching = self.target_cursor_touching
        new_target_cursor_touching = None
        for target_obj in self.target_objs.values():
            cursor_target_distance_px = (
                target_obj["position"] - self.cursor_position
            ).magnitude()
            if cursor_target_distance_px < self.TARGET_TOUCHING_DISTANCE_px:
                new_target_cursor_touching = target_obj["id"]

        if prev_target_cursor_touching is None and new_target_cursor_touching:
            # The cursor wasn't touching a target, and now it is. Store the timestamp.
            self.target_last_entered_timestamp = time.time()

        self.target_cursor_touching = new_target_cursor_touching

    def update_game_state_based_on_task(self):
        """
        Given the time elapsed and the task parameters, update the variables which
        define the state of the game (cursor position, etc.). This does not visually
        change things. These state changes will be visually reflected once the game
        objects are redrawn, which is done elsewhere.
        """

        now = time.time()

        # If a trial is happening, get how long it has been going.
        trial_so_far_secs = -1
        if self.target_cued_timestamp is not None:
            trial_so_far_secs = now - self.target_cued_timestamp

        # If the cursor is touching the cued target, get how long it has been touching
        # it.
        dwell_so_far_secs = -1
        if self.target_cursor_touching == self.target_cued:
            dwell_so_far_secs = now - self.target_last_entered_timestamp

        # Check if no target has been cued.
        if self.target_cued is None:
            # Cue a new target.
            self.cue_new_target()
        # Check if the current trial has timed out.
        elif trial_so_far_secs >= self.TRIAL_TIMEOUT_sec:
            # Center the cursor on the target that was previously cued.
            prev_target = self.target_objs[self.target_cued]
            self.cursor_position = prev_target["position"].copy()
            # Cue a new target.
            self.cue_new_target()
        # Check if the cursor has dwelled on a target long enough.
        elif dwell_so_far_secs >= self.dwell_requirement_secs:
            # Center the cursor on the target that was previously cued.
            prev_target = self.target_objs[self.target_cued]
            self.cursor_position = prev_target["position"].copy()
            # Cue a new target.
            self.cue_new_target()

    def cue_new_target(self):
        """
        Set a new cued target. Alternate between the center target and a random outer
        target. The first target (when there has been no target cued yet) is a random
        outer target.
        """
        outer_target_ids = [
            target_id
            for target_id in self.target_objs
            if target_id != self.CENTER_TARGET_ID
        ]
        new_target_id = (
            self.CENTER_TARGET_ID
            if self.target_cued != self.CENTER_TARGET_ID
            and self.target_cued is not None
            else random.choice(outer_target_ids)
        )

        self.trial_idx += 1
        self.target_cued = new_target_id
        self.target_cued_timestamp = time.time()

    def draw_game(self):
        """
        Updates the visuals of the game to reflect the current game state variables.
        """

        # Draw the background.
        self.game_window.fill(DARKEST_GRAY)

        # Draw the simulator mouse area.
        simulator_line_length = 40
        simulator_line_thickness = 5
        simulator_arrow_thickness = 3
        simulator_bounds = pygame.Rect(
            self.window_width_px - self.SIMULATOR_SIZE,
            self.window_height_px - self.SIMULATOR_SIZE,
            self.SIMULATOR_SIZE,
            self.SIMULATOR_SIZE,
        )
        pygame.draw.rect(self.game_window, WHITE, simulator_bounds)
        pygame.draw.line(
            self.game_window,
            DARKER_GRAY,
            Vector2(
                self.simulator_center.x, self.simulator_center.y - simulator_line_length
            ),
            Vector2(
                self.simulator_center.x, self.simulator_center.y + simulator_line_length
            ),
            simulator_line_thickness,
        )
        pygame.draw.line(
            self.game_window,
            DARKER_GRAY,
            Vector2(
                self.simulator_center.x - simulator_line_length, self.simulator_center.y
            ),
            Vector2(
                self.simulator_center.x + simulator_line_length, self.simulator_center.y
            ),
            simulator_line_thickness,
        )
        if self.simulator_mouse_position is not None:
            pygame.draw.line(
                self.game_window,
                BLACK,
                self.simulator_center,
                self.simulator_center + self.simulator_mouse_position,
                simulator_arrow_thickness,
            )

        # Draw the simulated electrodes.
        electrode_radius = 10
        for electrode_obj in self.electrode_objs.values():
            electrode_idx = electrode_obj["electrode_idx"]
            electrode_firing_rate = self.current_firing_rates[electrode_idx]
            electrode_color = self.get_electrode_color(electrode_firing_rate)
            pygame.draw.circle(
                self.game_window,
                electrode_color,
                electrode_obj["position"],
                electrode_radius,
            )

        # Draw the targets.
        for target_obj in self.target_objs.values():
            target_color = self.get_target_color(target_obj["id"])
            pygame.draw.circle(
                self.game_window,
                target_color,
                target_obj["position"],
                self.TARGET_RADIUS_px,
            )

        # Draw the cursor.
        pygame.draw.circle(
            self.game_window, WHITE, self.cursor_position, self.CURSOR_RADIUS_px
        )

        # Register the updates visually on the screen.
        pygame.display.update()

        # Maintain frame information.
        self.frame_update_timestamp = time.time()
        self.frame_idx += 1

    def get_target_color(self, target_id):
        """
        Get the color of a target, based on if it is cued, if the cursor is touching it,
        etc.

        :param str target_id: Id of the target we are getting the color for.

        :return (int, int, int): RGB tuple of values 0 to 255
        """

        # Default to light gray, or red if cued.
        target_color = RED if target_id == self.target_cued else LIGHT_GRAY

        # But if the cursor if touching this target, change its color.
        if target_id == self.target_cursor_touching:
            # Red becomes light red. Light gray becomes dark gray.
            target_color = LIGHT_RED if target_id == self.target_cued else DARK_GRAY

        return target_color

    def get_electrode_color(self, firing_rate):
        """
        TODO: doc this
        """
        # The more firing, the more red. The less firing, the more white (with some
        # blue).
        max_coloration_firing_rate = (
            self.neural_simulator.FIRING_RATE_SCALE
            + self.neural_simulator.FIRING_RATE_NOISE_STDDEV
        ) * 1.2
        red_portion = min([firing_rate / max_coloration_firing_rate, 1])
        electrode_color = np.array(RED) * red_portion + np.array(LIGHT_BLUE) * (
            1 - red_portion
        )

        return electrode_color


if __name__ == "__main__":
    game = Targets2dGame(
        dwell_requirement_secs=0,
        # calibration_filename="calibration_data_1678676399.npz",
    )
    game.init_game()
    game.run_game_loop()
