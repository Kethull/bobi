# environment.py
import gym
from gym import spaces
import numpy as np
import random
import math # Added for math.radians
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from config import config, ConfigurationError # Import the global config instance
from solarsystem import CelestialBody, OrbitalMechanics
from physics_utils import PhysicsError, safe_divide # Potentially useful for handling errors from orbital_mechanics

# Keep existing Message, Resource, SpaceEnvironment classes if they are still used as a base
# For brevity, assuming they are the same as before unless specified.
# If SolarSystemEnvironment is now the primary environment, SpaceEnvironment might be simplified or removed later.

@dataclass
class Message:
    """Represents a message for inter-probe communication.

    Messages can convey information such as resource locations, warnings, or
    other data relevant to cooperative or competitive probe behavior. The structure
    and types of messages are influenced by `config.Probe.MESSAGE_TYPES`.

    Attributes:
        sender_id (int): Unique ID of the probe that originated the message.
        msg_type (str): Type of the message (e.g., 'RESOURCE_LOCATION',
            'ALERT'). Must be one of the types defined in
            `config.Probe.MESSAGE_TYPES`.
        position (Tuple[float, float]): (x, y) coordinates in simulation units
            relevant to the message content (e.g., location of a discovered
            resource, or a point of interest).
        resource_amount (float): If the message pertains to a resource, this
            field indicates the amount of that resource. For other message
            types, this might be 0 or unused.
        timestamp (int): The simulation step (tick) at which this message
            was created. Useful for determining message age and relevance.
    """
    sender_id: int
    msg_type: str
    position: Tuple[float, float]
    resource_amount: float
    timestamp: int

@dataclass
class Resource:
    """Represents a harvestable resource node in the simulation.

    Resource nodes are defined by their location, current amount, and maximum
    capacity. Probes can harvest from these nodes, and resources may regenerate
    over time based on `config.Resource.REGEN_RATE_PER_STEP`.

    Attributes:
        position (Tuple[float, float]): The (x, y) coordinates of the resource
            node in the simulation's world units.
        amount (float): The current quantity of resource available at this node.
            This value decreases when harvested and can increase via regeneration,
            up to `max_amount`.
        max_amount (float): The maximum quantity of resource this node can hold.
            Regeneration will not cause the `amount` to exceed this value.
    """
    position: Tuple[float, float]
    amount: float
    max_amount: float
    
    def harvest(self, rate: float) -> float:
        """Harvests resources from this node.

        Reduces the `amount` of resource at this node by the specified `rate`,
        but not below zero. The actual amount harvested is returned.

        Args:
            rate (float): The amount of resource to attempt to harvest.
                If negative, 0.0 is harvested and a warning is logged.

        Returns:
            float: The actual amount of resource harvested, which is
                   `min(rate, self.amount)` if `rate` is non-negative,
                   otherwise 0.0.
        """
        if rate < 0:
            logging.warning(f"Attempted to harvest with a negative rate: {rate}. Harvesting 0.0.")
            return 0.0
        
        harvested_amount = min(rate, self.amount)
        self.amount -= harvested_amount
        # self.amount = max(0.0, self.amount - harvested_amount) # Ensure amount doesn't go negative if min logic fails
        return harvested_amount
    
    def regenerate(self):
        """Regenerates resources at this node.

        Increases the `amount` of resource by `config.Resource.REGEN_RATE_PER_STEP`,
        ensuring it does not exceed `self.max_amount`. This method is typically
        called once per simulation step for each active resource node.

        Relies on:
            - `config.Resource.REGEN_RATE_PER_STEP`
        """
        try:
            regen_amount = config.Resource.REGEN_RATE_PER_STEP
            if regen_amount < 0: # Defensive check
                logging.warning(f"Configured REGEN_RATE_PER_STEP ({regen_amount}) is negative. No regeneration will occur.")
                return
            self.amount = min(self.max_amount, self.amount + regen_amount)
        except AttributeError as e:
            logging.error(f"Error during resource regeneration: Missing config.Resource.REGEN_RATE_PER_STEP. {e}", exc_info=True)
            # No regeneration if config is missing
        except Exception as e:
            logging.error(f"Unexpected error during resource regeneration: {e}", exc_info=True)


class SpaceEnvironment(gym.Env):
    """Base class for a 2D space simulation, compatible with OpenAI Gym.

    Provides fundamental functionalities for a generic space simulation:
    - Manages probes, resources, and messages within a 2D world.
    - Handles basic simulation stepping, observation generation, and resetting.
    - Defines world boundaries, `observation_space`, and `action_space` based
      on the global `config` object.

    Subclasses like `SolarSystemEnvironment` are expected to implement specific
    physics (e.g., orbital mechanics), detailed probe behaviors, and complex
    interactions.

    Key Responsibilities:
    -   Defining world boundaries (`config.World`).
    -   Managing collections of `Resource` objects, `Probe` data (dictionaries),
        and `Message` objects.
    -   Providing a `step()` method structure for advancing simulation time.
    -   Defining `observation_space` and `action_space` (`config.RL`).
    -   Handling probe addition (`add_probe`) and basic observation retrieval
        (`get_observation`).
    -   Managing probe lifecycle (e.g., removal of 'dead' probes).

    Error Handling:
        Initialization and core methods include `try-except` blocks to catch
        `ConfigurationError` (if `config` values are invalid/missing) and other
        `Exception` types. Critical errors are logged and may be re-raised.
        Methods like `step()` and `get_observation()` aim for robustness,
        attempting to prevent individual probe failures from crashing the simulation
        by returning default or error-state values for the affected probe.

    Attributes:
        world_width (float): Width of the simulation world in simulation units,
            from `config.World.WIDTH_SIM`.
        world_height (float): Height of the simulation world in simulation units,
            from `config.World.HEIGHT_SIM`.
        resources (List[Resource]): List of `Resource` instances.
        probes (Dict[int, Dict]): Dictionary storing data for each active probe,
            keyed by unique probe ID. Probe data includes 'position', 'velocity',
            'energy', 'alive' status, etc. (structure defined by `add_probe`).
        messages (List[Message]): List of active `Message` instances.
        step_count (int): Current simulation step number within an episode.
        max_probe_id (int): Highest probe ID issued, for generating new unique IDs.
        total_resources_mined (float): Cumulative resources mined by all probes
            in the current episode.
        observation_space (gym.spaces.Box): Defines the structure and bounds of
            observations for agents (`config.RL.OBSERVATION_SPACE_SIZE`).
        action_space (gym.spaces.MultiDiscrete): Defines the structure of actions
            agents can take (`config.RL.ACTION_SPACE_DIMS`).
    """
    def __init__(self):
        """Initializes the `SpaceEnvironment`.

        Sets up:
        - World dimensions from `config.World`.
        - Empty collections for `resources`, `probes`, and `messages`.
        - `observation_space` and `action_space` from `config.RL`.
        - Calls `_generate_resources()` to populate initial resources.

        Raises:
            ConfigurationError: If essential configuration values (e.g., for world
                dimensions, observation/action space sizes, resource generation)
                are missing or invalid.
            Exception: For other unexpected errors during initialization.
        """
        super().__init__()
        try:
            # World dimensions
            self.world_width = float(config.World.WIDTH_SIM)
            self.world_height = float(config.World.HEIGHT_SIM)
            if self.world_width <= 0 or self.world_height <= 0:
                raise ConfigurationError("World dimensions (WIDTH_SIM, HEIGHT_SIM) must be positive.")

            # Entity collections and state
            self.resources: List[Resource] = []
            self.probes: Dict[int, Dict] = {} # Stores probe data dictionaries
            self.messages: List[Message] = []
            self.step_count: int = 0
            self.max_probe_id: int = 0 # Tracks the highest ID used
            self.total_resources_mined: float = 0.0
            
            self._generate_resources() # Can raise ConfigurationError
            
            # Gym spaces
            obs_size = config.RL.OBSERVATION_SPACE_SIZE
            if not isinstance(obs_size, int) or obs_size <= 0:
                raise ConfigurationError("RL.OBSERVATION_SPACE_SIZE must be a positive integer.")
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_size,),
                dtype=np.float32
            )
            
            action_dims = config.RL.ACTION_SPACE_DIMS
            if not isinstance(action_dims, list) or not all(isinstance(d, int) and d > 0 for d in action_dims):
                raise ConfigurationError("RL.ACTION_SPACE_DIMS must be a list of positive integers.")
            self.action_space = spaces.MultiDiscrete(action_dims)
            
            logging.info(f"SpaceEnvironment initialized with world size: {self.world_width}x{self.world_height}, "
                         f"Obs Space: {self.observation_space.shape}, Action Space: {self.action_space.nvec}")

        except AttributeError as e_attr: # Handles missing config attributes
            logging.critical(f"SpaceEnvironment initialization failed due to missing config attribute: {e_attr}", exc_info=True)
            raise ConfigurationError(f"Missing configuration attribute for SpaceEnvironment: {e_attr}")
        except ConfigurationError as e_config: # Handles explicit ConfigurationErrors
            logging.critical(f"SpaceEnvironment initialization failed due to ConfigurationError: {e_config}", exc_info=True)
            raise
        except Exception as e_unexpected: # Catch-all for other init errors
            logging.critical(f"Unexpected error during SpaceEnvironment initialization: {e_unexpected}", exc_info=True)
            raise # Re-raise as a critical failure

    def _generate_resources(self):
        """Generates and places resource nodes in the environment.

        Populates `self.resources` based on `config.Resource` settings:
        - `COUNT`: Number of resource nodes.
        - `MIN_AMOUNT`, `MAX_AMOUNT`: Range for initial and max resource quantity.
        Positions are randomly distributed within world boundaries.

        Raises:
            ConfigurationError: If essential resource configuration (e.g., `COUNT`,
                `MIN_AMOUNT`, `MAX_AMOUNT`) is missing or invalid.
            Exception: For other unexpected errors, resulting in an empty
                       `self.resources` list and a warning.
        """
        try:
            self.resources = [] # Clear previous resources
            num_resources = config.Resource.COUNT
            min_r_amount = config.Resource.MIN_AMOUNT
            max_r_amount = config.Resource.MAX_AMOUNT

            if not isinstance(num_resources, int) or num_resources < 0:
                raise ConfigurationError("config.Resource.COUNT must be a non-negative integer.")
            if not (isinstance(min_r_amount, (int, float)) and isinstance(max_r_amount, (int, float)) and min_r_amount <= max_r_amount and min_r_amount >=0):
                raise ConfigurationError("config.Resource MIN_AMOUNT/MAX_AMOUNT must be valid non-negative numbers with MIN <= MAX.")

            if num_resources == 0:
                logging.info("config.Resource.COUNT is 0. No resources will be generated.")
                return

            for _ in range(num_resources):
                pos_x = random.uniform(0, self.world_width)
                pos_y = random.uniform(0, self.world_height)
                # Amount is both initial and max for simplicity here; could be separate
                initial_amount = random.uniform(min_r_amount, max_r_amount)
                self.resources.append(Resource(position=(pos_x, pos_y), amount=initial_amount, max_amount=initial_amount))
            
            logging.info(f"Generated {len(self.resources)} resource nodes.")

        except AttributeError as e_attr:
            logging.error(f"Error generating resources: Missing attribute in config.Resource: {e_attr}", exc_info=True)
            raise ConfigurationError(f"Failed to generate resources due to missing config.Resource attribute: {e_attr}")
        except ConfigurationError as e_config: # Re-raise if we raised it
            raise
        except Exception as e_unexpected:
            logging.error(f"Unexpected error generating resources: {e_unexpected}", exc_info=True)
            self.resources = [] # Ensure resources list is empty on failure
            logging.warning("Resource generation failed. Environment will have no resources.")


    def add_probe(self, probe_id: int, position: Tuple[float, float],
                  energy: float = -1.0, generation: int = 0):
        """Adds a new probe to the simulation.

        Initializes a data dictionary for the probe in `self.probes`.
        This dictionary includes:
        - `position`, `velocity` (initially zero), `angle_rad` (random).
        - `energy` (defaults to `config.Probe.INITIAL_ENERGY` if `energy` < 0).
        - `alive` status (True), `trail` list, `max_trail_points`.
        - `generation` number.
        - State variables for actions: `current_thrust_level_idx`,
          `current_torque_level_idx`, ramp ratios, steps in current action,
          `last_action_timestamp`, `target_resource_idx`, `is_mining`,
          `time_since_last_target_switch`, `last_thrust_application_step`.

        Updates `self.max_probe_id` if `probe_id` is higher.

        Args:
            probe_id (int): Unique identifier for the new probe.
            position (Tuple[float, float]): Initial (x, y) coordinates (sim units).
            energy (float, optional): Initial energy. Defaults to
                `config.Probe.INITIAL_ENERGY` if negative.
            generation (int, optional): Generation number. Defaults to 0.

        Error Handling:
            Logs an error and avoids adding the probe if essential configurations
            (e.g., `config.Probe.INITIAL_ENERGY`,
            `config.Visualization.MAX_PROBE_TRAIL_POINTS`,
            `config.Probe.SWITCHING_DETECTION_WINDOW_STEPS`,
            `config.RL.TARGET_SWITCH_COOLDOWN_STEPS`) are missing (AttributeError),
            or if other unexpected errors occur.
        """
        try:
            # Default values from config, checked for existence
            initial_probe_energy = config.Probe.INITIAL_ENERGY
            max_trail_cfg = config.Visualization.MAX_PROBE_TRAIL_POINTS
            switching_window_cfg = config.Probe.SWITCHING_DETECTION_WINDOW_STEPS
            target_cooldown_cfg = config.RL.TARGET_SWITCH_COOLDOWN_STEPS

            # Validate inputs and config values
            if not (isinstance(initial_probe_energy, (int, float)) and initial_probe_energy >= 0):
                raise ConfigurationError("config.Probe.INITIAL_ENERGY must be a non-negative number.")
            if not (isinstance(max_trail_cfg, int) and max_trail_cfg > 0):
                raise ConfigurationError("config.Visualization.MAX_PROBE_TRAIL_POINTS must be a positive integer.")
            if not (isinstance(switching_window_cfg, int) and switching_window_cfg >= 0):
                raise ConfigurationError("config.Probe.SWITCHING_DETECTION_WINDOW_STEPS must be a non-negative integer.")
            if not (isinstance(target_cooldown_cfg, int) and target_cooldown_cfg >= 0):
                raise ConfigurationError("config.RL.TARGET_SWITCH_COOLDOWN_STEPS must be a non-negative integer.")

            effective_initial_energy = energy if energy >= 0 else initial_probe_energy

            self.probes[probe_id] = {
                'position': np.array(position, dtype=np.float32),
                'velocity': np.array([0.0, 0.0], dtype=np.float32), # km/s or sim_units/s
                'angle_rad': random.uniform(0, 2 * np.pi),
                'angular_velocity_rad_per_step': 0.0, # rad / (sim_step_duration_seconds)
                'energy': float(effective_initial_energy),
                'alive': True,
                'trail': [], # List of past screen positions for rendering
                'max_trail_points': max_trail_cfg,
                'generation': int(generation),
                # Action state variables
                'current_thrust_level_idx': 0, # Index in config.Probe.THRUST_FORCE_MAGNITUDES
                'current_torque_level_idx': 0, # Index in config.Probe.TORQUE_MAGNITUDES (0 = no torque)
                'thrust_ramp_ratio': 0.0,      # Current ramp progress (0 to 1)
                'rotation_ramp_ratio': 0.0,    # Current ramp progress (0 to 1)
                'steps_in_current_thrust': 0,  # Duration of current thrust application
                'steps_in_current_rotation': 0,# Duration of current rotation application
                'last_action_timestamp': - (switching_window_cfg + 1), # Allow immediate first action
                'target_resource_idx': None,   # Index of targeted resource in self.resources
                'is_mining': False,
                'time_since_last_target_switch': target_cooldown_cfg + 1, # Allow immediate first target
                'last_thrust_application_step': -1, # For thruster startup cost logic
                'last_rotation_application_step': -1 # For potential rotational startup cost
            }
            self.max_probe_id = max(self.max_probe_id, probe_id)
            logging.debug(f"Added probe {probe_id} at {position} with energy {effective_initial_energy}, gen {generation}.")
        
        except AttributeError as e_attr: # Missing config
            logging.error(f"Error adding probe {probe_id}: Missing attribute in config: {e_attr}", exc_info=True)
            # Do not add the probe if config is critically missing
        except ConfigurationError as e_config: # Our explicit config error
            logging.error(f"Error adding probe {probe_id} due to configuration issue: {e_config}", exc_info=True)
        except Exception as e_unexpected:
            logging.error(f"Unexpected error adding probe {probe_id}: {e_unexpected}", exc_info=True)
            # Do not add the probe on unexpected error


    def get_observation(self, probe_id: int) -> np.ndarray:
        """Generates an observation vector for a specified probe.

        This base implementation returns a zero vector of size
        `config.RL.OBSERVATION_SPACE_SIZE`. Subclasses (like
        `SolarSystemEnvironment`) must override this to provide meaningful,
        state-dependent observations (e.g., probe's state, sensor data about
        celestial bodies, resources, other probes).

        Args:
            probe_id (int): ID of the probe for which to get the observation.

        Returns:
            np.ndarray: Observation vector (`float32`). Returns a zero vector
                of configured size if the probe doesn't exist or is not alive.

        Error Handling:
            - Logs a warning and returns a default zero observation if the probe
              is non-existent or not alive.
            - Logs an error and returns a fallback zero observation (size 10 if
              config is critically broken, else configured size) if
              `config.RL.OBSERVATION_SPACE_SIZE` is missing or invalid, or if
              other unexpected errors occur. This aims to prevent crashes in
              the agent's prediction step.
        """
        try:
            obs_size = config.RL.OBSERVATION_SPACE_SIZE
            if not isinstance(obs_size, int) or obs_size <= 0:
                # This should have been caught in __init__, but defensive check here
                logging.error(f"Invalid config.RL.OBSERVATION_SPACE_SIZE ({obs_size}) in get_observation. Using fallback size 10.")
                obs_size = 10

            if probe_id not in self.probes or not self.probes[probe_id].get('alive', False):
                if probe_id not in self.probes:
                    logging.warning(f"get_observation: Probe {probe_id} not found. Returning zero vector of size {obs_size}.")
                else: # Probe exists but is not alive
                    logging.debug(f"get_observation: Probe {probe_id} is not alive. Returning zero vector.")
                return np.zeros(obs_size, dtype=np.float32)
            
            # --- Placeholder for actual observation generation ---
            # Subclasses (SolarSystemEnvironment) will populate this with meaningful data.
            # Example structure (must match OBSERATION_SPACE_SIZE):
            # probe_data = self.probes[probe_id]
            # obs_list = [
            #     probe_data['position'][0] / self.world_width, # Normalized X
            #     probe_data['position'][1] / self.world_height, # Normalized Y
            #     probe_data['velocity'][0] / config.Probe.MAX_VELOCITY_SIM_PER_STEP, # Normalized Vx
            #     probe_data['velocity'][1] / config.Probe.MAX_VELOCITY_SIM_PER_STEP, # Normalized Vy
            #     probe_data['angle_rad'] / (2 * np.pi), # Normalized angle (0-1)
            #     probe_data['angular_velocity_rad_per_step'] / config.Probe.MAX_ANGULAR_VELOCITY_RAD_PER_STEP,
            #     probe_data['energy'] / config.Probe.MAX_ENERGY,
            #     # Sensor data for N nearest resources (distance, angle, amount)
            #     # Sensor data for N nearest celestial bodies (distance, angle, mass_influence)
            #     # Sensor data for N nearest other probes (distance, angle, threat_level?)
            # ]
            # # Ensure obs_list is padded/truncated to obs_size
            # current_obs_data = np.array(obs_list[:obs_size], dtype=np.float32)
            # if len(current_obs_data) < obs_size:
            #     current_obs_data = np.pad(current_obs_data, (0, obs_size - len(current_obs_data)), 'constant', constant_values=0)
            # return current_obs_data
            # --- End Placeholder ---
            
            # Base class returns zeros
            return np.zeros(obs_size, dtype=np.float32)

        except AttributeError as e_attr: # Missing config.RL.OBSERVATION_SPACE_SIZE
            logging.error(f"Error getting observation for probe {probe_id} due to missing config attribute (likely RL.OBSERVATION_SPACE_SIZE): {e_attr}", exc_info=True)
            return np.zeros(10, dtype=np.float32) # Fallback for critical config error
        except Exception as e_unexpected:
            logging.error(f"Unexpected error getting observation for probe {probe_id}: {e_unexpected}", exc_info=True)
            # Attempt to return correctly sized zeros if possible, else fallback
            fallback_obs_size = 10
            try:
                fallback_obs_size = config.RL.OBSERVATION_SPACE_SIZE if (isinstance(config.RL.OBSERVATION_SPACE_SIZE, int) and config.RL.OBSERVATION_SPACE_SIZE > 0) else 10
            except AttributeError: pass # Keep fallback_obs_size = 10
            return np.zeros(fallback_obs_size, dtype=np.float32)


    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[int, Dict]]:
        """Advances the simulation by one time step based on agent actions.

        This base method iterates through active probes, applying placeholder
        logic for physics and action effects. Subclasses (`SolarSystemEnvironment`)
        are responsible for implementing detailed physics (gravity, thrust,
        rotation), resource interactions, energy management, and reward calculation.

        Core responsibilities handled here or delegated to subclasses:
        1.  Iterate through probes active at the start of the step.
        2.  For each probe:
            a.  Retrieve its current data.
            b.  (Subclass) Apply physics updates based on its action from `actions`.
            c.  (Subclass) Update probe state (position, velocity, energy, etc.).
            d.  Update the probe's visual trail data.
            e.  (Subclass) Handle energy decay and check for death conditions.
            f.  Generate a new observation (via `get_observation()`).
            g.  (Subclass) Calculate reward.
            h.  Determine 'done' status (e.g., if max steps reached, probe died, or
                an error occurred).
            i.  Collect informational dictionary.
        3.  Remove probes marked as not 'alive' AND 'done'.
        4.  Increment `self.step_count`.

        Args:
            actions (Dict[int, np.ndarray]): Maps probe IDs to their selected
                actions for this step. Action structure from `self.action_space`.

        Returns:
            Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[int, Dict]]:
            Dictionaries (keyed by probe ID) for new observations, rewards,
            'done' flags, and info.

        Error Handling:
            - If an error (e.g., `KeyError`, `Exception`) occurs while processing an
              individual probe, it's logged. That probe is marked as 'done', and
              default/error-state values (zero observation, zero reward) are
              returned for it. The simulation attempts to continue with other probes.
            - If a critical `Exception` occurs in the main step loop (outside
              individual probe processing), it's logged. The method then attempts
              to return default 'done' states and zero rewards for all probes
              that had actions submitted, to prevent crashes in the calling
              simulation manager.
        """
        observations_next: Dict[int, np.ndarray] = {}
        rewards_this_step: Dict[int, float] = {}
        dones_this_step: Dict[int, bool] = {}
        infos_this_step: Dict[int, Dict] = {}
        
        try:
            # Probes active at the beginning of this step
            # Iterate over a copy of keys in case self.probes is modified (e.g., replication)
            probe_ids_at_step_start = list(self.probes.keys())

            for probe_id in probe_ids_at_step_start:
                probe_data = self.probes.get(probe_id) # Get current data

                # If probe disappeared unexpectedly (e.g., removed by another process not reflected yet)
                if not probe_data:
                    logging.warning(f"Probe {probe_id} was in active list but data not found in step {self.step_count}.")
                    if probe_id in actions: # If an action was submitted for it
                         observations_next[probe_id] = self.get_observation(probe_id) # Default obs
                         rewards_this_step[probe_id] = 0.0
                         dones_this_step[probe_id] = True # Mark as done
                         infos_this_step[probe_id] = {'error': 'Probe data disappeared mid-step'}
                    continue

                # Initialize return dicts for this probe for this step
                # Observation will be updated after actions are processed by subclass
                observations_next[probe_id] = self.get_observation(probe_id) # Current obs before action
                rewards_this_step[probe_id] = 0.0 # Default reward, subclass should modify
                dones_this_step[probe_id] = False # Default done, subclass should modify
                infos_this_step[probe_id] = {}    # Default info, subclass can add to it

                if probe_data.get('alive', False): # Process only alive probes
                    try:
                        # --- Subclass Responsibility: Apply Actions & Physics ---
                        # This is where SolarSystemEnvironment would apply thrust, gravity,
                        # handle mining, replication, communication, update energy, etc.,
                        # and calculate specific rewards.
                        # The base SpaceEnvironment does minimal or no physics updates itself.
                        
                        # --- Update Probe Trail (Generic part) ---
                        if 'trail' in probe_data and 'position' in probe_data:
                            # Ensure max_trail_points is valid
                            max_trail_cfg = probe_data.get('max_trail_points', config.Visualization.MAX_PROBE_TRAIL_POINTS)
                            if not (isinstance(max_trail_cfg, int) and max_trail_cfg > 0): max_trail_cfg = 50 # Fallback
                            
                            probe_data['trail'].append(probe_data['position'].copy()) # Add current position
                            while len(probe_data['trail']) > max_trail_cfg:
                                probe_data['trail'].pop(0)
                        
                        # --- Check Max Episode Length (Generic part) ---
                        if self.step_count >= config.RL.EPISODE_LENGTH_STEPS -1 : # -1 because step_count increments after this loop
                            dones_this_step[probe_id] = True
                            infos_this_step[probe_id]['reason_done'] = 'max_episode_steps'
                        
                        # --- Subclass Responsibility: Check other 'done' conditions ---
                        # e.g., if probe_data['energy'] <= 0, set probe_data['alive'] = False, dones_this_step[probe_id] = True

                        # --- Update Observation (Generic part, calls subclass's get_observation) ---
                        # This should be called *after* all state changes for the step are complete.
                        observations_next[probe_id] = self.get_observation(probe_id)

                    except KeyError as e_key: # Error in accessing probe_data keys
                        logging.error(f"KeyError processing probe {probe_id} in step {self.step_count}: {e_key}. Probe data: {probe_data}", exc_info=True)
                        if probe_id in self.probes: self.probes[probe_id]['alive'] = False # Mark as dead on error
                        dones_this_step[probe_id] = True
                        infos_this_step[probe_id]['error'] = f"KeyError: {str(e_key)}"
                    except Exception as e_probe_step: # Other errors during individual probe processing
                        logging.error(f"Unexpected error processing probe {probe_id} in step {self.step_count}: {e_probe_step}", exc_info=True)
                        if probe_id in self.probes: self.probes[probe_id]['alive'] = False # Mark as dead
                        dones_this_step[probe_id] = True
                        infos_this_step[probe_id]['error'] = f"Exception: {str(e_probe_step)}"
                
                # If probe was marked not alive (either before this step or during processing)
                if probe_data and not probe_data.get('alive', True): # Check probe_data exists
                    dones_this_step[probe_id] = True # Ensure 'done' is true if not alive
                    if 'reason_done' not in infos_this_step[probe_id]: # Don't overwrite specific reason
                         infos_this_step[probe_id]['reason_done'] = 'probe_not_alive'


            # --- Remove probes that are now marked as not alive AND are 'done' ---
            # This ensures that 'done' probes are cleaned up at the end of the step
            # where their 'done' status was finalized.
            probes_to_remove_ids = []
            for pid_remove_check, pdata_remove_check in self.probes.items():
                # If probe is not alive AND its 'done' flag for this step is True
                if not pdata_remove_check.get('alive', True) and dones_this_step.get(pid_remove_check, False):
                    probes_to_remove_ids.append(pid_remove_check)
            
            for pid_remove in probes_to_remove_ids:
                try:
                    if config.Debug.DEBUG_MODE: # Log removal only in debug mode
                        logging.debug(f"Step {self.step_count}: Removing probe {pid_remove} (marked not alive and done).")
                    del self.probes[pid_remove]
                except KeyError: # Should not happen if iterating self.probes.items()
                    logging.warning(f"Attempted to remove probe {pid_remove} in step {self.step_count}, but it was already gone (KeyError).")
                except Exception as e_remove:
                    logging.error(f"Error removing probe {pid_remove} during step {self.step_count} cleanup: {e_remove}", exc_info=True)

            self.step_count += 1 # Increment global step counter

        except Exception as e_main_step: # Catch-all for errors in the main step loop itself
            logging.critical(f"Critical error in SpaceEnvironment.step main loop at step {self.step_count}: {e_main_step}", exc_info=True)
            # Fallback: Mark all probes that had actions submitted as 'done' to prevent agent crashes
            for pid_error_fallback in actions.keys(): # Iterate over keys from the input 'actions' dict
                observations_next[pid_error_fallback] = self.get_observation(pid_error_fallback) # Attempt to get last valid obs
                rewards_this_step[pid_error_fallback] = 0.0
                dones_this_step[pid_error_fallback] = True # Mark as done due to critical environment error
                infos_this_step[pid_error_fallback] = {'critical_environment_error': str(e_main_step)}
        
        return observations_next, rewards_this_step, dones_this_step, infos_this_step

    def reset(self) -> Dict[int, np.ndarray]:
        """Resets the environment to an initial state for a new episode.

        This involves:
        1.  Resetting `step_count` to 0.
        2.  Clearing `probes` and `messages`.
        3.  Resetting `max_probe_id` and `total_resources_mined`.
        4.  Calling `_generate_resources()` to repopulate resource nodes.
        5.  Adding initial probes (count from `config.Probe.INITIAL_PROBES`)
            at random positions near the world center, with default energy
            and generation 0. New unique IDs are assigned.
        6.  Generating and returning initial observations for these new probes.

        Returns:
            Dict[int, np.ndarray]: Maps new probe IDs to their initial
            observation vectors. Returns an empty dict if probe creation fails.

        Raises:
            ConfigurationError: If reset fails due to invalid/missing config
                (e.g., for initial probes, resources, world dimensions).
            Exception: For other unexpected critical errors during reset.
                These are logged and re-raised or result in an empty
                observation dictionary.
        """
        try:
            logging.info(f"Resetting {self.__class__.__name__} environment for new episode...")
            self.step_count = 0
            self.probes.clear() # Clear all probes from previous episode
            self.messages.clear() # Clear all messages
            self.max_probe_id = 0 # Reset for new episode IDs (e.g., starting from 1)
            self.total_resources_mined = 0.0
            
            self._generate_resources() # Can raise ConfigurationError
            
            initial_observations_dict: Dict[int, np.ndarray] = {}
            
            # Validate config for initial probes
            num_initial_probes_cfg = config.Probe.INITIAL_PROBES
            if not (isinstance(num_initial_probes_cfg, int) and num_initial_probes_cfg >= 0):
                raise ConfigurationError("config.Probe.INITIAL_PROBES must be a non-negative integer.")

            for i in range(num_initial_probes_cfg):
                try:
                    # Ensure world_width/height are available (should be from __init__)
                    if not (hasattr(self, 'world_width') and self.world_width > 0 and
                            hasattr(self, 'world_height') and self.world_height > 0):
                        raise ConfigurationError("World dimensions (world_width, world_height) not properly initialized or invalid.")
                    
                    # Place initial probes near the center of the world
                    center_x, center_y = self.world_width / 2, self.world_height / 2
                    spread_factor = 0.1 # e.g., place within 10% of world size around center
                    start_pos_x = random.uniform(center_x - self.world_width * spread_factor,
                                                 center_x + self.world_width * spread_factor)
                    start_pos_y = random.uniform(center_y - self.world_height * spread_factor,
                                                 center_y + self.world_height * spread_factor)
                    
                    new_probe_id = self.max_probe_id + 1 # Generate new unique ID for this episode
                    # add_probe updates self.max_probe_id internally
                    self.add_probe(probe_id=new_probe_id,
                                   position=(start_pos_x, start_pos_y),
                                   energy=-1.0, # Use default from config
                                   generation=0)
                    
                    if new_probe_id in self.probes: # Check if probe was successfully added
                        initial_observations_dict[new_probe_id] = self.get_observation(new_probe_id)
                    else:
                        logging.error(f"Probe with ID {new_probe_id} was not added successfully during reset. Skipping observation.")

                except ConfigurationError as e_add_probe_config: # Catch config errors from add_probe or here
                    logging.error(f"ConfigurationError adding initial probe index {i} (attempted ID {self.max_probe_id + 1}) during reset: {e_add_probe_config}", exc_info=True)
                    # Depending on severity, might re-raise or just skip this probe
                except Exception as e_add_probe: # Catch other errors from add_probe
                    logging.error(f"Failed to add initial probe index {i} (attempted ID {self.max_probe_id + 1}) during reset: {e_add_probe}", exc_info=True)
                    # Continue to try adding other initial probes if one fails

            logging.info(f"{self.__class__.__name__} reset complete. {len(self.probes)} initial probes added. "
                         f"Returning {len(initial_observations_dict)} observations.")
            return initial_observations_dict

        except ConfigurationError as e_config_reset: # From _generate_resources or config access here
            logging.critical(f"Cannot reset {self.__class__.__name__} due to ConfigurationError: {e_config_reset}", exc_info=True)
            raise # Re-raise critical config error
        except AttributeError as e_attr_reset: # Missing critical config attributes
            logging.critical(f"Cannot reset {self.__class__.__name__} due to missing config attribute: {e_attr_reset}", exc_info=True)
            raise ConfigurationError(f"Missing config attribute during reset: {e_attr_reset}")
        except Exception as e_reset_unexpected: # Catch-all for other critical errors
            logging.critical(f"Unexpected critical error during {self.__class__.__name__} reset: {e_reset_unexpected}", exc_info=True)
            # Re-raise to ensure simulation manager knows reset failed, or return empty dict if trying to be resilient
            raise Exception(f"Critical failure during {self.__class__.__name__} reset: {e_reset_unexpected}")


# Your new SolarSystemEnvironment class:
class SolarSystemEnvironment(SpaceEnvironment):
    """Extends `SpaceEnvironment` to simulate a solar system with orbital mechanics.

    Manages celestial bodies (Sun, planets, moons) and their gravitational
    interactions, as defined in `config.SolarSystem.PLANET_DATA`. Probes operate
    within this dynamic gravitational field. Uses `OrbitalMechanics` for
    celestial body physics and `CelestialBody` to represent them. Probe physics
    (thrust, rotation) and interactions (mining, replication, communication)
    are handled here, along with detailed reward calculation and comprehensive
    observations.

    Key Enhancements:
    - Initializes and manages `CelestialBody` objects (Sun, planets, moons).
    - Uses `OrbitalMechanics` to propagate celestial body orbits (Verlet integration).
    - Applies gravitational forces from celestial bodies to probes.
    - Handles detailed probe actions: thrust, torque, mining, replication, communication.
    - Calculates specific rewards based on probe actions and outcomes.
    - Generates rich observations: probe state, nearby celestial bodies, resources.
    - Optionally monitors total system energy conservation (`config.Debug`).

    Attributes:
        orbital_mechanics (OrbitalMechanics): Helper for orbital calculations.
        sun (Optional[CelestialBody]): The central star. `None` if not initialized.
        planets (List[CelestialBody]): List of planets, moons, etc. (excluding Sun).
        initial_system_energy (Optional[float]): Total mechanical energy of
            celestial bodies at initialization (if `config.Debug.MONITOR_ENERGY_CONSERVATION`).
            `None` if not calculated or failed.
        celestial_body_references (Dict[str, CelestialBody]): Maps body name to
            its `CelestialBody` instance for quick lookup.

    Raises:
        ConfigurationError: If initialization fails due to invalid/missing config
            from `config.SolarSystem`, `config.Probe`, `config.Resource`, or parent.
        PhysicsError: If critical errors occur in initial orbital calculations
            (e.g., Kepler's equation, state vectors) or during physics updates.
        Exception: For other unexpected errors during initialization or stepping.
    """
    def __init__(self):
        """Initializes the `SolarSystemEnvironment`.

        Steps:
        1.  Calls `super().__init__()` (base `SpaceEnvironment` initialization).
        2.  Creates an `OrbitalMechanics` instance.
        3.  Initializes `self.sun`, `self.planets`, `self.celestial_body_references`.
        4.  Calls `self._create_celestial_bodies()` to populate the solar system
            from `config.SolarSystem.PLANET_DATA`, calculating initial orbital states.
        5.  If `config.Debug.MONITOR_ENERGY_CONSERVATION` is true, calculates and
            stores `initial_system_energy` of celestial bodies.

        Error Handling:
            Catches and logs `ConfigurationError`, `PhysicsError`, and general
            `Exception` during these steps, re-raising them to signal failure.
        """
        try:
            super().__init__() # Initialize SpaceEnvironment components
            
            self.orbital_mechanics = OrbitalMechanics()
            self.sun: Optional[CelestialBody] = None
            self.planets: List[CelestialBody] = [] # Includes planets and moons
            self.celestial_body_references: Dict[str, CelestialBody] = {}
            self.initial_system_energy: Optional[float] = None

            self._create_celestial_bodies() # Populates sun, planets, references
            
            # Calculate initial system energy if monitoring is enabled
            if config.Debug.MONITOR_ENERGY_CONSERVATION:
                all_bodies_for_energy_calc = ([self.sun] if self.sun else []) + self.planets
                if all_bodies_for_energy_calc:
                    try:
                        self.initial_system_energy = self.orbital_mechanics.calculate_total_system_energy(all_bodies_for_energy_calc)
                        if config.Debug.ORBITAL_MECHANICS: # Log only if detailed orbital debug is on
                            logging.debug(f"Initial system energy of celestial bodies: {self.initial_system_energy:.6e} Joules (kg*(km/s)^2)")
                    except PhysicsError as pe_energy:
                        logging.error(f"PhysicsError calculating initial system energy of celestial bodies: {pe_energy}", exc_info=True)
                        self.initial_system_energy = None # Ensure it's None on error
                    except Exception as e_energy: # Catch other unexpected errors
                        logging.error(f"Unexpected error calculating initial system energy of celestial bodies: {e_energy}", exc_info=True)
                        self.initial_system_energy = None
            
            logging.info(f"SolarSystemEnvironment initialized with {len(self.planets) + (1 if self.sun else 0)} celestial bodies.")

        except ConfigurationError as ce_init:
            logging.critical(f"SolarSystemEnvironment initialization failed due to ConfigurationError: {ce_init}", exc_info=True)
            raise
        except PhysicsError as pe_init:
            logging.critical(f"SolarSystemEnvironment initialization failed due to PhysicsError: {pe_init}", exc_info=True)
            raise
        except Exception as e_init_unexpected:
            logging.critical(f"Unexpected error during SolarSystemEnvironment initialization: {e_init_unexpected}", exc_info=True)
            raise


    def _create_celestial_bodies(self):
        """Creates and initializes all celestial bodies from config.

        Responsibilities:
        1.  Iterates `config.SolarSystem.PLANET_DATA`.
        2.  Creates `CelestialBody` instances for Sun, planets, and moons.
        3.  Calculates initial 2D position and velocity vectors for each body
            relative to its primary (or absolute for Sun at `config.World.CENTER_SIM`)
            using `orbital_mechanics.calculate_initial_state_vector()`.
        4.  Converts relative positions/velocities to absolute simulation coordinates.
        5.  Calculates initial gravitational acceleration for each body (for Verlet).
        6.  Stores Sun in `self.sun`, others in `self.planets`, and all in
            `self.celestial_body_references` by name.

        Order of Creation: Sun -> Planets -> Moons (ensuring primaries exist).

        Raises:
            ConfigurationError: If `config.SolarSystem.PLANET_DATA` is missing/malformed,
                lacks required keys (e.g., 'mass_kg', 'semi_major_axis_au'), or if
                global configs (`config.AU_SCALE`, `config.World.CENTER_SIM`) are invalid.
            PhysicsError: If `calculate_initial_state_vector` or
                `calculate_gravitational_acceleration` fails critically.
            KeyError: If expected body names (e.g., 'Sun' in `PLANET_DATA`, or a
                'central_body' name not found) are missing.
            Exception: For other unexpected errors.
        """
        try:
            self.planets = [] # Clear previous list
            self.celestial_body_references = {} # Clear previous references

            # --- Create Sun ---
            sun_cfg = config.SolarSystem.PLANET_DATA.get('Sun')
            if not sun_cfg: raise ConfigurationError("Sun data missing from config.SolarSystem.PLANET_DATA")
            
            self.sun = CelestialBody(
                name="Sun",
                mass_kg=float(sun_cfg['mass_kg']), radius_km=float(sun_cfg['radius_km']),
                display_radius_sim=float(sun_cfg['display_radius_sim']), color=tuple(sun_cfg['color']),
                a_au=float(sun_cfg['semi_major_axis_au']), e=float(sun_cfg['eccentricity']),
                i_deg=float(sun_cfg['inclination_deg']), omega_deg=float(sun_cfg['longitude_of_ascending_node_deg']),
                w_deg=float(sun_cfg['argument_of_perihelion_deg']), m0_deg=float(sun_cfg['mean_anomaly_at_epoch_deg']),
                position_sim=np.array(config.World.CENTER_SIM, dtype=np.float64), # Placed at world center
                velocity_sim_s=np.array([0.0, 0.0], dtype=np.float64), # Sun is stationary reference
                orbits_around=None # Sun orbits nothing
            )
            all_created_bodies_list: List[CelestialBody] = [self.sun]
            self.celestial_body_references[self.sun.name] = self.sun
            if config.Debug.ORBITAL_MECHANICS:
                logging.debug(f"Created Sun: Pos_sim={self.sun.position_sim.tolist()}, Vel_sim_s={self.sun.velocity_sim_s.tolist()}")

            # --- Create Planets (orbiting Sun) ---
            # Define order to ensure primaries are created before their secondaries
            # This could be made more robust by dependency tracking if hierarchy is complex
            planet_creation_order = [
                "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"
            ]
            for planet_name in planet_creation_order:
                planet_cfg = config.SolarSystem.PLANET_DATA.get(planet_name)
                if not planet_cfg:
                    logging.warning(f"Configuration for planet '{planet_name}' not found. Skipping.")
                    continue
                if planet_cfg.get('central_body') != 'Sun': # Validate primary
                    logging.warning(f"Planet '{planet_name}' config has central_body '{planet_cfg.get('central_body')}', expected 'Sun'. Assuming 'Sun'.")

                planet_obj = CelestialBody(
                    name=planet_name, mass_kg=float(planet_cfg['mass_kg']), radius_km=float(planet_cfg['radius_km']),
                    display_radius_sim=float(planet_cfg['display_radius_sim']), color=tuple(planet_cfg['color']),
                    a_au=float(planet_cfg['semi_major_axis_au']), e=float(planet_cfg['eccentricity']),
                    i_deg=float(planet_cfg['inclination_deg']), omega_deg=float(planet_cfg['longitude_of_ascending_node_deg']),
                    w_deg=float(planet_cfg['argument_of_perihelion_deg']), m0_deg=float(planet_cfg['mean_anomaly_at_epoch_deg']),
                    orbits_around="Sun"
                )
                # Calculate initial state relative to Sun
                pos_rel_sun_sim_units, vel_rel_sun_sim_units_per_s = self.orbital_mechanics.calculate_initial_state_vector(
                    planet_obj, self.sun.mass_kg
                )
                planet_obj.position_sim = self.sun.position_sim + pos_rel_sun_sim_units
                planet_obj.velocity_sim_s = self.sun.velocity_sim_s + vel_rel_sun_sim_units_per_s
                
                self.planets.append(planet_obj)
                all_created_bodies_list.append(planet_obj)
                self.celestial_body_references[planet_name] = planet_obj
                if config.Debug.ORBITAL_MECHANICS:
                    logging.debug(f"Created {planet_name}: Pos_sim={planet_obj.position_sim.tolist()}, Vel_sim_s={planet_obj.velocity_sim_s.tolist()}")

            # --- Create Moons (orbiting Planets) ---
            for body_name, body_cfg in config.SolarSystem.PLANET_DATA.items():
                primary_body_name = body_cfg.get('central_body')
                if primary_body_name and primary_body_name != 'Sun' and body_name not in planet_creation_order: # Is a moon
                    if primary_body_name not in self.celestial_body_references:
                        logging.error(f"Primary body '{primary_body_name}' for moon '{body_name}' not found. Skipping '{body_name}'.")
                        continue
                    primary_celestial_body = self.celestial_body_references[primary_body_name]

                    moon_obj = CelestialBody(
                        name=body_name, mass_kg=float(body_cfg['mass_kg']), radius_km=float(body_cfg['radius_km']),
                        display_radius_sim=float(body_cfg['display_radius_sim']), color=tuple(body_cfg['color']),
                        a_au=float(body_cfg['semi_major_axis_au']), e=float(body_cfg['eccentricity']),
                        i_deg=float(body_cfg['inclination_deg']), omega_deg=float(body_cfg['longitude_of_ascending_node_deg']),
                        w_deg=float(body_cfg['argument_of_perihelion_deg']), m0_deg=float(body_cfg['mean_anomaly_at_epoch_deg']),
                        orbits_around=primary_body_name
                    )
                    pos_rel_primary_sim_units, vel_rel_primary_sim_units_per_s = self.orbital_mechanics.calculate_initial_state_vector(
                        moon_obj, primary_celestial_body.mass_kg
                    )
                    moon_obj.position_sim = primary_celestial_body.position_sim + pos_rel_primary_sim_units
                    moon_obj.velocity_sim_s = primary_celestial_body.velocity_sim_s + vel_rel_primary_sim_units_per_s

                    self.planets.append(moon_obj) # Moons are also added to the 'planets' list for general processing
                    all_created_bodies_list.append(moon_obj)
                    self.celestial_body_references[body_name] = moon_obj
                    if config.Debug.ORBITAL_MECHANICS:
                        logging.debug(f"Created {body_name} orbiting {primary_body_name}: Pos_sim={moon_obj.position_sim.tolist()}, Vel_sim_s={moon_obj.velocity_sim_s.tolist()}")
            
            # --- Calculate initial accelerations for Verlet integration ---
            for body_to_init_accel in all_created_bodies_list:
                # Influencers are all *other* bodies
                influencing_bodies_list = [other for other in all_created_bodies_list if other is not body_to_init_accel]
                initial_accel_km_per_s2 = self.orbital_mechanics.calculate_gravitational_acceleration(
                    body_to_init_accel, influencing_bodies_list
                )
                # Convert km/s^2 to sim_units/s^2 using config.KM_SCALE (sim_units per km)
                body_to_init_accel.previous_acceleration_sim_s2 = initial_accel_km_per_s2 * config.KM_SCALE
                if config.Debug.ORBITAL_MECHANICS and body_to_init_accel.name in ["Earth", "Moon", "Sun"]: # Log for key bodies
                     logging.debug(f"Initial Accel for {body_to_init_accel.name} (sim_units/s^2): {body_to_init_accel.previous_acceleration_sim_s2.tolist()}")
        
        except KeyError as e_key: # Missing key in PLANET_DATA or celestial_body_references
            logging.critical(f"Missing key during celestial body creation: {e_key}. Check PLANET_DATA structure and central_body names.", exc_info=True)
            raise ConfigurationError(f"Missing key in PLANET_DATA or body reference: {e_key}")
        except PhysicsError as e_phys: # From orbital_mechanics calls
            logging.critical(f"PhysicsError during celestial body creation: {e_phys}", exc_info=True)
            raise # Re-raise to signal critical failure
        except AttributeError as e_attr: # Missing global config (e.g., AU_SCALE, KM_SCALE, World.CENTER_SIM)
            logging.critical(f"Error creating celestial bodies due to missing global config value: {e_attr}", exc_info=True)
            raise ConfigurationError(f"Failed to create celestial bodies due to missing global config: {e_attr}")
        except Exception as e_unexpected: # Catch-all for other errors
            logging.critical(f"Unexpected error creating celestial bodies: {e_unexpected}", exc_info=True)
            raise # Re-raise as critical failure


    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[int, Dict]]:
        """Advances the solar system simulation by one time step.

        Orchestrates:
        1.  **Celestial Body Physics**: Propagates orbits of all celestial bodies
            (Sun, planets, moons) using Verlet integration via
            `OrbitalMechanics.propagate_orbits_verlet()`. Updates their
            positions and velocities based on mutual gravitational forces.
        2.  **Probe Physics & Actions**: Calls `super().step(actions)`.
            `SolarSystemEnvironment` overrides methods called by the base `step`
            (or the base `step` itself if necessary, though current design assumes
            base `step` iterates probes and calls helpers) to:
            a.  Apply gravitational forces from celestial bodies to each probe.
            b.  Process probe actions (thrust, rotation, mining, replication,
                communication) from `actions`.
            c.  Update probe states (position, velocity, energy, angle).
            d.  Handle resource interactions (mining by probes, regeneration by environment).
            e.  Calculate rewards for probes.
            f.  Generate new observations for probes (via `get_observation()`).
            g.  Determine 'done' status for probes.
        3.  **Logging**: Periodically logs orbital data and total system energy
            if debug flags in `config.Debug` are enabled.

        Args:
            actions (Dict[int, np.ndarray]): Maps probe IDs to their selected
                actions for this step. Action structure from `self.action_space`.

        Returns:
            Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[int, Dict]]:
            Dictionaries (keyed by probe ID) for new observations, rewards,
            'done' flags, and info.

        Error Handling:
            - `PhysicsError` during celestial body orbit propagation is logged;
              simulation attempts to continue (celestial states might be unstable).
              Other unexpected errors in orbit propagation are also logged.
            - Critical errors in `SolarSystemEnvironment.step` logic (outside
              `super().step()` or celestial mechanics) are caught, logged, and a
              fallback (all probes 'done', zero reward) is returned to prevent
              crashing the main simulation loop.
            - Individual probe processing errors within `super().step()` are
              managed by `SpaceEnvironment.step` (typically marks failing probe
              as 'done', continues with others).
        """
        try:
            # 1. Update Celestial Body Positions and Velocities
            all_celestial_entities: List[CelestialBody] = ([self.sun] if self.sun else []) + self.planets
            if all_celestial_entities:
                try:
                    sim_time_per_step = config.Time.SIM_SECONDS_PER_STEP
                    if not (isinstance(sim_time_per_step, (int,float)) and sim_time_per_step > 0):
                        raise ConfigurationError("config.Time.SIM_SECONDS_PER_STEP must be a positive number.")
                    
                    self.orbital_mechanics.propagate_orbits_verlet(
                        all_celestial_entities,
                        dt_seconds=sim_time_per_step
                    )
                except PhysicsError as e_phys_propagate:
                    logging.error(f"PhysicsError during orbit propagation in step {self.step_count}: {e_phys_propagate}. Celestial body states may be unstable.", exc_info=True)
                except ConfigurationError as e_config_propagate: # Catch our specific config error
                    logging.error(f"ConfigurationError during orbit propagation setup in step {self.step_count}: {e_config_propagate}", exc_info=True)
                    # This might be critical enough to stop or mark all as done. For now, log and continue.
                except Exception as e_propagate_unexpected:
                    logging.error(f"Unexpected error during orbit propagation in step {self.step_count}: {e_propagate_unexpected}", exc_info=True)

            # 2. Debug Logging for Orbital Mechanics and Energy Conservation
            if self.step_count > 0: # Avoid logging at step 0 before first real update
                if config.Debug.ORBITAL_MECHANICS and (self.step_count % config.Debug.LOG_ORBIT_INTERVAL_STEPS == 0) and all_celestial_entities:
                    for body_log in all_celestial_entities:
                        if body_log.name in config.Debug.LOG_ORBIT_BODY_NAMES:
                            pos_au_log = body_log.position_sim / config.AU_SCALE
                            logging.debug(f"Step {self.step_count}: {body_log.name} Pos_AU=[{pos_au_log[0]:.4f}, {pos_au_log[1]:.4f}]")
                
                if config.Debug.MONITOR_ENERGY_CONSERVATION and \
                   (self.step_count % config.Debug.ENERGY_CHECK_INTERVAL_STEPS == 0) and \
                   self.initial_system_energy is not None and all_celestial_entities:
                    try:
                        current_system_energy = self.orbital_mechanics.calculate_total_system_energy(all_celestial_entities)
                        energy_change = current_system_energy - self.initial_system_energy
                        energy_change_percent = safe_divide(energy_change * 100.0, self.initial_system_energy, default_on_zero_denom=0.0)
                        logging.info(f"ENERGY CHECK (Step {self.step_count}): Celestial Bodies - Current: {current_system_energy:.6e}, Initial: {self.initial_system_energy:.6e}, Delta: {energy_change:.6e} ({energy_change_percent:.6f}%)")
                    except PhysicsError as e_phys_energy_check:
                        logging.error(f"PhysicsError calculating system energy during step {self.step_count} check: {e_phys_energy_check}", exc_info=True)
                    except Exception as e_energy_check_unexpected:
                        logging.error(f"Unexpected error calculating system energy during step {self.step_count} check: {e_energy_check_unexpected}", exc_info=True)
            
            # 3. Call base class step for probe logic, resource regeneration, etc.
            # SolarSystemEnvironment will need to override get_observation and potentially
            # helper methods called by SpaceEnvironment.step to include gravity,
            # detailed probe actions, and specific rewards.
            observations_dict, rewards_dict, dones_dict, infos_dict = super().step(actions)
            
            # --- Additional SolarSystem specific logic after super().step() if needed ---
            # For example, checking for system-wide events or conditions.

            return observations_dict, rewards_dict, dones_dict, infos_dict

        except Exception as e_solar_step_main: # Catch-all for errors in SolarSystemEnvironment.step itself
            logging.critical(f"Critical error in SolarSystemEnvironment.step (main logic, outside super().step or celestial mechanics) at step {self.step_count}: {e_solar_step_main}", exc_info=True)
            # Fallback: Mark all probes that had actions submitted as 'done'
            obs_fallback, rew_fallback, dn_fallback, inf_fallback = {}, {}, {}, {}
            for pid_action_fallback in actions.keys(): # Iterate over keys from the input 'actions' dict
                 obs_fallback[pid_action_fallback] = self.get_observation(pid_action_fallback) # Attempt to get last valid obs
                 rew_fallback[pid_action_fallback] = 0.0
                 dn_fallback[pid_action_fallback] = True # Mark as done due to critical error
                 inf_fallback[pid_action_fallback] = {'critical_env_step_error': str(e_solar_step_main)}
            return obs_fallback, rew_fallback, dn_fallback, inf_fallback


    def reset(self) -> Dict[int, np.ndarray]:
        """Resets the `SolarSystemEnvironment` to its initial state.

        Performs:
        1.  Calls `self._create_celestial_bodies()` to re-initialize Sun, planets,
            moons to their epoch states (positions, velocities, initial accelerations).
            Ensures a consistent starting solar system physics configuration.
        2.  If `config.Debug.MONITOR_ENERGY_CONSERVATION` is enabled, recalculates
            and stores `initial_system_energy` of the reset celestial bodies.
        3.  Calls `super().reset()` for base `SpaceEnvironment` reset logic:
            - Resets `step_count`, `max_probe_id`, `total_resources_mined`.
            - Clears `probes` and `messages`.
            - Regenerates `resources`.
            - Adds initial probes.
            - Returns initial observations for new probes.

        Returns:
            Dict[int, np.ndarray]: Maps new probe IDs to their initial
            observation vectors, from `super().reset()`.

        Raises:
            ConfigurationError: If config issues arise during celestial body creation
                or base environment reset (e.g., missing `config` for probes, resources).
            PhysicsError: If critical physics calculations fail during celestial body
                creation (e.g., initial state vectors, accelerations).
            Exception: For other unexpected critical errors during reset. These are
                logged and re-raised.
        """
        try:
            logging.info(f"Resetting {self.__class__.__name__} environment for a new episode...")
            
            # 1. Re-initialize celestial bodies to their defined epoch states
            self._create_celestial_bodies() # Can raise ConfigurationError, PhysicsError
            
            # 2. Recalculate initial system energy for the newly reset bodies
            if config.Debug.MONITOR_ENERGY_CONSERVATION:
                all_celestial_entities_reset = ([self.sun] if self.sun else []) + self.planets
                if all_celestial_entities_reset:
                    try:
                        self.initial_system_energy = self.orbital_mechanics.calculate_total_system_energy(all_celestial_entities_reset)
                        if config.Debug.ORBITAL_MECHANICS: # Log if detailed orbital debug is on
                             logging.debug(f"Initial system energy of celestial bodies (after reset): {self.initial_system_energy:.6e} Joules")
                    except PhysicsError as e_phys_energy_reset:
                        logging.error(f"PhysicsError calculating initial system energy of celestial bodies after reset: {e_phys_energy_reset}", exc_info=True)
                        self.initial_system_energy = None # Ensure it's None on error
                    except Exception as e_energy_reset_unexpected: # Catch other unexpected errors
                        logging.error(f"Unexpected error calculating initial system energy of celestial bodies after reset: {e_energy_reset_unexpected}", exc_info=True)
                        self.initial_system_energy = None
            
            # 3. Call base class reset for probes, resources, step_count, etc.
            initial_probe_observations = super().reset() # Can raise ConfigurationError, Exception
            
            logging.info(f"{self.__class__.__name__} reset complete. Initial observations for {len(initial_probe_observations)} probes generated.")
            return initial_probe_observations

        except ConfigurationError as e_config_reset_solar: # From _create_celestial_bodies or super().reset()
            logging.critical(f"SolarSystemEnvironment reset failed due to ConfigurationError: {e_config_reset_solar}", exc_info=True)
            raise
        except PhysicsError as e_phys_reset_solar: # From _create_celestial_bodies or energy calculation
            logging.critical(f"SolarSystemEnvironment reset failed due to PhysicsError: {e_phys_reset_solar}", exc_info=True)
            raise
        except Exception as e_reset_solar_unexpected: # Catch-all for other unexpected errors
            logging.critical(f"Unexpected critical error during SolarSystemEnvironment reset: {e_reset_solar_unexpected}", exc_info=True)
            raise # Re-raise to ensure the simulation manager knows the reset failed


    def get_celestial_bodies_data_for_render(self) -> List[Dict]:
        """Retrieves data for celestial bodies and active probes for rendering.

        Compiles a list of dictionaries for `Visualization` to render game entities.
        Includes data for Sun, planets (which includes moons from `self.planets`),
        and all 'alive' probes.

        Celestial Body Data Format:
        -   `name` (str): Name of the body.
        -   `position` (List[float, float]): Current [x, y] simulation coordinates.
        -   `radius_sim` (float): Display radius in simulation units.
        -   `color` (Tuple[int, int, int]): RGB color.
        -   `orbit_path` (List[List[float, float]]): Recent orbital path points.

        Probe Data Format:
        -   `name` (str): "Probe_ID".
        -   `type` (str): 'probe'.
        -   `position` (List[float, float]): Current [x, y] simulation coordinates.
        -   `radius_sim` (float): Probe display radius (from `config.Visualization.PROBE_SIZE_PX`).
        -   `color` (Tuple[int, int, int]): Default probe color.
        -   `trail` (List[List[float, float]]): Recent movement trail points.

        Returns:
            List[Dict]: List of entity dictionaries. Empty if no entities or error.

        Error Handling:
            - Catches `AttributeError` for missing config (e.g.,
              `config.Visualization.PROBE_SIZE_PX`), logs, and continues if possible.
            - Catches general `Exception` for unexpected issues, logs, and attempts
              to return partially collected data to prevent render crash.
        """
        renderable_entities_data: List[Dict] = []
        try:
            # Add Sun data (if exists)
            if self.sun and isinstance(self.sun, CelestialBody):
                renderable_entities_data.append({
                    'name': self.sun.name,
                    'position': self.sun.position_sim.tolist() if isinstance(self.sun.position_sim, np.ndarray) else list(self.sun.position_sim),
                    'radius_sim': float(self.sun.display_radius_sim),
                    'color': tuple(self.sun.color),
                    'orbit_path': [p.tolist() if isinstance(p, np.ndarray) else list(p) for p in self.sun.orbit_path if p is not None]
                })
            
            # Add Planets and Moons data (all are in self.planets list)
            for celestial_obj in self.planets:
                if isinstance(celestial_obj, CelestialBody):
                    renderable_entities_data.append({
                        'name': celestial_obj.name,
                        'position': celestial_obj.position_sim.tolist() if isinstance(celestial_obj.position_sim, np.ndarray) else list(celestial_obj.position_sim),
                        'radius_sim': float(celestial_obj.display_radius_sim),
                        'color': tuple(celestial_obj.color),
                        'orbit_path': [p.tolist() if isinstance(p, np.ndarray) else list(p) for p in celestial_obj.orbit_path if p is not None]
                    })
            
            # Add Probe data
            # Ensure config values are valid before use
            probe_disp_size_px = config.Visualization.PROBE_SIZE_PX
            if not (isinstance(probe_disp_size_px, (int,float)) and probe_disp_size_px > 0):
                logging.warning(f"Invalid config.Visualization.PROBE_SIZE_PX ({probe_disp_size_px}). Using fallback 10.")
                probe_disp_size_px = 10.0
            # Assuming PROBE_SIZE_PX is diameter, radius is half. This needs clarification if it's radius.
            # For now, let's assume it's a general size indicator, and visualization might interpret it.
            # If it's truly pixels, it shouldn't be 'radius_sim' unless converted.
            # Let's assume it's a sim unit radius for consistency with celestial bodies.
            # This part needs careful review based on how PROBE_SIZE_PX is intended.
            # If PROBE_SIZE_PX is pixels, then it's not a 'sim' radius.
            # For now, I'll treat it as a sim_unit radius for the dict, but this is a potential mismatch.
            # A better approach might be for Visualization to handle PROBE_SIZE_PX directly.
            # Let's assume for now it's a small sim_unit radius.
            probe_render_radius_sim_units = probe_disp_size_px * 0.01 # Arbitrary conversion if it was pixels
                                                                # This needs to be defined by a scale factor.
                                                                # Or, pass PROBE_SIZE_PX directly to viz.
                                                                # For now, using a small fixed sim radius.
            probe_render_radius_sim_units = 5.0 # Placeholder sim units radius for probes

            probe_color_cfg = config.Visualization.PROBE_DEFAULT_COLOR
            if not (isinstance(probe_color_cfg, (list, tuple)) and len(probe_color_cfg) == 3 and all(isinstance(c, int) for c in probe_color_cfg)):
                logging.warning(f"Invalid config.Visualization.PROBE_DEFAULT_COLOR. Using fallback (100,150,255).")
                probe_color_cfg = (100,150,255)


            for probe_unique_id, probe_info_dict in self.probes.items():
                if isinstance(probe_info_dict, dict) and probe_info_dict.get('alive', False):
                    probe_pos_sim = probe_info_dict.get('position', np.array([0.0, 0.0]))
                    probe_trail_list = probe_info_dict.get('trail', [])
                    
                    renderable_entities_data.append({
                        'name': f"Probe_{probe_unique_id}",
                        'type': 'probe', # Differentiator for renderer
                        'position': probe_pos_sim.tolist() if isinstance(probe_pos_sim, np.ndarray) else list(probe_pos_sim),
                        'radius_sim': probe_render_radius_sim_units, # This needs to be a sim unit value
                        'color': tuple(probe_color_cfg),
                        'trail': [t.tolist() if isinstance(t, np.ndarray) else list(t) for t in probe_trail_list if t is not None]
                    })
        except AttributeError as e_attr_render: # Specifically for missing config attributes
            logging.error(f"Error getting data for render due to missing config attribute: {e_attr_render}", exc_info=True)
            # renderable_entities_data might be partially populated; return what has been collected
        except Exception as e_render_unexpected: # Catch any other unexpected errors
            logging.error(f"Unexpected error in get_celestial_bodies_data_for_render: {e_render_unexpected}", exc_info=True)
            # renderable_entities_data might be partially populated
        return renderable_entities_data