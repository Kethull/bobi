# config.py
import numpy as np
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# Fundamental Physical Constants (used across different config sections)
AU_KM = 149597870.7  # Astronomical Unit in kilometers
GRAVITATIONAL_CONSTANT_KM3_KG_S2 = 6.67430e-20  # G in km^3 kg^-1 s^-2
SECONDS_PER_DAY = 86400.0

# Simulation Scale Constants (also fundamental for conversions)
AU_SCALE = 10000.0  # Sim units per AU
KM_SCALE = AU_SCALE / AU_KM  # Sim units per km

class ConfigurationError(Exception):
    """Custom exception for simulation configuration errors.

    Raised by `SimulationConfig.validate()` and other configuration-dependent
    components when settings are invalid, inconsistent, or missing, which
    would prevent the simulation from running correctly.

    Attributes:
        message (str): A human-readable explanation of the configuration error.
                       This is the first argument passed to the exception constructor.
    """
    pass

class SimulationConfig:
    """Centralized, hierarchical configuration for the Bobiverse Orbital Simulation.

    This class consolidates all simulation parameters into nested static classes
    (e.g., `SimulationConfig.World`, `SimulationConfig.Physics`, `SimulationConfig.Probe`)
    for organized access. An instance of this class, named `config`, is created
    at the end of this module, making it globally available via `from config import config`.

    The `__init__` method performs initial calculations for derived parameters (like
    RL action space dimensions based on probe capabilities) and critically invokes
    the `validate()` method. The `validate()` method performs extensive checks
    across all configuration sections to ensure logical consistency, valid ranges,
    and correct interdependencies, raising a `ConfigurationError` if any issues
    are found. This preempts runtime failures due to faulty configuration.

    Example Usage:
        >>> from config import config
        >>> print(f"World Size (AU): {config.World.SIZE_AU}")
        >>> print(f"Physics Timestep (s): {config.Physics.TIMESTEP_SECONDS}")
        >>> print(f"Probe Max Energy: {config.Probe.MAX_ENERGY}")
    """

    # --- World Configuration ---
    class World:
        """Configuration for the simulation world's physical properties.

        Attributes:
            SIZE_AU (float): Diameter of the primary simulation area in Astronomical Units (AU).
            SIZE_SIM (float): Diameter of the simulation area in internal simulation units,
                              derived from `SIZE_AU` and `AU_SCALE`.
            WIDTH_SIM (float): Width of the simulation world in simulation units.
            HEIGHT_SIM (float): Height of the simulation world in simulation units.
            CENTER_SIM (List[float, float]): Coordinates [x, y] of the world's center
                                             in simulation units.
            ASTEROID_BELT_INNER_AU (float): Inner radius of the asteroid belt in AU.
            ASTEROID_BELT_OUTER_AU (float): Outer radius of the asteroid belt in AU.
            ASTEROID_COUNT (int): Number of asteroids to generate in the belt.
            ASTEROID_MIN_RADIUS_SIM (float): Minimum display radius for asteroids in sim units.
            ASTEROID_MAX_RADIUS_SIM (float): Maximum display radius for asteroids in sim units.
            ASTEROID_DEFAULT_COLOR (Tuple[int, int, int]): Default RGB color for asteroids.
            ASTEROID_MASS_KG_MIN (float): Minimum mass for an asteroid in kilograms.
            ASTEROID_MASS_KG_MAX (float): Maximum mass for an asteroid in kilograms.
        """
        SIZE_AU = 10.0
        SIZE_SIM = SIZE_AU * AU_SCALE
        WIDTH_SIM = SIZE_SIM
        HEIGHT_SIM = SIZE_SIM
        CENTER_SIM = [WIDTH_SIM / 2, HEIGHT_SIM / 2]

        ASTEROID_BELT_INNER_AU = 2.2
        ASTEROID_BELT_OUTER_AU = 3.2
        ASTEROID_COUNT = 500
        ASTEROID_MIN_RADIUS_SIM = 0.2
        ASTEROID_MAX_RADIUS_SIM = 0.8
        ASTEROID_DEFAULT_COLOR = (100, 100, 100)
        ASTEROID_MASS_KG_MIN = 1e10
        ASTEROID_MASS_KG_MAX = 1e15

    # --- Physics Configuration ---
    class Physics:
        """Configuration for the simulation's physics engine.

        Attributes:
            TIMESTEP_SECONDS (float): Duration of one simulation step in real-world seconds.
                                      This is a critical parameter for integration accuracy.
            INTEGRATION_METHOD (str): Specifies the numerical integration method to use.
                                      Supported: "euler", "verlet", "leapfrog".
            DEFAULT_PROBE_MASS_KG (float): Default mass of a probe in kilograms. Used if
                                           a specific probe type doesn't define its own mass.
        """
        TIMESTEP_SECONDS = 3600.0
        INTEGRATION_METHOD = "verlet"
        DEFAULT_PROBE_MASS_KG = 8.0

    # --- Time Configuration ---
    class Time:
        """Configuration related to simulation time progression.

        Attributes:
            SIM_SECONDS_PER_STEP (float): Real-world seconds that elapse per simulation step.
                                          Typically mirrors `Physics.TIMESTEP_SECONDS` but
                                          is kept separate for potential future distinctions
                                          (e.g., variable time scaling).
        """
        SIM_SECONDS_PER_STEP = 3600.0 # Effectively Physics.TIMESTEP_SECONDS

    # --- Probe Configuration ---
    class Probe:
        """Configuration for autonomous probes.

        Defines their quantity, energy systems, movement capabilities (linear and angular),
        action mechanics (smoothing, anti-spam), and communication abilities.

        Attributes:
            MAX_PROBES (int): Maximum number of probes allowed simultaneously.
            INITIAL_PROBES (int): Number of probes created at simulation start.
            MAX_ENERGY (float): Maximum energy capacity of a probe.
            INITIAL_ENERGY (float): Energy level of newly created probes.
            REPLICATION_COST (float): Energy cost for a probe to replicate.
            REPLICATION_MIN_ENERGY (float): Minimum energy a probe must have to attempt replication.
            MAX_VELOCITY_SIM_PER_STEP (float): Maximum linear speed in simulation units per step.
            THRUST_FORCE_MAGNITUDES (List[float]): Available discrete thrust force magnitudes.
            THRUST_ENERGY_COST_FACTOR (float): Energy cost per unit of thrust applied per step.
            ENERGY_DECAY_RATE_PER_STEP (float): Passive energy loss per step.
            LOW_POWER_PENALTY_FACTOR (float): Multiplicative factor applied to performance
                                              (e.g., thrust) when energy is critically low.
            MOMENT_OF_INERTIA (float): Probe's moment of inertia (kg*m^2 or sim equivalent).
            TORQUE_MAGNITUDES (List[float]): Available discrete torque magnitudes for rotation.
            ROTATIONAL_ENERGY_COST_FACTOR (float): Energy cost per unit of torque per step.
            MAX_ANGULAR_VELOCITY_RAD_PER_STEP (float): Max angular speed in radians per step.
            ANGULAR_DAMPING_FACTOR (float): Passive decay factor for angular velocity.
            ACTION_SMOOTHING_FACTOR (float): Factor for smoothing thrust/torque application over time.
            MIN_THRUST_DURATION_STEPS (int): Minimum steps a thrust level must be maintained.
            THRUST_RAMP_TIME_STEPS (int): Steps over which thrust ramps up/down.
            ROTATION_SMOOTHING_FACTOR (float): Factor for smoothing rotational changes.
            MIN_ROTATION_DURATION_STEPS (int): Minimum steps a rotation command is active.
            ROTATION_RAMP_TIME_STEPS (int): Steps over which rotation ramps up/down.
            THRUSTER_STARTUP_ENERGY_COST (float): One-time energy cost for initiating thrust.
            RAPID_SWITCHING_PENALTY_FACTOR (float): Penalty for frequently changing actions.
            SWITCHING_DETECTION_WINDOW_STEPS (int): Time window (steps) to detect rapid action switching.
            COMM_RANGE_SIM (float): Communication range in simulation units.
            MESSAGE_TYPES (List[str]): Allowed types of messages probes can exchange.
        """
        MAX_PROBES = 20
        INITIAL_PROBES = 1

        MAX_ENERGY = 100000.0
        INITIAL_ENERGY = 90000.0
        REPLICATION_COST = 80000.0
        REPLICATION_MIN_ENERGY = 99900.0
        MAX_VELOCITY_SIM_PER_STEP = 10000.0
        
        THRUST_FORCE_MAGNITUDES = [0.0, 0.08, 0.18, 0.32]
        THRUST_ENERGY_COST_FACTOR = 0.001
        ENERGY_DECAY_RATE_PER_STEP = 0.001
        LOW_POWER_PENALTY_FACTOR = 0.9

        MOMENT_OF_INERTIA = 5.0
        TORQUE_MAGNITUDES = [0.0, 0.008, 0.018]
        ROTATIONAL_ENERGY_COST_FACTOR = 0.11
        MAX_ANGULAR_VELOCITY_RAD_PER_STEP = np.pi / 4
        ANGULAR_DAMPING_FACTOR = 0.05

        ACTION_SMOOTHING_FACTOR = 0.85
        MIN_THRUST_DURATION_STEPS = 30
        THRUST_RAMP_TIME_STEPS = 60
        ROTATION_SMOOTHING_FACTOR = 0.9
        MIN_ROTATION_DURATION_STEPS = 6
        ROTATION_RAMP_TIME_STEPS = 4

        THRUSTER_STARTUP_ENERGY_COST = 1.5
        RAPID_SWITCHING_PENALTY_FACTOR = 0.8
        SWITCHING_DETECTION_WINDOW_STEPS = 10

        COMM_RANGE_SIM = 100.0
        MESSAGE_TYPES = ['RESOURCE_LOCATION']

    # --- Resource Configuration ---
    class Resource:
        """Configuration for harvestable resources.

        Attributes:
            COUNT (int): Number of resource nodes to generate in the environment.
            MIN_AMOUNT (float): Minimum initial amount of resource a node can have.
            MAX_AMOUNT (float): Maximum initial/capacity amount of resource a node can have.
            REGEN_RATE_PER_STEP (float): Amount of resource regenerated per node per step.
            HARVEST_RATE_PER_STEP (float): Amount of resource a probe can harvest per step
                                           when actively mining.
            HARVEST_DISTANCE_SIM (float): Maximum distance (sim units) from which a probe
                                          can harvest a resource.
            DISCOVERY_RANGE_SIM (float): Distance (sim units) within which a probe can
                                         detect/discover a resource node.
            DISCOVERY_REWARD_FACTOR (float): Multiplier for the reward given upon discovering
                                             a new resource node.
        """
        COUNT = 15
        MIN_AMOUNT = 10000
        MAX_AMOUNT = 20000
        REGEN_RATE_PER_STEP = 0.00
        HARVEST_RATE_PER_STEP = 2.0
        HARVEST_DISTANCE_SIM = 5.0
        DISCOVERY_RANGE_SIM = HARVEST_DISTANCE_SIM * 2.5
        DISCOVERY_REWARD_FACTOR = 0.05

    # --- Reinforcement Learning Configuration ---
    class RL:
        """Configuration for the Reinforcement Learning agent(s).

        Covers reward shaping, training hyperparameters, and observation/action space definitions.

        Attributes:
            SUSTAINED_MINING_REWARD_PER_STEP (float): Reward per step for continuous mining.
            HIGH_ENERGY_THRESHOLD_PERCENT (float): Energy level (as % of max) above which
                                                   a high energy bonus is applied.
            HIGH_ENERGY_REWARD_BONUS (float): Bonus reward for maintaining high energy.
            TARGET_PROXIMITY_REWARD_FACTOR (float): Multiplier for reward based on proximity to target.
            MOVE_AWAY_FROM_TARGET_PENALTY_FACTOR (float): Penalty for moving away from the current target.
            REACH_TARGET_BONUS (float): One-time bonus for reaching a designated target.
            PROXIMITY_REWARD_FALLOFF_SIM (float): Distance (sim units) over which proximity reward diminishes.
            TARGET_SWITCH_ENERGY_COST (float): Energy cost incurred when a probe switches its target.
            TARGET_SWITCH_COOLDOWN_STEPS (int): Minimum steps before a probe can switch targets again.
            CRITICAL_ENERGY_THRESHOLD_PERCENT (float): Energy level (as % of max) below which
                                                       critical low energy penalties apply.
            STAY_ALIVE_REWARD_BONUS_PER_STEP (float): Small reward per step for simply remaining active.
            LOW_ENERGY_PENALTY_LEVEL_1_THRESHOLD_PERCENT (float): First threshold for low energy penalty.
            LOW_ENERGY_PENALTY_LEVEL_1_FACTOR (float): Penalty factor for energy below level 1 threshold.
            LOW_ENERGY_PENALTY_LEVEL_2_THRESHOLD_PERCENT (float): Second (lower) threshold for low energy penalty.
            LOW_ENERGY_PENALTY_LEVEL_2_FACTOR (float): Penalty factor for energy below level 2 threshold.
            EPISODE_LENGTH_STEPS (int): Maximum number of steps per training episode.
            LEARNING_RATE (float): Learning rate for the RL optimizer.
            BATCH_SIZE (int): Batch size for training updates.
            NUM_OBSERVED_RESOURCES_FOR_TARGETING (int): How many nearby resource nodes are
                                                        included in the observation for targeting.
            OBSERVATION_SPACE_SIZE (int): Dimensionality of the agent's observation vector.
                                          (Note: Review if this should be dynamically calculated).
            ACTION_SPACE_DIMS (List[int]): Dimensions for the discrete action space,
                                           dynamically calculated in `SimulationConfig.__init__`.
                                           Order: [thrust_options, torque_options, communicate,
                                           replicate, target_selection].
        """
        SUSTAINED_MINING_REWARD_PER_STEP = 0.05
        HIGH_ENERGY_THRESHOLD_PERCENT = 0.75
        HIGH_ENERGY_REWARD_BONUS = 0.1
        TARGET_PROXIMITY_REWARD_FACTOR = 1.95
        MOVE_AWAY_FROM_TARGET_PENALTY_FACTOR = 0.5
        REACH_TARGET_BONUS = 2.0
        PROXIMITY_REWARD_FALLOFF_SIM = 5.0
        TARGET_SWITCH_ENERGY_COST = 1.0
        TARGET_SWITCH_COOLDOWN_STEPS = 20
        
        CRITICAL_ENERGY_THRESHOLD_PERCENT = 0.10
        STAY_ALIVE_REWARD_BONUS_PER_STEP = 0.02
        LOW_ENERGY_PENALTY_LEVEL_1_THRESHOLD_PERCENT = 0.25
        LOW_ENERGY_PENALTY_LEVEL_1_FACTOR = 0.1
        LOW_ENERGY_PENALTY_LEVEL_2_THRESHOLD_PERCENT = 0.10
        LOW_ENERGY_PENALTY_LEVEL_2_FACTOR = 0.3

        EPISODE_LENGTH_STEPS = 50000
        LEARNING_RATE = 3e-4
        BATCH_SIZE = 64

        NUM_OBSERVED_RESOURCES_FOR_TARGETING = 3
        OBSERVATION_SPACE_SIZE = 25 # Placeholder, ensure this matches actual observation vector
        ACTION_SPACE_DIMS = []

    # --- Solar System Data ---
    class SolarSystem:
        """Configuration for celestial bodies and orbital mechanics.

        Attributes:
            REFERENCE_EPOCH_JD (float): Julian Date for the J2000.0 epoch, used as the
                                        reference time for orbital element calculations.
            SUN_MASS_KG (float): Mass of the Sun in kilograms. This is the primary mass
                                 for the system's gravitational calculations.
            PLANET_DATA (Dict[str, Dict]): A dictionary where keys are celestial body names
                                           (e.g., "Sun", "Earth", "Moon") and values are
                                           dictionaries containing their physical and orbital
                                           parameters (mass_kg, radius_km, display_radius_sim,
                                           color, semi_major_axis_au, eccentricity, etc.,
                                           and 'central_body' name for orbiting bodies).
        """
        REFERENCE_EPOCH_JD = 2451545.0
        SUN_MASS_KG = 1.9885e30
        
        PLANET_DATA = {
            'Sun': {
                'mass_kg': SUN_MASS_KG, # Will be overwritten by SUN_MASS_KG in __init__
                'radius_km': 695700.0,
                'display_radius_sim': 500,
                'color': (255, 255, 100),
                'semi_major_axis_au': 0.0, 'eccentricity': 0.0, 'inclination_deg': 0.0,
                'longitude_of_ascending_node_deg': 0.0, 'argument_of_perihelion_deg': 0.0,
                'mean_anomaly_at_epoch_deg': 0.0, 'central_body': None
            },
            'Mercury': {
                'mass_kg': 0.33011e24, 'radius_km': 2439.7, 'display_radius_sim': 30, 'color': (169, 169, 169),
                'semi_major_axis_au': 0.387098, 'eccentricity': 0.205630, 'inclination_deg': 7.005,
                'longitude_of_ascending_node_deg': 48.331, 'argument_of_perihelion_deg': 29.124,
                'mean_anomaly_at_epoch_deg': 174.794, 'central_body': 'Sun'
            },
            'Venus': {
                'mass_kg': 4.8675e24, 'radius_km': 6051.8, 'display_radius_sim': 50, 'color': (255, 198, 73),
                'semi_major_axis_au': 0.723332, 'eccentricity': 0.006772, 'inclination_deg': 3.39458,
                'longitude_of_ascending_node_deg': 76.680, 'argument_of_perihelion_deg': 54.884,
                'mean_anomaly_at_epoch_deg': 50.447, 'central_body': 'Sun'
            },
            'Earth': {
                'mass_kg': 5.97237e24, 'radius_km': 6371.0, 'display_radius_sim': 50, 'color': (100, 149, 237),
                'semi_major_axis_au': 1.00000261, 'eccentricity': 0.01671123, 'inclination_deg': 0.00005,
                'longitude_of_ascending_node_deg': -11.26064, 'argument_of_perihelion_deg': 114.20783,
                'mean_anomaly_at_epoch_deg': 357.51716, 'central_body': 'Sun'
            },
            'Moon': {
                'mass_kg': 0.07346e24, 'radius_km': 1737.4, 'display_radius_sim': 15, 'color': (200, 200, 200),
                'semi_major_axis_au': 0.00257, 'eccentricity': 0.0549, 'inclination_deg': 5.145,
                'longitude_of_ascending_node_deg': 125.08, 'argument_of_perihelion_deg': 318.15,
                'mean_anomaly_at_epoch_deg': 115.36, 'central_body': 'Earth'
            },
            'Mars': {
                'mass_kg': 0.64171e24, 'radius_km': 3389.5, 'display_radius_sim': 40, 'color': (193, 68, 14),
                'semi_major_axis_au': 1.523679, 'eccentricity': 0.09340, 'inclination_deg': 1.850,
                'longitude_of_ascending_node_deg': 49.558, 'argument_of_perihelion_deg': 286.502,
                'mean_anomaly_at_epoch_deg': 19.412, 'central_body': 'Sun'
            },
            'Jupiter': {
                'mass_kg': 1898.19e24, 'radius_km': 69911.0, 'display_radius_sim': 200, 'color': (200, 160, 120),
                'semi_major_axis_au': 5.2044, 'eccentricity': 0.0489, 'inclination_deg': 1.303,
                'longitude_of_ascending_node_deg': 100.464, 'argument_of_perihelion_deg': 273.867,
                'mean_anomaly_at_epoch_deg': 20.020, 'central_body': 'Sun'
            },
            'Io': {
                'mass_kg': 0.089319e24, 'radius_km': 1821.6, 'display_radius_sim': 20, 'color': (255, 255, 150),
                'semi_major_axis_au': 0.002819, 'eccentricity': 0.0041, 'inclination_deg': 0.050,
                'longitude_of_ascending_node_deg': 0.0, 'argument_of_perihelion_deg': 0.0,
                'mean_anomaly_at_epoch_deg': 0.0, 'central_body': 'Jupiter'
            },
            'Europa': {
                'mass_kg': 0.04800e24, 'radius_km': 1560.8, 'display_radius_sim': 18, 'color': (200, 200, 255),
                'semi_major_axis_au': 0.004486, 'eccentricity': 0.0094, 'inclination_deg': 0.470,
                'longitude_of_ascending_node_deg': 0.0, 'argument_of_perihelion_deg': 0.0,
                'mean_anomaly_at_epoch_deg': 100.0, 'central_body': 'Jupiter'
            },
            'Ganymede': {
                'mass_kg': 0.14819e24, 'radius_km': 2634.1, 'display_radius_sim': 25, 'color': (160, 160, 180),
                'semi_major_axis_au': 0.007155, 'eccentricity': 0.0013, 'inclination_deg': 0.204,
                'longitude_of_ascending_node_deg': 0.0, 'argument_of_perihelion_deg': 0.0,
                'mean_anomaly_at_epoch_deg': 200.0, 'central_body': 'Jupiter'
            },
            'Callisto': {
                'mass_kg': 0.10759e24, 'radius_km': 2410.3, 'display_radius_sim': 22, 'color': (100, 80, 70),
                'semi_major_axis_au': 0.012585, 'eccentricity': 0.0074, 'inclination_deg': 0.205,
                'longitude_of_ascending_node_deg': 0.0, 'argument_of_perihelion_deg': 0.0,
                'mean_anomaly_at_epoch_deg': 300.0, 'central_body': 'Jupiter'
            },
            'Saturn': {
                'mass_kg': 568.34e24, 'radius_km': 58232.0, 'display_radius_sim': 180, 'color': (234, 214, 184),
                'semi_major_axis_au': 9.5826, 'eccentricity': 0.0565, 'inclination_deg': 2.485,
                'longitude_of_ascending_node_deg': 113.665, 'argument_of_perihelion_deg': 339.392,
                'mean_anomaly_at_epoch_deg': 317.020, 'central_body': 'Sun'
            },
            'Titan': {
                'mass_kg': 0.13452e24, 'radius_km': 2574.7, 'display_radius_sim': 24, 'color': (240, 190, 100),
                'semi_major_axis_au': 0.008168, 'eccentricity': 0.0288, 'inclination_deg': 0.34854,
                'longitude_of_ascending_node_deg': 0.0, 'argument_of_perihelion_deg': 0.0,
                'mean_anomaly_at_epoch_deg': 0.0, 'central_body': 'Saturn'
            },
            'Uranus': {
                'mass_kg': 86.813e24, 'radius_km': 25362.0, 'display_radius_sim': 120, 'color': (155, 221, 221),
                'semi_major_axis_au': 19.2184, 'eccentricity': 0.0457, 'inclination_deg': 0.772,
                'longitude_of_ascending_node_deg': 74.006, 'argument_of_perihelion_deg': 96.999,
                'mean_anomaly_at_epoch_deg': 142.238600, 'central_body': 'Sun'
            },
            'Neptune': {
                'mass_kg': 102.413e24, 'radius_km': 24622.0, 'display_radius_sim': 110, 'color': (63, 81, 181),
                'semi_major_axis_au': 30.110, 'eccentricity': 0.0113, 'inclination_deg': 1.770,
                'longitude_of_ascending_node_deg': 131.783, 'argument_of_perihelion_deg': 276.336,
                'mean_anomaly_at_epoch_deg': 256.228, 'central_body': 'Sun'
            }
        }

    # --- Visualization Configuration ---
    class Visualization:
        """Configuration for the simulation's graphical visualization.

        Attributes:
            SCREEN_WIDTH_PX (int): Width of the display window in pixels.
            SCREEN_HEIGHT_PX (int): Height of the display window in pixels.
            FPS (int): Target frames per second for rendering.
            PROBE_SIZE_PX (int): Visual size (diameter) of probes in pixels.
            SHOW_SIMPLE_PROBE_INFO (bool): If True, display basic info text next to probes.
            SHOW_SIMPLE_RESOURCE_INFO (bool): If True, display basic info for resources.
            ORGANIC_SHIP_ENABLED (bool): Toggles the organic ship visual theme.
            SHIP_SCALE_FACTOR (float): General scaling factor for organic ship visuals.
            ELEGANT_PARTICLE_MODE (bool): Toggles a specific alternative particle style.
            ENABLE_PARTICLE_EFFECTS (bool): Master toggle for all particle effects (e.g., exhaust).
            METALLIC_SHEEN_INTENSITY (float): Intensity of the metallic sheen effect on visuals.
            ORGANIC_CURVE_RESOLUTION (int): Number of segments used to render organic curves.
            ENGINE_GLOW_INTENSITY (float): Intensity of the engine glow effect.
            ORGANIC_EXHAUST_PARTICLE_SCALE (float): Scale factor for organic exhaust particles.
            MAX_ORBIT_PATH_POINTS (int): Max points to store for drawing celestial body orbit trails.
            MAX_PROBE_TRAIL_POINTS (int): Max points to store for drawing probe movement trails.
            MAX_PARTICLES_PER_SYSTEM (int): Max particles for any single `AdvancedParticleSystem`.
            PROBE_DEFAULT_COLOR (Tuple[int,int,int]): Default RGB color for probes.
        """
        SCREEN_WIDTH_PX = 1400
        SCREEN_HEIGHT_PX = 900
        FPS = 60
        PROBE_SIZE_PX = 12
        SHOW_SIMPLE_PROBE_INFO = True
        SHOW_SIMPLE_RESOURCE_INFO = True

        ORGANIC_SHIP_ENABLED = True
        SHIP_SCALE_FACTOR = 1.2
        ELEGANT_PARTICLE_MODE = False
        ENABLE_PARTICLE_EFFECTS = False

        METALLIC_SHEEN_INTENSITY = 0.7
        ORGANIC_CURVE_RESOLUTION = 20
        ENGINE_GLOW_INTENSITY = 1.2
        ORGANIC_EXHAUST_PARTICLE_SCALE = 0.1
        MAX_ORBIT_PATH_POINTS = 1000
        MAX_PROBE_TRAIL_POINTS = 500
        MAX_PARTICLES_PER_SYSTEM = 2000
        PROBE_DEFAULT_COLOR = (255, 0, 255) # Added for completeness

    # --- Monitoring Configuration ---
    class Monitoring:
        """Configuration for system resource monitoring.

        Attributes:
            MEMORY_USAGE_WARN_MB (int): Memory usage threshold in Megabytes. If exceeded,
                                        a warning is logged.
            MEMORY_CHECK_INTERVAL_STEPS (int): Frequency (in simulation steps) at which
                                               memory usage is checked.
        """
        MEMORY_USAGE_WARN_MB = 2048
        MEMORY_CHECK_INTERVAL_STEPS = 500

    # --- Debug Configuration ---
    class Debug:
        """Configuration for debugging features and logging verbosity.

        Attributes:
            DEBUG_MODE (bool): Master toggle for enabling/disabling all debug features.
            ENERGY_RESET_VALUE (float): Value to reset probe energy to when a debug
                                        reset command is issued.
            ORBITAL_MECHANICS (bool): Toggle for verbose logging from orbital mechanics calculations.
            KEPLER_SOLVER (bool): Toggle for verbose logging from Kepler's equation solver.
            INITIAL_STATE_VECTOR (bool): Toggle for logging details of initial state vector calculations.
            GRAVITY_CALCULATION (bool): Toggle for logging details of gravity calculations.
            VERLET_INTEGRATION (bool): Toggle for logging details of Verlet integration steps.
            CONFIG_VALIDATION (bool): If True, prints detailed validation messages on startup.
            UI_DEBUG (bool): Toggle for debugging UI elements (e.g., button states, mouse clicks).
            MONITOR_ENERGY_CONSERVATION (bool): If True, periodically logs the total energy
                                                of the celestial system to check for conservation.
            ENERGY_CHECK_INTERVAL_STEPS (int): Frequency (sim steps) for energy conservation checks.
            LOG_ORBIT_INTERVAL_STEPS (int): Frequency (sim steps) for logging positions of selected bodies.
            LOG_ORBIT_BODY_NAMES (List[str]): Names of celestial bodies whose orbits to log.
        """
        DEBUG_MODE = False
        ENERGY_RESET_VALUE = 50.0
        ORBITAL_MECHANICS = True
        KEPLER_SOLVER = False
        INITIAL_STATE_VECTOR = False
        GRAVITY_CALCULATION = False
        VERLET_INTEGRATION = False
        CONFIG_VALIDATION = True
        UI_DEBUG = False
        MONITOR_ENERGY_CONSERVATION = True
        ENERGY_CHECK_INTERVAL_STEPS = 100
        LOG_ORBIT_INTERVAL_STEPS = 100 # Added for consistency with usage
        LOG_ORBIT_BODY_NAMES = ["Earth", "Sun", "Moon"] # Added for consistency

    # --- Spatial Partitioning Configuration ---
    class SpatialPartitioning:
        """Configuration for spatial partitioning structures (e.g., Quadtree).

        Used to optimize collision detection and neighbor searches.

        Attributes:
            QUADTREE_CAPACITY (int): Maximum number of entities a Quadtree node can hold
                                     before it subdivides.
            QUADTREE_MAX_DEPTH (int): Maximum depth the Quadtree is allowed to reach.
                                      Prevents excessively deep trees.
        """
        QUADTREE_CAPACITY = 4
        QUADTREE_MAX_DEPTH = 8

    def __init__(self):
        """Initializes the `SimulationConfig` instance and performs setup.

        This constructor performs several key tasks:
        1.  **Calculates Derived RL Action Space Dimensions**:
            Determines the dimensions of the Reinforcement Learning (RL) agent's
            discrete action space based on `SimulationConfig.Probe` settings
            (e.g., number of thrust/torque levels) and
            `SimulationConfig.RL.NUM_OBSERVED_RESOURCES_FOR_TARGETING`.
            The `RL.ACTION_SPACE_DIMS` attribute is populated with a list:
            `[num_thrust_options, num_torque_options, communicate_binary,
            replicate_binary, target_resource_options]`.

        2.  **Ensures Sun's Mass Consistency**:
            Updates the Sun's mass within `SolarSystem.PLANET_DATA` to match the
            primary `SolarSystem.SUN_MASS_KG` attribute, ensuring a single source of
            truth for this critical parameter.

        3.  **Invokes Configuration Validation**:
            Calls `self.validate()` to perform a comprehensive check of all
            configuration settings. This is crucial for catching errors early.

        Raises:
            ConfigurationError: If `self.validate()` detects any issues with the
                                configuration values, preventing the simulation from
                                starting with an invalid or inconsistent state.
        """
        # --- Calculate dependent RL Action Space Dimensions ---
        num_thrust_options = len(self.Probe.THRUST_FORCE_MAGNITUDES)

        # Calculate torque options: 1 (no torque) + 2 for each non-zero magnitude (positive/negative)
        num_torque_magnitudes = len(self.Probe.TORQUE_MAGNITUDES)
        torque_options_count = 1 # For zero torque
        if num_torque_magnitudes > 0:
            non_zero_torque_mags = [m for m in self.Probe.TORQUE_MAGNITUDES if m != 0.0]
            if non_zero_torque_mags:
                 torque_options_count = 1 + (2 * len(non_zero_torque_mags)) # 0, +m1, -m1, +m2, -m2 ...

        self.RL.ACTION_SPACE_DIMS = [
            num_thrust_options,  # e.g., [0, 0.1, 0.5, 1.0] -> 4 options
            torque_options_count, # e.g., [0, 0.01, 0.05] -> 1 (0) + 2*2 (non-zero) = 5 options
            2,  # Communicate action (0: No, 1: Yes)
            2,  # Replicate action (0: No, 1: Yes)
            self.RL.NUM_OBSERVED_RESOURCES_FOR_TARGETING + 1  # Target: N resources + 1 (no target/clear target)
        ]
        
        # Ensure Sun's mass in PLANET_DATA is consistent with the main SUN_MASS_KG
        if 'Sun' in self.SolarSystem.PLANET_DATA:
            self.SolarSystem.PLANET_DATA['Sun']['mass_kg'] = self.SolarSystem.SUN_MASS_KG

        self.validate() # Perform validation after all initial setup

    def validate(self):
        """Performs comprehensive validation of all simulation configuration settings.

        This method systematically checks various aspects of the configuration:
        -   **Global Scales**: Ensures `AU_SCALE` and derived `KM_SCALE` are positive.
        -   **World**: Validates `SIZE_AU`, dimensions, and asteroid belt parameters
            (e.g., inner radius < outer radius, within world bounds).
        -   **Physics**: Checks `TIMESTEP_SECONDS` and `DEFAULT_PROBE_MASS_KG` are positive.
        -   **Time**: Verifies `SIM_SECONDS_PER_STEP` consistency with `Physics.TIMESTEP_SECONDS`.
        -   **Probe**: Validates probe counts, energy levels (initial, max, replication costs
            are logical), thrust/torque magnitudes (non-negative).
        -   **RL**: Ensures energy thresholds are valid percentages (0-1) and logically ordered
            (e.g., critical < low < high). Checks `EPISODE_LENGTH_STEPS`.
        -   **Visualization**: Validates screen dimensions, FPS, and limits for
            path points/particles are positive.
        -   **SolarSystem**: Checks `SUN_MASS_KG`, presence and mass consistency of 'Sun'
            in `PLANET_DATA`. For all celestial bodies, validates mass, radius,
            orbital elements (SMA > 0 for non-Sun, 0 <= eccentricity < 1,
            0 <= inclination <= 180), and ensures `central_body` references are valid
            and not self-referential.
        -   **SpatialPartitioning**: Ensures `QUADTREE_CAPACITY` and `QUADTREE_MAX_DEPTH`
            are positive.

        If any validation check fails, a `ConfigurationError` is raised with a
        descriptive message, detailing the problematic setting. This prevents the
        simulation from starting with an invalid, inconsistent, or potentially
        nonsensical configuration. If all checks pass, an informational message
        is logged.

        Raises:
            ConfigurationError: If any configuration setting is found to be invalid.
        """
        # Global scale checks
        if AU_SCALE <= 0:
            raise ConfigurationError("Global AU_SCALE must be positive.")
        if KM_SCALE <= 0: # Derived, but check for sanity
            raise ConfigurationError("Global KM_SCALE must be positive (derived from AU_SCALE and AU_KM).")

        # World validation
        if self.World.SIZE_AU <= 0:
            raise ConfigurationError("World.SIZE_AU must be positive.")
        if self.World.WIDTH_SIM <= 0 or self.World.HEIGHT_SIM <= 0:
            raise ConfigurationError("World.WIDTH_SIM and World.HEIGHT_SIM must be positive (derived from SIZE_AU).")
        if not (0 < self.World.ASTEROID_BELT_INNER_AU < self.World.ASTEROID_BELT_OUTER_AU < self.World.SIZE_AU):
            raise ConfigurationError(
                f"Asteroid belt AUs (Inner: {self.World.ASTEROID_BELT_INNER_AU}, Outer: {self.World.ASTEROID_BELT_OUTER_AU}) "
                f"must be positive, ordered correctly, and within world size ({self.World.SIZE_AU})."
            )

        # Physics validation
        if self.Physics.TIMESTEP_SECONDS <= 0:
            raise ConfigurationError("Physics.TIMESTEP_SECONDS must be positive.")
        if self.Physics.DEFAULT_PROBE_MASS_KG <= 0:
            raise ConfigurationError("Physics.DEFAULT_PROBE_MASS_KG must be positive.")

        # Time validation
        if self.Time.SIM_SECONDS_PER_STEP != self.Physics.TIMESTEP_SECONDS:
            logging.warning(
                f"Time.SIM_SECONDS_PER_STEP ({self.Time.SIM_SECONDS_PER_STEP}) "
                f"differs from Physics.TIMESTEP_SECONDS ({self.Physics.TIMESTEP_SECONDS}). "
                "Ensure this is intentional, as they are often expected to match."
            )
            # Not raising an error for now, as they might be decoupled in some scenarios.

        # Probe validation
        if not (0 < self.Probe.INITIAL_PROBES <= self.Probe.MAX_PROBES):
            raise ConfigurationError(
                f"Probe counts invalid: INITIAL_PROBES ({self.Probe.INITIAL_PROBES}) "
                f"must be > 0 and <= MAX_PROBES ({self.Probe.MAX_PROBES})."
            )
        if self.Probe.MAX_ENERGY <= 0:
            raise ConfigurationError("Probe.MAX_ENERGY must be positive.")
        if not (0 <= self.Probe.INITIAL_ENERGY <= self.Probe.MAX_ENERGY):
            raise ConfigurationError(
                f"Probe.INITIAL_ENERGY ({self.Probe.INITIAL_ENERGY}) "
                f"must be between 0 and MAX_ENERGY ({self.Probe.MAX_ENERGY})."
            )
        if not (0 < self.Probe.REPLICATION_COST <= self.Probe.MAX_ENERGY):
            raise ConfigurationError(
                f"Probe.REPLICATION_COST ({self.Probe.REPLICATION_COST}) "
                f"must be positive and not exceed MAX_ENERGY ({self.Probe.MAX_ENERGY})."
            )
        if not (self.Probe.REPLICATION_COST <= self.Probe.REPLICATION_MIN_ENERGY <= self.Probe.MAX_ENERGY):
             raise ConfigurationError(
                f"Probe.REPLICATION_MIN_ENERGY ({self.Probe.REPLICATION_MIN_ENERGY}) must be between "
                f"REPLICATION_COST ({self.Probe.REPLICATION_COST}) and MAX_ENERGY ({self.Probe.MAX_ENERGY})."
            )
        if not all(t >= 0 for t in self.Probe.THRUST_FORCE_MAGNITUDES):
            raise ConfigurationError("All Probe.THRUST_FORCE_MAGNITUDES must be non-negative.")
        if 0.0 not in self.Probe.THRUST_FORCE_MAGNITUDES:
            raise ConfigurationError("Probe.THRUST_FORCE_MAGNITUDES must include 0.0 for no thrust option.")
        if not all(t >= 0 for t in self.Probe.TORQUE_MAGNITUDES):
            raise ConfigurationError("All Probe.TORQUE_MAGNITUDES must be non-negative.")
        if 0.0 not in self.Probe.TORQUE_MAGNITUDES:
            raise ConfigurationError("Probe.TORQUE_MAGNITUDES must include 0.0 for no torque option.")


        # RL validation (ensure percentages are valid and logically ordered)
        energy_threshold_names = [
            "HIGH_ENERGY_THRESHOLD_PERCENT", "CRITICAL_ENERGY_THRESHOLD_PERCENT",
            "LOW_ENERGY_PENALTY_LEVEL_1_THRESHOLD_PERCENT", "LOW_ENERGY_PENALTY_LEVEL_2_THRESHOLD_PERCENT"
        ]
        for name in energy_threshold_names:
            threshold = getattr(self.RL, name)
            if not (0.0 <= threshold <= 1.0):
                raise ConfigurationError(f"RL.{name} ({threshold}) must be a percentage between 0.0 and 1.0.")
        
        if not (self.RL.CRITICAL_ENERGY_THRESHOLD_PERCENT < self.RL.LOW_ENERGY_PENALTY_LEVEL_2_THRESHOLD_PERCENT <= \
                self.RL.LOW_ENERGY_PENALTY_LEVEL_1_THRESHOLD_PERCENT < self.RL.HIGH_ENERGY_THRESHOLD_PERCENT):
             raise ConfigurationError(
                 "RL energy thresholds are not logically ordered. Expected: \n"
                 "CRITICAL_ENERGY_THRESHOLD_PERCENT < LOW_ENERGY_PENALTY_LEVEL_2_THRESHOLD_PERCENT "
                 "<= LOW_ENERGY_PENALTY_LEVEL_1_THRESHOLD_PERCENT < HIGH_ENERGY_THRESHOLD_PERCENT.\n"
                 f"Current Values: Crit={self.RL.CRITICAL_ENERGY_THRESHOLD_PERCENT}, "
                 f"LowL2={self.RL.LOW_ENERGY_PENALTY_LEVEL_2_THRESHOLD_PERCENT}, "
                 f"LowL1={self.RL.LOW_ENERGY_PENALTY_LEVEL_1_THRESHOLD_PERCENT}, "
                 f"High={self.RL.HIGH_ENERGY_THRESHOLD_PERCENT}"
             )
        if self.RL.EPISODE_LENGTH_STEPS <= 0:
            raise ConfigurationError("RL.EPISODE_LENGTH_STEPS must be positive.")


        # Visualization
        if self.Visualization.SCREEN_WIDTH_PX <=0 or self.Visualization.SCREEN_HEIGHT_PX <=0:
            raise ConfigurationError("Visualization screen dimensions (SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX) must be positive.")
        if self.Visualization.FPS <=0:
            raise ConfigurationError("Visualization.FPS must be positive.")
        if not (self.Visualization.MAX_ORBIT_PATH_POINTS > 0 and \
                self.Visualization.MAX_PROBE_TRAIL_POINTS > 0 and \
                self.Visualization.MAX_PARTICLES_PER_SYSTEM > 0):
            raise ConfigurationError("Visualization max points/particles settings (MAX_ORBIT_PATH_POINTS, MAX_PROBE_TRAIL_POINTS, MAX_PARTICLES_PER_SYSTEM) must be positive.")

        # Solar System Data Validation
        if self.SolarSystem.SUN_MASS_KG <= 0:
            raise ConfigurationError("SolarSystem.SUN_MASS_KG must be positive.")
        if 'Sun' not in self.SolarSystem.PLANET_DATA or \
           self.SolarSystem.PLANET_DATA['Sun'].get('mass_kg') != self.SolarSystem.SUN_MASS_KG:
            raise ConfigurationError(
                "Sun data missing in PLANET_DATA or Sun's mass in PLANET_DATA "
                "does not match SolarSystem.SUN_MASS_KG after __init__ update."
            )

        for name, data in self.SolarSystem.PLANET_DATA.items():
            if data.get('mass_kg', -1.0) < 0: # Use .get for safety before full validation
                raise ConfigurationError(f"Mass of celestial body '{name}' cannot be negative.")
            if data.get('radius_km', -1.0) < 0:
                raise ConfigurationError(f"Radius of celestial body '{name}' cannot be negative.")
            if data.get('semi_major_axis_au', 0.0) < 0 and name != 'Sun': # Sun SMA is 0
                 raise ConfigurationError(f"Semi-major axis of celestial body '{name}' cannot be negative.")
            if not (0.0 <= data.get('eccentricity', 0.0) < 1.0): # Eccentricity [0, 1)
                 raise ConfigurationError(f"Eccentricity of celestial body '{name}' ({data.get('eccentricity', 0.0)}) must be >= 0 and < 1.")
            if not (0.0 <= data.get('inclination_deg', 0.0) <= 180.0): # Inclination [0, 180]
                raise ConfigurationError(f"Inclination of '{name}' ({data.get('inclination_deg', 0.0)}) must be between 0 and 180 degrees inclusive.")

            central_body_name = data.get('central_body')
            if central_body_name is not None: # If it orbits something
                if central_body_name not in self.SolarSystem.PLANET_DATA:
                    raise ConfigurationError(f"Central body '{central_body_name}' for '{name}' not found in PLANET_DATA.")
                if central_body_name == name:
                    raise ConfigurationError(f"Celestial body '{name}' cannot orbit itself.")
            elif name != 'Sun': # If not Sun and central_body is None, it's an issue
                raise ConfigurationError(f"Celestial body '{name}' (which is not the Sun) must have a 'central_body' defined.")


        # Spatial Partitioning
        if self.SpatialPartitioning.QUADTREE_CAPACITY <= 0:
            raise ConfigurationError("SpatialPartitioning.QUADTREE_CAPACITY must be positive.")
        if self.SpatialPartitioning.QUADTREE_MAX_DEPTH <= 0:
            raise ConfigurationError("SpatialPartitioning.QUADTREE_MAX_DEPTH must be positive.")
            
        logging.info("Configuration validated successfully.")


# --- Instantiate the configuration ---
# This makes the config object available for import and runs validation.
# e.g., from config import config
try:
    config = SimulationConfig()
except ConfigurationError as e:
    logging.error(f"FATAL CONFIGURATION ERROR: {e}", exc_info=True)
    raise