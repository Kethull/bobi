# config.py
import numpy as np

# Fundamental Physical Constants (used across different config sections)
AU_KM = 149597870.7  # Astronomical Unit in kilometers
GRAVITATIONAL_CONSTANT_KM3_KG_S2 = 6.67430e-20  # G in km^3 kg^-1 s^-2
SECONDS_PER_DAY = 86400.0

# Simulation Scale Constants (also fundamental for conversions)
AU_SCALE = 10000.0  # Sim units per AU
KM_SCALE = AU_SCALE / AU_KM  # Sim units per km

class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass

class SimulationConfig:
    """
    Hierarchical configuration for the Bobiverse Orbital Simulation.
    All constants are organized into nested classes.
    An instance 'config' is created at the end of this file for easy import.
    """

    # --- World Configuration ---
    class World:
        SIZE_AU = 10.0  # Diameter of the simulation area in AU
        SIZE_SIM = SIZE_AU * AU_SCALE # Diameter of the simulation area in simulation units
        WIDTH_SIM = SIZE_SIM # World width in simulation units
        HEIGHT_SIM = SIZE_SIM # World height in simulation units
        CENTER_SIM = [WIDTH_SIM / 2, HEIGHT_SIM / 2] # Center of the world in simulation units

        # Asteroid Belt Configuration
        ASTEROID_BELT_INNER_AU = 2.2
        ASTEROID_BELT_OUTER_AU = 3.2
        ASTEROID_COUNT = 500
        ASTEROID_MIN_RADIUS_SIM = 0.2 # Min display radius in sim units
        ASTEROID_MAX_RADIUS_SIM = 0.8 # Max display radius in sim units
        ASTEROID_DEFAULT_COLOR = (100, 100, 100)
        ASTEROID_MASS_KG_MIN = 1e10 # Minimum mass in kg
        ASTEROID_MASS_KG_MAX = 1e15 # Maximum mass in kg

    # --- Physics Configuration ---
    class Physics:
        TIMESTEP_SECONDS = 3600.0  # Duration of one simulation step in real-world seconds
        INTEGRATION_METHOD = "verlet" # Preferred integration method: "euler", "verlet", "leapfrog"
        DEFAULT_PROBE_MASS_KG = 8.0   # Default mass of a probe in kg

    # --- Time Configuration ---
    class Time:
        # SIM_SECONDS_PER_STEP is effectively Physics.TIMESTEP_SECONDS
        # Kept separate if other time-specific global settings are needed later.
        SIM_SECONDS_PER_STEP = 3600.0

    # --- Probe Configuration ---
    class Probe:
        MAX_PROBES = 20 # Maximum number of probes allowed in the simulation
        INITIAL_PROBES = 1 # Number of probes at the start of the simulation

        MAX_ENERGY = 100000.0
        INITIAL_ENERGY = 90000.0
        REPLICATION_COST = 80000.0
        REPLICATION_MIN_ENERGY = 99900.0 # Minimum energy required to consider replication
        MAX_VELOCITY_SIM_PER_STEP = 10000.0 # Max speed in simulation units per step
        
        THRUST_FORCE_MAGNITUDES = [0.0, 0.08, 0.18, 0.32] # Available thrust magnitudes
        THRUST_ENERGY_COST_FACTOR = 0.001 # Energy cost per unit of thrust per step
        ENERGY_DECAY_RATE_PER_STEP = 0.001 # Passive energy decay per step
        LOW_POWER_PENALTY_FACTOR = 0.9 # Performance penalty factor when energy is low

        # Rotational Physics for Probes
        MOMENT_OF_INERTIA = 5.0 # kg*m^2 (or equivalent in sim units)
        TORQUE_MAGNITUDES = [0.0, 0.008, 0.018] # Available torque magnitudes (0.0 is no torque)
        ROTATIONAL_ENERGY_COST_FACTOR = 0.11 # Energy cost per unit of torque per step
        MAX_ANGULAR_VELOCITY_RAD_PER_STEP = np.pi / 4 # Max angular speed in radians per step
        ANGULAR_DAMPING_FACTOR = 0.05 # Passive angular velocity decay

        # Smoothing Parameters for Probe Actions
        ACTION_SMOOTHING_FACTOR = 0.85
        MIN_THRUST_DURATION_STEPS = 30
        THRUST_RAMP_TIME_STEPS = 60
        ROTATION_SMOOTHING_FACTOR = 0.9
        MIN_ROTATION_DURATION_STEPS = 6
        ROTATION_RAMP_TIME_STEPS = 4

        # Anti-spam parameters for Probe Actions
        THRUSTER_STARTUP_ENERGY_COST = 1.5
        RAPID_SWITCHING_PENALTY_FACTOR = 0.8 # Penalty for rapidly switching actions
        SWITCHING_DETECTION_WINDOW_STEPS = 10

        # Communication
        COMM_RANGE_SIM = 100.0 # Communication range in simulation units
        MESSAGE_TYPES = ['RESOURCE_LOCATION'] # Types of messages probes can send/receive

    # --- Resource Configuration ---
    class Resource:
        COUNT = 15 # Number of resource nodes
        MIN_AMOUNT = 10000
        MAX_AMOUNT = 20000
        REGEN_RATE_PER_STEP = 0.00 # Regeneration rate per step
        HARVEST_RATE_PER_STEP = 2.0 # Amount harvested per step
        HARVEST_DISTANCE_SIM = 5.0 # Max distance to harvest in simulation units
        DISCOVERY_RANGE_SIM = HARVEST_DISTANCE_SIM * 2.5 # Range to discover resources
        DISCOVERY_REWARD_FACTOR = 0.05

    # --- Reinforcement Learning Configuration ---
    class RL:
        # Reward Shaping Parameters
        SUSTAINED_MINING_REWARD_PER_STEP = 0.05
        HIGH_ENERGY_THRESHOLD_PERCENT = 0.75 # Percentage of MAX_ENERGY
        HIGH_ENERGY_REWARD_BONUS = 0.1
        TARGET_PROXIMITY_REWARD_FACTOR = 1.95
        MOVE_AWAY_FROM_TARGET_PENALTY_FACTOR = 0.5
        REACH_TARGET_BONUS = 2.0
        PROXIMITY_REWARD_FALLOFF_SIM = 5.0 # Falloff distance for proximity rewards
        TARGET_SWITCH_ENERGY_COST = 1.0
        TARGET_SWITCH_COOLDOWN_STEPS = 20
        
        CRITICAL_ENERGY_THRESHOLD_PERCENT = 0.10 # Percentage of MAX_ENERGY
        STAY_ALIVE_REWARD_BONUS_PER_STEP = 0.02
        LOW_ENERGY_PENALTY_LEVEL_1_THRESHOLD_PERCENT = 0.25
        LOW_ENERGY_PENALTY_LEVEL_1_FACTOR = 0.1
        LOW_ENERGY_PENALTY_LEVEL_2_THRESHOLD_PERCENT = 0.10 # This was 0.10 in original, should be > CRITICAL and < LEVEL_1 if ordered that way. Assuming original intent.
        LOW_ENERGY_PENALTY_LEVEL_2_FACTOR = 0.3


        # Training Parameters
        EPISODE_LENGTH_STEPS = 50000
        LEARNING_RATE = 3e-4
        BATCH_SIZE = 64

        # Agent Observation/Action Space related constants
        NUM_OBSERVED_RESOURCES_FOR_TARGETING = 3
        OBSERVATION_SPACE_SIZE = 25 # Placeholder, review based on actual observation components
        ACTION_SPACE_DIMS = [] # Dynamically populated in __init__

    # --- Solar System Data ---
    class SolarSystem:
        REFERENCE_EPOCH_JD = 2451545.0 # Julian Date for J2000.0
        SUN_MASS_KG = 1.9885e30  # Mass of the Sun in kg
        
        PLANET_DATA = {
            'Sun': {
                'mass_kg': SUN_MASS_KG, # Will be updated in __init__ to ensure it uses the class var
                'radius_km': 695700.0,
                'display_radius_sim': 500, # Visual radius in simulation units
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
            'Moon': { # Orbital elements are relative to Earth
                'mass_kg': 0.07346e24, 'radius_km': 1737.4, 'display_radius_sim': 15, 'color': (200, 200, 200),
                'semi_major_axis_au': 0.00257, 'eccentricity': 0.0549, 'inclination_deg': 5.145, # Relative to Earth's equatorial plane
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
            'Io': { # Orbits Jupiter
                'mass_kg': 0.089319e24, 'radius_km': 1821.6, 'display_radius_sim': 20, 'color': (255, 255, 150),
                'semi_major_axis_au': 0.002819, 'eccentricity': 0.0041, 'inclination_deg': 0.050, # Relative to Jupiter's equatorial plane
                'longitude_of_ascending_node_deg': 0.0, 'argument_of_perihelion_deg': 0.0, 
                'mean_anomaly_at_epoch_deg': 0.0, 'central_body': 'Jupiter' # Placeholder values for angles
            },
            'Europa': { # Orbits Jupiter
                'mass_kg': 0.04800e24, 'radius_km': 1560.8, 'display_radius_sim': 18, 'color': (200, 200, 255),
                'semi_major_axis_au': 0.004486, 'eccentricity': 0.0094, 'inclination_deg': 0.470,
                'longitude_of_ascending_node_deg': 0.0, 'argument_of_perihelion_deg': 0.0, 
                'mean_anomaly_at_epoch_deg': 100.0, 'central_body': 'Jupiter' # Placeholder values for angles
            },
            'Ganymede': { # Orbits Jupiter
                'mass_kg': 0.14819e24, 'radius_km': 2634.1, 'display_radius_sim': 25, 'color': (160, 160, 180),
                'semi_major_axis_au': 0.007155, 'eccentricity': 0.0013, 'inclination_deg': 0.204,
                'longitude_of_ascending_node_deg': 0.0, 'argument_of_perihelion_deg': 0.0, 
                'mean_anomaly_at_epoch_deg': 200.0, 'central_body': 'Jupiter' # Placeholder values for angles
            },
            'Callisto': { # Orbits Jupiter
                'mass_kg': 0.10759e24, 'radius_km': 2410.3, 'display_radius_sim': 22, 'color': (100, 80, 70),
                'semi_major_axis_au': 0.012585, 'eccentricity': 0.0074, 'inclination_deg': 0.205,
                'longitude_of_ascending_node_deg': 0.0, 'argument_of_perihelion_deg': 0.0, 
                'mean_anomaly_at_epoch_deg': 300.0, 'central_body': 'Jupiter' # Placeholder values for angles
            },
            'Saturn': {
                'mass_kg': 568.34e24, 'radius_km': 58232.0, 'display_radius_sim': 180, 'color': (234, 214, 184),
                'semi_major_axis_au': 9.5826, 'eccentricity': 0.0565, 'inclination_deg': 2.485,
                'longitude_of_ascending_node_deg': 113.665, 'argument_of_perihelion_deg': 339.392,
                'mean_anomaly_at_epoch_deg': 317.020, 'central_body': 'Sun'
            },
            'Titan': { # Orbits Saturn
                'mass_kg': 0.13452e24, 'radius_km': 2574.7, 'display_radius_sim': 24, 'color': (240, 190, 100),
                'semi_major_axis_au': 0.008168, 'eccentricity': 0.0288, 'inclination_deg': 0.34854, # Relative to Saturn's equatorial plane
                'longitude_of_ascending_node_deg': 0.0, 'argument_of_perihelion_deg': 0.0, 
                'mean_anomaly_at_epoch_deg': 0.0, 'central_body': 'Saturn' # Placeholder values for angles
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
        SCREEN_WIDTH_PX = 1400 # Screen width in pixels
        SCREEN_HEIGHT_PX = 900 # Screen height in pixels
        FPS = 60 # Target frames per second
        PROBE_SIZE_PX = 12 # Visual size of probes in pixels
        SHOW_SIMPLE_PROBE_INFO = True # Toggle for basic probe info display
        SHOW_SIMPLE_RESOURCE_INFO = True # Toggle for basic resource info display (if resources are used)

        # Organic Ship Visual Theme (if applicable)
        ORGANIC_SHIP_ENABLED = True
        SHIP_SCALE_FACTOR = 1.2 # General scaling for ship visuals
        ELEGANT_PARTICLE_MODE = False # Toggle for a specific particle style
        ENABLE_PARTICLE_EFFECTS = False # Master toggle for all particle effects

        # Enhanced visual settings (can be grouped under a sub-theme if grows)
        METALLIC_SHEEN_INTENSITY = 0.7
        ORGANIC_CURVE_RESOLUTION = 20 # For rendering organic shapes
        ENGINE_GLOW_INTENSITY = 1.2
        ORGANIC_EXHAUST_PARTICLE_SCALE = 0.1
        MAX_ORBIT_PATH_POINTS = 1000 # Maximum number of points to store for drawing orbit trails
        MAX_PROBE_TRAIL_POINTS = 500  # Maximum number of points for probe trails
        MAX_PARTICLES_PER_SYSTEM = 2000 # Max particles for a single AdvancedParticleSystem instance

    # --- Monitoring Configuration ---
    class Monitoring:
        MEMORY_USAGE_WARN_MB = 2048  # Warn if memory usage exceeds this (e.g., 2GB)
        MEMORY_CHECK_INTERVAL_STEPS = 500 # How often to check memory (in simulation steps)
        # ENERGY_CHECK_INTERVAL_STEPS is now part of Debug as it's a debug feature

    # --- Debug Configuration ---
    class Debug:
        DEBUG_MODE = False # Master debug toggle
        ENERGY_RESET_VALUE = 50.0 # Value to reset probe energy to in debug mode
        ORBITAL_MECHANICS = True # Specific toggle for debugging orbital mechanics
        # Add new debug flags here
        KEPLER_SOLVER = False    # Debug Kepler's equation solver
        INITIAL_STATE_VECTOR = False # Debug initial state vector calculation from orbital elements
        GRAVITY_CALCULATION = False # Debug gravitational acceleration calculations
        VERLET_INTEGRATION = False # Debug Verlet integration steps
        CONFIG_VALIDATION = True # Enable configuration validation prints on startup (already existed, ensure it's kept)
        UI_DEBUG = False # For debugging UI elements like button states
        MONITOR_ENERGY_CONSERVATION = True # Enable printing of system energy
        ENERGY_CHECK_INTERVAL_STEPS = 100 # How often to check energy conservation (moved here)


    def __init__(self):
        """
        Initializes the configuration object.
        Calculates dependent configuration values and runs validation.
        """
        # --- Calculate dependent RL Action Space Dimensions ---
        num_thrust_options = len(self.Probe.THRUST_FORCE_MAGNITUDES)

        # Torque options: 1 (None) + 2 * (number of non-zero torque magnitudes)
        num_torque_magnitudes = len(self.Probe.TORQUE_MAGNITUDES)
        torque_options_count = 1 # Default to 1 (for "None" torque)
        if num_torque_magnitudes > 0:
            # Count non-zero magnitudes. Assumes 0.0 is for "no torque" if present.
            non_zero_torque_mags = [m for m in self.Probe.TORQUE_MAGNITUDES if m != 0.0]
            if non_zero_torque_mags:
                 torque_options_count = 1 + 2 * len(non_zero_torque_mags)
            # If TORQUE_MAGNITUDES was e.g. [0.0], torque_options_count remains 1.
            # If TORQUE_MAGNITUDES was e.g. [0.0, 0.1, 0.2], non_zero_torque_mags is [0.1, 0.2] (len 2), count = 1 + 2*2 = 5.

        self.RL.ACTION_SPACE_DIMS = [
            num_thrust_options,
            torque_options_count,
            2,  # Communicate (Yes/No)
            2,  # Replicate (Yes/No)
            self.RL.NUM_OBSERVED_RESOURCES_FOR_TARGETING + 1  # Target selection (N resources + None)
        ]
        
        # Ensure Sun's mass in PLANET_DATA is correctly referenced
        if 'Sun' in self.SolarSystem.PLANET_DATA:
            self.SolarSystem.PLANET_DATA['Sun']['mass_kg'] = self.SolarSystem.SUN_MASS_KG

        self.validate()

    def validate(self):
        """
        Validates the configuration for consistency and logical errors.
        Raises ConfigurationError if an issue is found.
        """
        # Global scale checks
        if AU_SCALE <= 0:
            raise ConfigurationError("Global AU_SCALE must be positive.")
        if KM_SCALE <= 0:
            raise ConfigurationError("Global KM_SCALE must be positive (derived from AU_SCALE and AU_KM).")

        # World validation
        if self.World.SIZE_AU <= 0:
            raise ConfigurationError("World.SIZE_AU must be positive.")
        if self.World.SIZE_SIM <= 0: # Should be derived correctly if SIZE_AU and AU_SCALE are positive
            raise ConfigurationError("World.SIZE_SIM must be positive.")
        if not (0 < self.World.ASTEROID_BELT_INNER_AU < self.World.ASTEROID_BELT_OUTER_AU < self.World.SIZE_AU):
            raise ConfigurationError(
                f"Asteroid belt AUs (Inner: {self.World.ASTEROID_BELT_INNER_AU}, Outer: {self.World.ASTEROID_BELT_OUTER_AU}) "
                f"must be within world size ({self.World.SIZE_AU}) and inner < outer."
            )

        # Physics validation
        if self.Physics.TIMESTEP_SECONDS <= 0:
            raise ConfigurationError("Physics.TIMESTEP_SECONDS must be positive.")
        if self.Physics.DEFAULT_PROBE_MASS_KG <= 0:
            raise ConfigurationError("Physics.DEFAULT_PROBE_MASS_KG must be positive.")

        # Time validation
        if self.Time.SIM_SECONDS_PER_STEP != self.Physics.TIMESTEP_SECONDS:
            raise ConfigurationError("Time.SIM_SECONDS_PER_STEP must match Physics.TIMESTEP_SECONDS for consistency.")

        # Probe validation
        if self.Probe.MAX_ENERGY <= 0:
            raise ConfigurationError("Probe.MAX_ENERGY must be positive.")
        if not (0 <= self.Probe.INITIAL_ENERGY <= self.Probe.MAX_ENERGY):
            raise ConfigurationError("Probe.INITIAL_ENERGY must be between 0 and MAX_ENERGY.")
        # ... (add more comprehensive probe checks as needed)

        # RL validation (ensure percentages are valid)
        thresholds_to_check = {
            "HIGH_ENERGY_THRESHOLD_PERCENT": self.RL.HIGH_ENERGY_THRESHOLD_PERCENT,
            "CRITICAL_ENERGY_THRESHOLD_PERCENT": self.RL.CRITICAL_ENERGY_THRESHOLD_PERCENT,
            "LOW_ENERGY_PENALTY_LEVEL_1_THRESHOLD_PERCENT": self.RL.LOW_ENERGY_PENALTY_LEVEL_1_THRESHOLD_PERCENT,
            # "LOW_ENERGY_PENALTY_LEVEL_2_THRESHOLD_PERCENT": self.RL.LOW_ENERGY_PENALTY_LEVEL_2_THRESHOLD_PERCENT # This was 0.10, same as CRITICAL.
        }
        for name, threshold in thresholds_to_check.items():
            if not (0 <= threshold <= 1):
                raise ConfigurationError(f"RL.{name} ({threshold}) must be a percentage between 0 and 1.")
        
        # Validate RL threshold ordering
        if not (self.RL.CRITICAL_ENERGY_THRESHOLD_PERCENT < self.RL.LOW_ENERGY_PENALTY_LEVEL_1_THRESHOLD_PERCENT):
             raise ConfigurationError(
                 f"RL energy thresholds: CRITICAL_ENERGY_THRESHOLD_PERCENT ({self.RL.CRITICAL_ENERGY_THRESHOLD_PERCENT}) "
                 f"must be less than LOW_ENERGY_PENALTY_LEVEL_1_THRESHOLD_PERCENT ({self.RL.LOW_ENERGY_PENALTY_LEVEL_1_THRESHOLD_PERCENT})."
             )
        # Note: Original LOW_ENERGY_PENALTY_LEVEL_2_THRESHOLD was 0.10, same as CRITICAL.
        # If it's intended to be a distinct level, its relation to other thresholds needs clarification.
        # For now, only checking L1 vs Critical.

        # Visualization
        if self.Visualization.SCREEN_WIDTH_PX <=0 or self.Visualization.SCREEN_HEIGHT_PX <=0:
            raise ConfigurationError("Screen dimensions (SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX) must be positive.")
        if self.Visualization.FPS <=0:
            raise ConfigurationError("Visualization.FPS must be positive.")

        # Solar System Data Validation
        if self.SolarSystem.SUN_MASS_KG <= 0:
            raise ConfigurationError("SolarSystem.SUN_MASS_KG must be positive.")
        for name, data in self.SolarSystem.PLANET_DATA.items():
            if data['mass_kg'] < 0: # Sun's mass is positive, others can be 0 for massless points if needed, but not negative.
                raise ConfigurationError(f"Mass of celestial body {name} cannot be negative.")
            if data['radius_km'] < 0:
                raise ConfigurationError(f"Radius of celestial body {name} cannot be negative.")
            if data['semi_major_axis_au'] < 0 and name != 'Sun': # Sun SMA is 0
                 raise ConfigurationError(f"Semi-major axis of celestial body {name} cannot be negative.")
            if not (0 <= data['eccentricity'] < 1): # Eccentricity must be [0, 1) for ellipses
                 raise ConfigurationError(f"Eccentricity of celestial body {name} ({data['eccentricity']}) must be between 0 (inclusive) and 1 (exclusive).")
            # Inclination can be 0-180. Longitudes/Arguments 0-360. Mean Anomaly 0-360.
            if not (0 <= data['inclination_deg'] <= 180):
                raise ConfigurationError(f"Inclination of {name} ({data['inclination_deg']}) must be between 0 and 180 degrees.")

            # Check central body validity
            central_body_name = data.get('central_body')
            if central_body_name is not None and central_body_name not in self.SolarSystem.PLANET_DATA:
                raise ConfigurationError(f"Central body '{central_body_name}' for '{name}' not found in PLANET_DATA.")
            if name == 'Sun' and central_body_name is not None:
                raise ConfigurationError("The Sun cannot have a central body.")
            if name != 'Sun' and central_body_name is None:
                raise ConfigurationError(f"Celestial body '{name}' (not Sun) must have a 'central_body' defined.")


        # print("Configuration validated successfully.") # Optional: use logging instead

# --- Instantiate the configuration ---
# This makes the config object available for import and runs validation.
# e.g., from config import config
try:
    config = SimulationConfig()
except ConfigurationError as e:
    print(f"FATAL CONFIGURATION ERROR: {e}")
    # In a real application, you might log this and exit, or re-raise
    # to ensure the application does not run with an invalid config.
    raise