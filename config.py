# config.py
import numpy as np

# Physical Constants
AU_KM = 149597870.7  # Astronomical Unit in kilometers
GRAVITATIONAL_CONSTANT_KM3_KG_S2 = 6.67430e-20  # G in km^3 kg^-1 s^-2
SECONDS_PER_DAY = 86400.0

# World Configuration / Solar System Scale
AU_SCALE = 10000.0  # Sim units per AU (simulation units per Astronomical Unit)
KM_SCALE = AU_SCALE / AU_KM  # Sim units per km

WORLD_SIZE_AU = 10.0  # Diameter of the simulation area in AU (reduced for better default view)
WORLD_SIZE_SIM = WORLD_SIZE_AU * AU_SCALE # Diameter of the simulation area in simulation units
WORLD_WIDTH = WORLD_SIZE_SIM
WORLD_HEIGHT = WORLD_SIZE_SIM
SUN_POSITION_SIM = [WORLD_SIZE_SIM / 2, WORLD_SIZE_SIM / 2]

# MAX_PROBES = 20 # Retained from previous config
INITIAL_PROBES = 1 # Retained

# Resource Configuration (largely retained, may be less relevant for pure solar system sim)
RESOURCE_COUNT = 1
RESOURCE_MIN_AMOUNT = 10000
RESOURCE_MAX_AMOUNT = 20000
RESOURCE_REGEN_RATE = 0.00
HARVEST_RATE = 2.0
HARVEST_DISTANCE = 5
DISCOVERY_RANGE = HARVEST_DISTANCE * 2.5
RESOURCE_DISCOVERY_REWARD_FACTOR = 0.05

# Probe Configuration (retained, may be less relevant)
MAX_ENERGY = 100000
INITIAL_ENERGY = 90000
REPLICATION_COST = 80000
REPLICATION_MIN_ENERGY = 99900
PROBE_MASS = 8.0
MAX_VELOCITY = 10000.0
THRUST_FORCE = [0.0, 0.08, 0.18, 0.32]
THRUST_ENERGY_COST_FACTOR = 0.001
ENERGY_DECAY_RATE = 0.001
LOW_POWER_PENALTY = 0.9
SUSTAINED_MINING_REWARD_PER_STEP = 0.05
HIGH_ENERGY_THRESHOLD = 0.75
HIGH_ENERGY_REWARD_BONUS = 0.1
TARGET_PROXIMITY_REWARD_FACTOR = 1.95
MOVE_AWAY_FROM_TARGET_PENALTY_FACTOR = 0.5
REACH_TARGET_BONUS = 2.0
PROXIMITY_REWARD_FALLOFF = 5.0
TARGET_SWITCH_ENERGY_COST = 1.0
TARGET_SWITCH_COOLDOWN = 20

# Reward Shaping (retained)
CRITICAL_ENERGY_THRESHOLD = 0.10
STAY_ALIVE_REWARD_BONUS = 0.02
LOW_ENERGY_PENALTY_LEVEL_1_THRESHOLD = 0.25
LOW_ENERGY_PENALTY_LEVEL_1_FACTOR = 0.1
LOW_ENERGY_PENALTY_LEVEL_2_THRESHOLD = 0.10
LOW_ENERGY_PENALTY_LEVEL_2_FACTOR = 0.3

# Rotational Physics Configuration (retained)
MOMENT_OF_INERTIA = 5.0
TORQUE_MAGNITUDES = [0.0, 0.008, 0.018]
ROTATIONAL_ENERGY_COST_FACTOR = 0.11
MAX_ANGULAR_VELOCITY = np.pi / 4
ANGULAR_DAMPING_FACTOR = 0.05

# Smoothing Parameters (retained)
ACTION_SMOOTHING_FACTOR = 0.85
MIN_THRUST_DURATION = 30
THRUST_RAMP_TIME = 60
ROTATION_SMOOTHING_FACTOR = 0.9
MIN_ROTATION_DURATION = 6
ROTATION_RAMP_TIME = 4

# Anti-spam parameters (retained)
THRUSTER_STARTUP_ENERGY_COST = 1.5
RAPID_SWITCHING_PENALTY = 0.8
SWITCHING_DETECTION_WINDOW = 10

# Communication (retained)
COMM_RANGE = 100
MESSAGE_TYPES = ['RESOURCE_LOCATION']

# Debug Configuration
DEBUG_MODE = False
DEBUG_ENERGY_RESET_VALUE = 50.0
DEBUG_ORBITAL_MECHANICS = True # New debug flag

# Training Configuration (retained)
EPISODE_LENGTH = 50000
LEARNING_RATE = 3e-4
BATCH_SIZE = 64

# RL Agent Configuration (retained)
NUM_OBSERVED_RESOURCES_FOR_TARGETING = 3
OBSERVATION_SPACE_SIZE = 25 # Based on previous analysis
ACTION_SPACE_DIMS = [
    4, # Linear thrust
    5, # Rotational torque (corrected from 2 to 5 options: None, L_Low, L_High, R_Low, R_High)
    2, # Communicate
    2, # Replicate
    NUM_OBSERVED_RESOURCES_FOR_TARGETING + 1
]

# Solar System Configuration
# Reference Epoch for orbital elements: J2000.0
SUN_MASS_KG = 1.9885e30  # Mass of the Sun in kg

PLANET_DATA = {
    'Sun': {
        'mass_kg': SUN_MASS_KG,
        'radius_km': 695700.0,
        'display_radius_sim': 500, # Visual radius in simulation units
        'color': (255, 255, 100),
        # Orbital elements are not applicable for the central body
        'semi_major_axis_au': 0.0,
        'eccentricity': 0.0,
        'inclination_deg': 0.0,
        'longitude_of_ascending_node_deg': 0.0,
        'argument_of_perihelion_deg': 0.0,
        'mean_anomaly_at_epoch_deg': 0.0
    },
    'Mercury': {
        'mass_kg': 0.33011e24,
        'radius_km': 2439.7,
        'display_radius_sim': 30,
        'color': (169, 169, 169),
        'semi_major_axis_au': 0.387098,
        'eccentricity': 0.205630,
        'inclination_deg': 7.005,
        'longitude_of_ascending_node_deg': 48.331,
        'argument_of_perihelion_deg': 29.124, # (Lp - LoAN = 77.455 - 48.331)
        'mean_anomaly_at_epoch_deg': 174.794 # (L - Lp = 252.25084 - 77.45645)
    },
    'Venus': {
        'mass_kg': 4.8675e24,
        'radius_km': 6051.8,
        'display_radius_sim': 50,
        'color': (255, 198, 73),
        'semi_major_axis_au': 0.723332,
        'eccentricity': 0.006772,
        'inclination_deg': 3.39458,
        'longitude_of_ascending_node_deg': 76.680,
        'argument_of_perihelion_deg': 54.884, # (Lp - LoAN = 131.564 - 76.680)
        'mean_anomaly_at_epoch_deg': 50.447 # (L - Lp = 181.97973 - 131.53298)
    },
    'Earth': {
        'mass_kg': 5.97237e24,
        'radius_km': 6371.0,
        'display_radius_sim': 50,
        'color': (100, 149, 237),
        'semi_major_axis_au': 1.00000261,
        'eccentricity': 0.01671123,
        'inclination_deg': 0.00005, # Effectively 0 for ecliptic alignment
        'longitude_of_ascending_node_deg': -11.26064, # Or 0 if Earth is the reference plane
        'argument_of_perihelion_deg': 114.20783, # (Lp - LoAN = 102.94719 - (-11.26064))
        'mean_anomaly_at_epoch_deg': 357.51716 # (L - Lp = 100.46435 - 102.94719, then +360)
    },
    'Moon': { # Orbital elements are relative to Earth
        'mass_kg': 0.07346e24,
        'radius_km': 1737.4,
        'display_radius_sim': 15,
        'color': (200, 200, 200),
        'semi_major_axis_au': 0.00257, # (384,400 km)
        'eccentricity': 0.0549,
        'inclination_deg': 5.145, # Relative to Earth's equator
        'longitude_of_ascending_node_deg': 125.08, # Approximate for J2000, varies
        'argument_of_perihelion_deg': 318.15, # Approximate for J2000, varies
        'mean_anomaly_at_epoch_deg': 115.36 # Approximate for J2000
    },
    'Mars': {
        'mass_kg': 0.64171e24,
        'radius_km': 3389.5,
        'display_radius_sim': 40,
        'color': (193, 68, 14),
        'semi_major_axis_au': 1.523679,
        'eccentricity': 0.09340,
        'inclination_deg': 1.850,
        'longitude_of_ascending_node_deg': 49.558,
        'argument_of_perihelion_deg': 286.502, # (Lp - LoAN = 336.060 - 49.558)
        'mean_anomaly_at_epoch_deg': 19.412 # (L - Lp = 355.45332 - 336.04084)
    }
}

# Time scaling
SIM_SECONDS_PER_STEP = 3600.0  # 1 simulation step = 1 hour

# Visualization
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
FPS = 60
SPACESHIP_SIZE = 12
SHOW_SIMPLE_PROBE_INFO = True
SHOW_SIMPLE_RESOURCE_INFO = True

# Organic Ship Visual Theme (retained)
ORGANIC_SHIP_ENABLED = True
SHIP_SCALE_FACTOR = 1.2
ELEGANT_PARTICLE_MODE = False
ENABLE_PARTICLE_EFFECTS = False

# Enhanced visual settings (retained)
METALLIC_SHEEN_INTENSITY = 0.7
ORGANIC_CURVE_RESOLUTION = 20
ENGINE_GLOW_INTENSITY = 1.2
ORGANIC_EXHAUST_PARTICLE_SCALE = 0.1

# Asteroid Belt (retained from original Phase 2 spec, values might need review later)
ASTEROID_BELT_INNER_AU = 2.2
ASTEROID_BELT_OUTER_AU = 3.2
ASTEROID_COUNT = 500
ASTEROID_MIN_RADIUS_SIM = 0.2 # Min display radius in sim units for asteroids (Phase 2 Spec)
ASTEROID_MAX_RADIUS_SIM = 0.8 # Max display radius in sim units for asteroids (Phase 2 Spec)
ASTEROID_DEFAULT_COLOR = (100, 100, 100)
ASTEROID_MASS_KG_MIN = 1e10
ASTEROID_MASS_KG_MAX = 1e15