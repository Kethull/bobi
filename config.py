# config.py
import numpy as np

# World Configuration
WORLD_WIDTH = 2000
WORLD_HEIGHT = 2000
MAX_PROBES = 20
INITIAL_PROBES = 1

# Resource Configuration
RESOURCE_COUNT = 1
RESOURCE_MIN_AMOUNT = 50
RESOURCE_MAX_AMOUNT = 200
RESOURCE_REGEN_RATE = 0.00  # units per step
HARVEST_RATE = 2.0
HARVEST_DISTANCE = 5
DISCOVERY_RANGE = HARVEST_DISTANCE * 2.5 # Range within which a resource is "discovered"
RESOURCE_DISCOVERY_REWARD = 1.0 # One-time reward for discovering a resource

# Probe Configuration
MAX_ENERGY = 100
INITIAL_ENERGY = 50
REPLICATION_COST = 40
REPLICATION_MIN_ENERGY = 60
PROBE_MASS = 10.0  # Modified - Mass of the probe
MAX_VELOCITY = 10.0  # Modified - Max speed for normalization/safety
THRUST_FORCE = [0.0, 1.0, 2.0]  # Renamed and values updated
THRUST_ENERGY_COST_FACTOR = 0.05 # New - Cost per unit of force per step
ENERGY_DECAY_RATE = 0.00
LOW_POWER_PENALTY = 0.1 # New - Penalty per step for being in low power mode (energy <= 0)
TARGET_PROXIMITY_REWARD_FACTOR = 0.05 # Reward factor for getting closer to a selected target

# Communication
COMM_RANGE = 100
MESSAGE_TYPES = ['RESOURCE_LOCATION']

# Training Configuration
EPISODE_LENGTH = 50000
LEARNING_RATE = 3e-4
BATCH_SIZE = 64

# RL Agent Configuration
NUM_OBSERVED_RESOURCES_FOR_TARGETING = 3 # How many nearest resources can be targeted
OBSERVATION_SPACE_SIZE = 22 # pos(2) + vel(2) + energy(1) + age(1) + resources(3*3=9) + probes(2*2=4) + messages(1) + target_rel_pos(2) + target_active(1)
ACTION_SPACE_DIMS = [
    9,  # thrust_dir (0=None, 1-8=directions)
    3,  # thrust_power (0-2)
    2,  # communicate (0=No, 1=Yes)
    2,  # replicate (0=No, 1=Yes)
    NUM_OBSERVED_RESOURCES_FOR_TARGETING + 1  # target_select (0=None, 1-N=select_observed_resource_N)
]

# Visualization
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
SPACESHIP_SIZE = 12 # New - Visual size of the spaceship