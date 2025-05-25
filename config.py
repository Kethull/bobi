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

# Communication
COMM_RANGE = 100
MESSAGE_TYPES = ['RESOURCE_LOCATION']

# Training Configuration
EPISODE_LENGTH = 50000
LEARNING_RATE = 3e-4
BATCH_SIZE = 64

# Visualization
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
SPACESHIP_SIZE = 12 # New - Visual size of the spaceship