# config.py
import numpy as np

# World Configuration
WORLD_WIDTH = 2000
WORLD_HEIGHT = 2000
MAX_PROBES = 20
INITIAL_PROBES = 3

# Resource Configuration
RESOURCE_COUNT = 1
RESOURCE_MIN_AMOUNT = 50
RESOURCE_MAX_AMOUNT = 200
RESOURCE_REGEN_RATE = 0.00  # units per step
HARVEST_RATE = 2.0
HARVEST_DISTANCE = 5
DISCOVERY_RANGE = HARVEST_DISTANCE * 2.5 # Range within which a resource is "discovered"
RESOURCE_DISCOVERY_REWARD = 1.0 # One-time reward for discovering a resource (kept for compatibility, new factor preferred)
RESOURCE_DISCOVERY_REWARD_FACTOR = 0.05 # Reward per unit of discovered resource amount

# Probe Configuration
MAX_ENERGY = 1000
INITIAL_ENERGY = 900
REPLICATION_COST = 800
REPLICATION_MIN_ENERGY = 999
PROBE_MASS = 8.0  # Slightly lighter for more responsive feel
MAX_VELOCITY = 10.0  # Modified - Max speed for normalization/safety
THRUST_FORCE = [0.0, 0.08, 0.18, 0.32]  # Reduce from [0.0, 0.15, 0.35, 0.6]
THRUST_ENERGY_COST_FACTOR = 0.001  # Lower cost for more action
ENERGY_DECAY_RATE = 0.001  # Slightly lower decay
LOW_POWER_PENALTY = 0.9 # New - Penalty per step for being in low power mode (energy <= 0)
SUSTAINED_MINING_REWARD_PER_STEP = 0.05 # Small reward for each step actively mining a valid target
HIGH_ENERGY_THRESHOLD = 0.75 # Percentage of MAX_ENERGY to qualify for high energy bonus
HIGH_ENERGY_REWARD_BONUS = 0.1 # Bonus reward for maintaining energy above threshold
TARGET_PROXIMITY_REWARD_FACTOR = 1.95 # Base reward factor for getting closer to a selected target
MOVE_AWAY_FROM_TARGET_PENALTY_FACTOR = 0.5 # Penalty factor for moving away from selected target
REACH_TARGET_BONUS = 2.0 # One-time bonus for reaching a selected target
PROXIMITY_REWARD_FALLOFF = 5.0 # Smaller value = sharper reward increase at very close distances. Added to distance in denominator.
TARGET_SWITCH_ENERGY_COST = 1.0 # Energy cost for switching to a different resource target
TARGET_SWITCH_COOLDOWN = 20     # Steps a probe must wait before switching to another different resource target without penalty

# Reward Shaping
CRITICAL_ENERGY_THRESHOLD = 0.10  # Fraction of MAX_ENERGY below which is critical, but not zero
STAY_ALIVE_REWARD_BONUS = 0.02  # Small reward per step for being above CRITICAL_ENERGY_THRESHOLD
LOW_ENERGY_PENALTY_LEVEL_1_THRESHOLD = 0.25  # e.g., 25% energy
LOW_ENERGY_PENALTY_LEVEL_1_FACTOR = 0.1  # Penalty factor if energy is below this but above level 2
LOW_ENERGY_PENALTY_LEVEL_2_THRESHOLD = 0.10  # e.g., 10% energy
LOW_ENERGY_PENALTY_LEVEL_2_FACTOR = 0.3  # Penalty factor if energy is below this but above zero
# Rotational Physics Configuration
MOMENT_OF_INERTIA = 5.0  # Affects rotational acceleration (higher = slower)
# ROTATIONAL_THRUST_TORQUE defines torque values for [None, Left_Low, Left_High, Right_Low, Right_High] actions
# For simplicity, let's use direct torque values for actions: 0=None, 1=Torque_L1, 2=Torque_L2, 3=Torque_R1, 4=Torque_R2
# Let's define torque magnitudes and apply direction in environment
TORQUE_MAGNITUDES = [0.0, 0.008, 0.018]  # Reduce from [0.0, 0.015, 0.035]
ROTATIONAL_ENERGY_COST_FACTOR = 0.11  # Lower rotational cost
MAX_ANGULAR_VELOCITY = np.pi / 4  # Max turn rate (radians/step) for normalization
ANGULAR_DAMPING_FACTOR = 0.05    # Reduces angular velocity each step (e.g., 0.05 = 5% reduction)

# Smoothing Parameters
ACTION_SMOOTHING_FACTOR = 0.85  # Increase from 0.7 for more smoothing
MIN_THRUST_DURATION = 8        # Increase from 4 for more realistic burns
THRUST_RAMP_TIME = 6           # Increase from 3
ROTATION_SMOOTHING_FACTOR = 0.9 # Increase from 0.8
MIN_ROTATION_DURATION = 6      # Increase from 3
ROTATION_RAMP_TIME = 4         # Increase from 2

# Anti-spam parameters
THRUSTER_STARTUP_ENERGY_COST = 1.5  # Extra cost for starting thrusters
RAPID_SWITCHING_PENALTY = 0.8       # Penalty for frequent action changes
SWITCHING_DETECTION_WINDOW = 10     # Frames to track switching frequency

# Communication
COMM_RANGE = 100
MESSAGE_TYPES = ['RESOURCE_LOCATION']

# Debug Configuration
DEBUG_MODE = False  # Set to True to enable debug features
DEBUG_ENERGY_RESET_VALUE = 50.0 # Energy value to reset to if probe runs out in debug mode

# Training Configuration
EPISODE_LENGTH = 50000
LEARNING_RATE = 3e-4
BATCH_SIZE = 64

# RL Agent Configuration
NUM_OBSERVED_RESOURCES_FOR_TARGETING = 3 # How many nearest resources can be targeted
# OBSERVATION_SPACE_SIZE:
# Base (19 values): pos(2), vel(2), energy(1), age(1), resources(3*3=9), probes(2*2=4), messages(1)
# Target info (3 values): target_active(1), target_rel_pos(2)
# Rotational info (2 values): angle(1), angular_velocity(1)
# Total = 19 + 3 + 2 = 24.  Wait, this is still the old calculation.
# Let's list them out carefully:
# 1. pos_x, pos_y (2)
# 2. vel_x, vel_y (2)
# 3. energy (1)
# 4. age (1)
# 5. resource1_dx, resource1_dy, resource1_amt (3)
# 6. resource2_dx, resource2_dy, resource2_amt (3)
# 7. resource3_dx, resource3_dy, resource3_amt (3)
# 8. probe1_dist, probe1_energy (2)
# 9. probe2_dist, probe2_energy (2)
# 10. message_count (1)
# --- Subtotal so far: 2+2+1+1+3+3+3+2+2+1 = 20. This is different from the "Base (19)" comment.
# Let's use the previous diff's indexing logic for environment.py as the source of truth for counts:
# Own state: 6 (indices 0-5)
# Resources (NUM_OBSERVED_RESOURCES_FOR_TARGETING * 3): 9 (indices 6-14)
# Other Probes (2 * 2): 4 (indices 15-18)
# Messages: 1 (index 19)
# Target Active: 1 (index 20)
# Target Rel Pos (X, Y): 2 (indices 21-22)
# Angle: 1 (index 23)
# Angular Velocity: 1 (index 24)
# Total size needed is 25 (for indices 0-24).
OBSERVATION_SPACE_SIZE = 25

# ACTION_SPACE_DIMS:
# 1. Linear Thrust Power (forward only): 0=None, 1=Low, 2=High (3 options)
# 2. Rotational Torque: 0=None, 1=Left_Low, 2=Left_High, 3=Right_Low, 4=Right_High (5 options)
# 3. Communicate: 0=No, 1=Yes (2 options)
# 4. Replicate: 0=No, 1=Yes (2 options)
# 5. Target Select: 0=None, 1-N for observed resources (NUM_OBSERVED_RESOURCES_FOR_TARGETING + 1 options)
ACTION_SPACE_DIMS = [
    4,  # Linear thrust power (0=None, 1=Low, 2=Mid, 3=High) corresponding to THRUST_FORCE levels
    2,  # Rotational torque (0=None, 1=L_Low, 2=L_High, 3=R_Low, 4=R_High)
    2,  # Communicate
    2,  # Replicate
    NUM_OBSERVED_RESOURCES_FOR_TARGETING + 1
]

# Visualization
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
SPACESHIP_SIZE = 12 # New - Visual size of the spaceship
SHOW_SIMPLE_PROBE_INFO = True # If True, shows basic ID/Energy bar above probes (can be disabled if ModernUI handles it)
SHOW_SIMPLE_RESOURCE_INFO = True # If True, shows basic amount text near resources (can be disabled if ModernUI handles it)
# Organic Ship Visual Theme
ORGANIC_SHIP_ENABLED = True
SHIP_SCALE_FACTOR = 1.2  # Slightly larger for elegance
ELEGANT_PARTICLE_MODE = False
ENABLE_PARTICLE_EFFECTS = False # Global toggle for all particle effects

# Enhanced visual settings
METALLIC_SHEEN_INTENSITY = 0.7
ORGANIC_CURVE_RESOLUTION = 20  # Higher for smoother curves
ENGINE_GLOW_INTENSITY = 1.2
ORGANIC_EXHAUST_PARTICLE_SCALE = 0.1 # Scale factor for organic ship exhaust particles (default 1.0)