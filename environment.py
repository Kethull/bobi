# environment.py
import gym
from gym import spaces
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from config import *

@dataclass
class Message:
    sender_id: int
    msg_type: str
    position: Tuple[float, float]
    resource_amount: float
    timestamp: int

@dataclass
class Resource:
    position: Tuple[float, float]
    amount: float
    max_amount: float
    
    def harvest(self, rate: float) -> float:
        harvested = min(rate, self.amount)
        self.amount -= harvested
        return harvested
    
    def regenerate(self):
        self.amount = min(self.max_amount, self.amount + RESOURCE_REGEN_RATE)

class SpaceEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        self.world_width = WORLD_WIDTH
        self.world_height = WORLD_HEIGHT
        self.resources = []
        self.probes = {}
        self.messages = []
        self.step_count = 0
        self.max_probe_id = 0
        self.total_resources_mined = 0.0
        
        # Initialize resources
        self._generate_resources()
        
        # Observation space for each probe
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(OBSERVATION_SPACE_SIZE,), # Uses constant from config.py
            dtype=np.float32
        )
        
        # Action space: [thrust_dir, thrust_power, communicate, replicate, target_select]
        self.action_space = spaces.MultiDiscrete(ACTION_SPACE_DIMS) # Uses constant from config.py
        
    def _generate_resources(self):
        self.resources = []
        for _ in range(RESOURCE_COUNT):
            pos = (random.uniform(0, self.world_width), 
                   random.uniform(0, self.world_height))
            amount = random.uniform(RESOURCE_MIN_AMOUNT, RESOURCE_MAX_AMOUNT)
            self.resources.append(Resource(pos, amount, amount))
    
    def add_probe(self, probe_id: int, position: Tuple[float, float], 
                  energy: float = INITIAL_ENERGY, generation: int = 0):
        """Add a new probe to the environment"""
        self.probes[probe_id] = {
            'position': np.array(position, dtype=np.float32),
            'velocity': np.array([0.0, 0.0], dtype=np.float32),
            'energy': energy,
            'age': 0,
            'generation': generation,
            'visited_positions': set(),
            'total_reward': 0,
            'alive': True,
            'mass': PROBE_MASS,  # Initialize probe mass
            'is_thrusting_visual': False, # For visualization
            'thrust_power_visual': 0,     # For visualization
            'is_mining_visual': False,    # For mining laser visualization
            'mining_target_pos_visual': None, # For mining laser visualization
            'discovered_resources': set(), # For resource discovery reward
            'selected_target_info': None, # For target selection: {'type': 'resource', 'id': res_idx, 'world_pos': (x,y)}
            'distance_to_target_last_step': float('inf'), # For target proximity reward
            'angle': random.uniform(0, 2 * np.pi), # Initial orientation in radians
            'angular_velocity': 0.0, # Initial angular velocity in radians/step
            'last_target_switch_time': 0, # For target switching cooldown
            'current_target_id': None, # Actual ID of the currently selected resource, for anti-spam
            'has_reached_current_target': False, # For one-time bonus upon reaching selected target
            'action_smoothing_state': {
                'previous_linear_action': 0,
                'previous_rotation_action': 0, # This was in the prompt but might not be directly used if blending raw actions
                'thrust_timer': 0,
                'target_thrust_power': 0,
                'current_thrust_ramp': 0.0,  # Current ramped power for linear thrust (0 to target_thrust_power)
                'linear_blend': 0.0,         # Smoothed linear command (0-3 for linear_thrust_action)
                
                'rotation_blend': 0.0,       # Smoothed rotation command choice (0-4 for rotational_torque_action)
                'rotation_timer': 0,         # Timer for minimum rotation duration
                'target_rotation_action': 0, # The target rotational action (0-4) being ramped/maintained
                'current_rotation_ramp': 0.0, # Current ramped "intensity" of rotation (0 to target_rotation_action's implied level)
                
                # For multi-stage smoothing
                'secondary_linear_blend': 0.0,
                'secondary_rotation_blend': 0.0,

                # For anti-spam measures
                'action_switches_recent': 0,
                'last_effective_linear': 0,
                'last_effective_rotation': 0,
                'startup_energy_debt': 0.0
            }
        }
        self.max_probe_id = max(self.max_probe_id, probe_id)
    
    def get_observation(self, probe_id: int) -> np.ndarray:
        """Get observation for a specific probe"""
        probe = self.probes[probe_id]
        obs = np.zeros(OBSERVATION_SPACE_SIZE, dtype=np.float32) # Use constant
        
        # Own state (6 values) - Indices 0-5
        obs[0:2] = probe['position'] / np.array([self.world_width, self.world_height])
        obs[2:4] = probe['velocity'] / MAX_VELOCITY
        obs[4] = probe['energy'] / MAX_ENERGY
        obs[5] = probe['age'] / EPISODE_LENGTH
        
        # Nearest 3 resources (9 values)
        resource_distances = []
        for i, resource in enumerate(self.resources):
            if resource.amount > 0:
                dist = self._distance(probe['position'], resource.position)
                resource_distances.append((dist, i))
        
        resource_distances.sort()
        # Nearest NUM_OBSERVED_RESOURCES_FOR_TARGETING resources (default 3*3=9 values) - Indices 6-14
        for i in range(NUM_OBSERVED_RESOURCES_FOR_TARGETING):
            base_idx = 6 + i * 3
            if i < len(resource_distances):
                _, res_idx = resource_distances[i]
                resource = self.resources[res_idx]
                rel_pos = np.array(resource.position) - probe['position']
                rel_pos = self._wrap_position(rel_pos)
                obs[base_idx:base_idx+2] = rel_pos / np.array([self.world_width, self.world_height])
                obs[base_idx+2] = resource.amount / RESOURCE_MAX_AMOUNT
        
        # Nearest 2 other probes (4 values)
        other_probes = [(pid, p) for pid, p in self.probes.items() 
                       if pid != probe_id and p['alive']]
        probe_distances = []
        for other_id, other_probe in other_probes:
            dist = self._distance(probe['position'], other_probe['position'])
            probe_distances.append((dist, other_id))
        
        probe_distances.sort()
        # Nearest 2 other probes (2*2=4 values) - Indices 15-18
        for i in range(2): # Assuming we always observe 2 other probes if available
            base_idx = 6 + NUM_OBSERVED_RESOURCES_FOR_TARGETING * 3 + i * 2 # Adjusted base_idx
            if i < len(probe_distances):
                _, other_id = probe_distances[i]
                other_probe = self.probes[other_id]
                rel_pos = np.array(other_probe['position']) - probe['position']
                rel_pos = self._wrap_position(rel_pos)
                obs[base_idx:base_idx+1] = [self._distance([0,0], rel_pos) / (self.world_width/2)]
                obs[base_idx+1] = other_probe['energy'] / MAX_ENERGY
        
        # Recent messages (1 value - simplified)
        recent_messages = [msg for msg in self.messages 
                          if self.step_count - msg.timestamp < 10 and
                          self._distance(probe['position'], msg.position) < COMM_RANGE]
        obs[6 + NUM_OBSERVED_RESOURCES_FOR_TARGETING * 3 + 2 * 2] = min(len(recent_messages) / 5.0, 1.0)  # Message count - Index 19
        
        # Target information (3 values) - Indices 20-22 (or 19-21 if OBS_SIZE is 22)
        # Correcting indices based on OBS_SIZE = 22:
        # Own state: 0-5 (6)
        # Resources: 6-14 (9) (assuming NUM_OBSERVED_RESOURCES_FOR_TARGETING = 3)
        # Other Probes: 15-18 (4)
        # Messages: 19 (1)
        # Target Active: 20 (1)
        # Target Rel Pos: 21-22 (2) -> This makes it 23. Let's adjust.
        # OBS_SIZE = 22 means indices 0-21.
        # Messages will be at index 19.
        # Target Active at 20. Target Rel Pos at 21, 22 is not possible.
        # Let's make Target Rel Pos use 2 slots, so OBS_SIZE should be 19 (current) + 1 (target_active) + 2 (target_rel_pos) = 22.
        # Indices: Messages (18), Target Active (19), Target Rel Pos (20, 21)

        # Indices for OBS_SIZE = 24:
        # Own state: 0-5 (6)
        # Resources: 6-14 (9) (NUM_OBSERVED_RESOURCES_FOR_TARGETING = 3)
        # Other Probes: 15-18 (4)
        # Messages: 19 (1)
        # Target Active: 20 (1)
        # Target Rel Pos: 21-22 (2)
        # Angle: 22 (1) -> Error in manual calculation, should be 22 for angle, 23 for ang_vel if OBS_SIZE=24
        # Corrected indices for OBS_SIZE = 24:
        # Messages: obs[19]
        # Target Active: obs[20]
        # Target Rel Pos X: obs[21]
        # Target Rel Pos Y: obs[22]
        # Angle: obs[23] -> This is still off.
        # Let's use the config breakdown: Base (19 values, indices 0-18), Target (3 values, indices 19-21), Rotation (2 values, indices 22-23)

        msg_idx = 6 + NUM_OBSERVED_RESOURCES_FOR_TARGETING * 3 + 4 # Base index for messages is 19
        obs[msg_idx] = min(len(recent_messages) / 5.0, 1.0) # Messages at index 19

        target_active_idx = msg_idx + 1 # Index 20
        target_rel_pos_x_idx = msg_idx + 2 # Index 21
        target_rel_pos_y_idx = msg_idx + 3 # Index 22

        if probe.get('selected_target_info') and probe['selected_target_info'].get('world_pos') is not None:
            obs[target_active_idx] = 1.0
            target_world_pos = probe['selected_target_info']['world_pos']
            rel_pos_to_target = np.array(target_world_pos) - probe['position']
            rel_pos_to_target = self._wrap_position(rel_pos_to_target)
            obs[target_rel_pos_x_idx] = rel_pos_to_target[0] / (self.world_width / 2) # Normalize to [-1, 1] approx
            obs[target_rel_pos_y_idx] = rel_pos_to_target[1] / (self.world_height / 2)
        else:
            obs[target_active_idx] = 0.0
            obs[target_rel_pos_x_idx] = 0.0
            obs[target_rel_pos_y_idx] = 0.0
            
        # Rotational Info - Indices 22 & 23 (if OBS_SIZE = 24)
        # Corrected: Angle at index 22, Angular Velocity at index 23
        angle_idx = target_rel_pos_y_idx + 1 # Index 22
        angular_velocity_idx = target_rel_pos_y_idx + 2 # Index 23

        obs[angle_idx] = (probe['angle'] % (2 * np.pi)) / (2 * np.pi)  # Normalize angle to [0, 1]
        obs[angular_velocity_idx] = np.clip(probe['angular_velocity'], -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY) / MAX_ANGULAR_VELOCITY # Normalize to [-1,1]
            
        return obs
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Execute one step of the environment"""
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        # Process actions for each probe
        for probe_id, action in actions.items():
            if probe_id in self.probes and self.probes[probe_id]['alive']:
                reward = self._process_probe_action(probe_id, action)
                rewards[probe_id] = reward
        
        # Update environment
        self._update_physics()
        self._regenerate_resources()
        self._cleanup_old_messages()
        self.step_count += 1
        
        # Generate observations
        for probe_id in self.probes:
            if self.probes[probe_id]['alive']:
                observations[probe_id] = self.get_observation(probe_id)
                dones[probe_id] = self.step_count >= EPISODE_LENGTH
                infos[probe_id] = {'generation': self.probes[probe_id]['generation']}
        
        return observations, rewards, dones, infos
    
    def _process_probe_action(self, probe_id: int, action: np.ndarray) -> float:
        """Process a single probe's action with smoothing and return reward"""
        probe = self.probes[probe_id]
        reward = 0.0
        
        # Reset visual flags
        probe['is_thrusting_visual'] = False
        probe['thrust_power_visual'] = 0
        probe['is_mining_visual'] = False
        probe['mining_target_pos_visual'] = None

        # Scaled Penalty for Low Energy
        if 0 < probe['energy'] <= MAX_ENERGY * LOW_ENERGY_PENALTY_LEVEL_2_THRESHOLD:
            reward -= LOW_ENERGY_PENALTY_LEVEL_2_FACTOR
        elif 0 < probe['energy'] <= MAX_ENERGY * LOW_ENERGY_PENALTY_LEVEL_1_THRESHOLD:
            reward -= LOW_ENERGY_PENALTY_LEVEL_1_FACTOR

        is_low_power = probe['energy'] <= 0
        if is_low_power:
            reward -= LOW_POWER_PENALTY

        # Parse raw actions from the agent
        raw_linear_thrust_action, raw_rotational_torque_action, \
        raw_communicate_action, raw_replicate_action, raw_target_select_action = action

        smoothing_state = probe['action_smoothing_state']

        # --- MULTI-STAGE ACTION SMOOTHING FOR LINEAR THRUST ---
        # First stage (existing, now primary blend)
        input_linear_action = 0 if is_low_power else raw_linear_thrust_action
        smoothing_state['linear_blend'] = (
            ACTION_SMOOTHING_FACTOR * smoothing_state['linear_blend'] +
            (1 - ACTION_SMOOTHING_FACTOR) * input_linear_action
        )

        # Second stage for ultra-smoothness
        smoothing_state['secondary_linear_blend'] = (
            0.9 * smoothing_state['secondary_linear_blend'] +
            0.1 * smoothing_state['linear_blend']
        )
        effective_linear_action = int(round(smoothing_state['secondary_linear_blend']))

        # --- LINEAR THRUST DURATION & RAMPING ---
        # Manage thrust timer and target power
        if effective_linear_action > 0 and not is_low_power:
            if smoothing_state['thrust_timer'] <= 0:  # Start new thrust
                smoothing_state['target_thrust_power'] = effective_linear_action
                smoothing_state['thrust_timer'] = MIN_THRUST_DURATION
        
        if smoothing_state['thrust_timer'] > 0:
            smoothing_state['thrust_timer'] -= 1
            if is_low_power:
                 smoothing_state['thrust_timer'] = 0 # Stop thrust if low power
        
        if smoothing_state['thrust_timer'] <= 0:
            smoothing_state['target_thrust_power'] = 0

        # Update current_thrust_ramp
        if smoothing_state['target_thrust_power'] > 0 and not is_low_power: # Ramp Up
            # ramp_progress_up is how much of the MIN_THRUST_DURATION has passed relative to RAMP_TIME
            # This means current_thrust_ramp will reach target_thrust_power when timer is (MIN_THRUST_DURATION - THRUST_RAMP_TIME)
            ramp_progress_up = min(1.0, (MIN_THRUST_DURATION - smoothing_state['thrust_timer']) / max(1, THRUST_RAMP_TIME))
            smoothing_state['current_thrust_ramp'] = smoothing_state['target_thrust_power'] * ramp_progress_up
        else: # Ramp Down (target_thrust_power is 0 or is_low_power)
            max_ramp_val = float(len(THRUST_FORCE) - 1) # Max possible value for target_thrust_power (e.g., 3)
            decrement_amount = max_ramp_val / max(1, THRUST_RAMP_TIME) # Ramp down from max in RAMP_TIME steps
            smoothing_state['current_thrust_ramp'] = max(0.0, smoothing_state['current_thrust_ramp'] - decrement_amount)

        # Apply actual thrust
        actual_thrust_magnitude = 0.0
        if smoothing_state['current_thrust_ramp'] > 0.01 and not is_low_power:
            # current_thrust_ramp is target_power * ramp_progress_up.
            # To get actual force, scale the THRUST_FORCE[target_power] by (current_ramp / target_power)
            # This simplifies to THRUST_FORCE[target_power] * ramp_progress_up
            # target_thrust_power should be valid index for THRUST_FORCE (0 to len-1)
            target_power_idx = smoothing_state['target_thrust_power']
            if 0 <= target_power_idx < len(THRUST_FORCE):
                 # Calculate ramp_progress based on current_thrust_ramp and target_thrust_power
                ramp_progress_for_force = 0.0
                if smoothing_state['target_thrust_power'] > 0: # Avoid division by zero if target is 0
                    ramp_progress_for_force = smoothing_state['current_thrust_ramp'] / smoothing_state['target_thrust_power']
                
                actual_thrust_magnitude = THRUST_FORCE[target_power_idx] * ramp_progress_for_force
            
            if actual_thrust_magnitude > 0.01:
                direction_vector = np.array([np.cos(probe['angle']), np.sin(probe['angle'])])
                force_vector = direction_vector * actual_thrust_magnitude
                acceleration = force_vector / probe['mass']
                probe['velocity'] += acceleration
                
                probe['is_thrusting_visual'] = True
                # Visual shows the intended power level, ramp affects actual force
                probe['thrust_power_visual'] = smoothing_state['target_thrust_power']
                
                energy_cost = actual_thrust_magnitude * THRUST_ENERGY_COST_FACTOR
                probe['energy'] = max(0, probe['energy'] - energy_cost)
                reward -= energy_cost * 0.01
        else: # Ensure visual is off if no thrust applied
            probe['is_thrusting_visual'] = False
            probe['thrust_power_visual'] = 0


        # --- MULTI-STAGE ACTION SMOOTHING FOR ROTATIONAL TORQUE ---
        input_rotational_action = 0 if is_low_power else raw_rotational_torque_action
        smoothing_state['rotation_blend'] = (
            ROTATION_SMOOTHING_FACTOR * smoothing_state['rotation_blend'] +
            (1 - ROTATION_SMOOTHING_FACTOR) * input_rotational_action
        )
        
        # Second stage for ultra-smoothness
        smoothing_state['secondary_rotation_blend'] = (
            0.95 * smoothing_state['secondary_rotation_blend'] +
            0.05 * smoothing_state['rotation_blend']
        )
        effective_rotation_action = int(round(smoothing_state['secondary_rotation_blend'])) # This is 0-4

        # --- THRUSTER ANTI-SPAM MEASURES ---
        # Detect and penalize rapid action switching
        # Note: effective_linear_action was defined earlier in the multi-stage linear smoothing
        if effective_linear_action != smoothing_state.get('last_effective_linear', 0):
            smoothing_state['action_switches_recent'] += 1
            # Only apply startup cost if the thruster is actually trying to engage (action > 0)
            if effective_linear_action > 0:
                 smoothing_state['startup_energy_debt'] += THRUSTER_STARTUP_ENERGY_COST
            
        if effective_rotation_action != smoothing_state.get('last_effective_rotation', 0):
            smoothing_state['action_switches_recent'] += 1
            # Rotational startup cost could also be added if desired, similar to linear

        # Apply switching penalty
        # Consider if SWITCHING_DETECTION_WINDOW should be used here to average switches over time
        if smoothing_state['action_switches_recent'] > 3:  # More than 3 switches recently (adjust threshold as needed)
            penalty = RAPID_SWITCHING_PENALTY * (smoothing_state['action_switches_recent'] - 3) # Penalize excess switches
            reward -= penalty
            probe['energy'] = max(0, probe['energy'] - penalty * 0.5) # Penalize energy too

        # Decay switching counter (e.g., per step or over SWITCHING_DETECTION_WINDOW)
        # For per-step decay:
        smoothing_state['action_switches_recent'] = max(0, smoothing_state['action_switches_recent'] - (1.0 / SWITCHING_DETECTION_WINDOW))


        # Apply startup energy debt
        if smoothing_state['startup_energy_debt'] > 0:
            debt_payment = min(smoothing_state['startup_energy_debt'], THRUSTER_STARTUP_ENERGY_COST * 0.1) # Pay small portion each step
            probe['energy'] = max(0, probe['energy'] - debt_payment)
            smoothing_state['startup_energy_debt'] -= debt_payment
            reward -= debt_payment * 0.01 # Small reward penalty for paying debt

        # Update last actions
        smoothing_state['last_effective_linear'] = effective_linear_action
        smoothing_state['last_effective_rotation'] = effective_rotation_action
        
        # Manage rotation timer and target action
        if effective_rotation_action > 0 and not is_low_power: # Action 0 is no torque
            if smoothing_state['rotation_timer'] <= 0: # Start new rotation
                smoothing_state['target_rotation_action'] = effective_rotation_action
                smoothing_state['rotation_timer'] = MIN_ROTATION_DURATION
        
        if smoothing_state['rotation_timer'] > 0:
            smoothing_state['rotation_timer'] -= 1
            if is_low_power:
                smoothing_state['rotation_timer'] = 0 # Stop rotation if low power
        
        if smoothing_state['rotation_timer'] <= 0:
            smoothing_state['target_rotation_action'] = 0

        # Update current_rotation_ramp (0.0 to 1.0)
        if smoothing_state['target_rotation_action'] > 0 and not is_low_power: # Ramp Up
            rotation_ramp_progress_up = min(1.0, (MIN_ROTATION_DURATION - smoothing_state['rotation_timer']) / max(1, ROTATION_RAMP_TIME))
            smoothing_state['current_rotation_ramp'] = rotation_ramp_progress_up
        else: # Ramp Down
            decrement_step = 1.0 / max(1, ROTATION_RAMP_TIME) # Ramp from 1.0 to 0.0 in RAMP_TIME steps
            smoothing_state['current_rotation_ramp'] = max(0.0, smoothing_state['current_rotation_ramp'] - decrement_step)

        # Apply actual torque
        actual_torque_applied = 0.0
        if smoothing_state['current_rotation_ramp'] > 0.01 and smoothing_state['target_rotation_action'] > 0 and not is_low_power:
            base_torque_magnitude = 0.0
            target_action = smoothing_state['target_rotation_action'] # 1=L_Low, 2=L_High, 3=R_Low, 4=R_High
            
            if target_action == 1: base_torque_magnitude = -TORQUE_MAGNITUDES[1]
            elif target_action == 2: base_torque_magnitude = -TORQUE_MAGNITUDES[2]
            elif target_action == 3: base_torque_magnitude = TORQUE_MAGNITUDES[1]
            elif target_action == 4: base_torque_magnitude = TORQUE_MAGNITUDES[2]
            
            actual_torque_applied = base_torque_magnitude * smoothing_state['current_rotation_ramp']

            if abs(actual_torque_applied) > 1e-4: # Apply if significant
                angular_acceleration = actual_torque_applied / MOMENT_OF_INERTIA
                probe['angular_velocity'] += angular_acceleration
                
                energy_cost_rotation = abs(actual_torque_applied) * ROTATIONAL_ENERGY_COST_FACTOR
                probe['energy'] = max(0, probe['energy'] - energy_cost_rotation)
                reward -= energy_cost_rotation * 0.01
        
        # --- Target Selection Logic --- (Using raw_target_select_action)
            can_switch_target = (self.step_count - probe.get('last_target_switch_time', 0)) >= TARGET_SWITCH_COOLDOWN
            
            if raw_target_select_action == 0: # Action to clear target
                if probe.get('current_target_id') is not None: # If there was an active target
                    probe['last_target_switch_time'] = self.step_count # Clearing a target also counts as a switch for cooldown
                probe['selected_target_info'] = None
                probe['distance_to_target_last_step'] = float('inf')
                probe['current_target_id'] = None
                probe['has_reached_current_target'] = False # Reset flag when target is cleared
            elif raw_target_select_action > 0: # Action to select a new target
                target_idx_in_observed_list = raw_target_select_action - 1
                
                observed_resource_distances = []
                for r_idx, resource in enumerate(self.resources):
                    if resource.amount > 0:
                        dist = self._distance(probe['position'], resource.position)
                        # Store the actual resource index (r_idx) as 'id'
                        observed_resource_distances.append({'dist': dist, 'id': r_idx, 'world_pos': resource.position})
                observed_resource_distances.sort(key=lambda r: r['dist'])

                if target_idx_in_observed_list < len(observed_resource_distances) and \
                   target_idx_in_observed_list < NUM_OBSERVED_RESOURCES_FOR_TARGETING:
                    
                    selected_res_info = observed_resource_distances[target_idx_in_observed_list]
                    new_target_resource_id = selected_res_info['id'] # This is the actual index in self.resources

                    # Check if switching to a *different* target
                    if new_target_resource_id != probe.get('current_target_id'):
                        if can_switch_target:
                            probe['energy'] = max(0, probe['energy'] - TARGET_SWITCH_ENERGY_COST)
                            reward -= TARGET_SWITCH_ENERGY_COST * 0.1 # Small penalty for switching
                            probe['selected_target_info'] = {
                                'type': 'resource', 'id': new_target_resource_id, # Store actual resource ID
                                'world_pos': selected_res_info['world_pos']
                            }
                            probe['distance_to_target_last_step'] = selected_res_info['dist']
                            probe['current_target_id'] = new_target_resource_id
                            probe['last_target_switch_time'] = self.step_count
                            probe['has_reached_current_target'] = False # Reset flag when new target is selected
                        else:
                            # Cannot switch yet (cooldown active), so no change to target, maybe a small penalty
                            reward -= 0.05 # Small penalty for attempting to switch during cooldown
                            pass # Target remains unchanged
                    else:
                        # Re-selecting the same target, no cost/cooldown, just update info if needed
                        probe['selected_target_info'] = {
                            'type': 'resource', 'id': new_target_resource_id,
                            'world_pos': selected_res_info['world_pos']
                        }
                        probe['distance_to_target_last_step'] = selected_res_info['dist']
                        # probe['current_target_id'] is already new_target_resource_id
                else:
                    # Invalid target selection index, clear target
                    if probe.get('current_target_id') is not None:
                         probe['last_target_switch_time'] = self.step_count
                    probe['selected_target_info'] = None
                    probe['distance_to_target_last_step'] = float('inf')
                    probe['current_target_id'] = None
                    probe['has_reached_current_target'] = False # Reset flag
        
        # --- Communication --- (Using raw_communicate_action)
        if raw_communicate_action > 0 and not is_low_power:
            reward += self._handle_communication(probe_id)
        
        # --- Replication --- (Using raw_replicate_action)
        if raw_replicate_action > 0 and probe['energy'] > REPLICATION_MIN_ENERGY and not is_low_power:
            reward += self._handle_replication(probe_id)
        
        # --- Standard Rewards & State Updates ---
        reward += 0.1 # Survival reward
        
        grid_pos = (int(probe['position'][0] // 50), int(probe['position'][1] // 50))
        if grid_pos not in probe['visited_positions']:
            probe['visited_positions'].add(grid_pos)
            reward += 1.0 # Exploration reward
        
        for res_idx, resource_node in enumerate(self.resources):
            if resource_node.amount > 0 and res_idx not in probe['discovered_resources']:
                dist_to_resource = self._distance(probe['position'], resource_node.position)
                if dist_to_resource < DISCOVERY_RANGE:
                    probe['discovered_resources'].add(res_idx)
                    # Value-Based Resource Discovery
                    reward += RESOURCE_DISCOVERY_REWARD_FACTOR * resource_node.amount
        
        if probe.get('selected_target_info') and probe['selected_target_info'].get('world_pos') is not None:
            target_world_pos = probe['selected_target_info']['world_pos']
            current_distance_to_target = self._distance(probe['position'], target_world_pos)
            
            if probe['distance_to_target_last_step'] != float('inf'):
                distance_delta = probe['distance_to_target_last_step'] - current_distance_to_target
                if distance_delta > 0: # If moved closer
                    # Non-linear reward: reward increases as 1/(distance + falloff)
                    # This means the reward for the *same* distance_delta is higher when closer.
                    # Ensure current_distance_to_target is not zero or too small to cause extreme rewards.
                    # PROXIMITY_REWARD_FALLOFF helps prevent division by zero and tunes the curve.
                    # A smaller PROXIMITY_REWARD_FALLOFF makes the reward spike more sharply at close distances.
                    # max(0.1, current_distance_to_target) prevents division by zero if distance is 0.
                    proximity_bonus_factor = (1.0 / (max(0.1, current_distance_to_target) + PROXIMITY_REWARD_FALLOFF))
                    
                    # The reward is the distance covered, scaled by the base factor, and then amplified by the proximity bonus.
                    # The (1 + proximity_bonus_factor * 5.0) term means the bonus can significantly increase the reward.
                    # Adjust the '5.0' to control the strength of the non-linear effect.
                    reward += distance_delta * TARGET_PROXIMITY_REWARD_FACTOR * (1 + proximity_bonus_factor * 5.0)
                elif distance_delta < 0: # Moved away from target
                    # Penalty for Moving Away from Target
                    reward -= abs(distance_delta) * MOVE_AWAY_FROM_TARGET_PENALTY_FACTOR
            
            # Reward for Reaching Target (one-time bonus)
            if not probe.get('has_reached_current_target', False) and current_distance_to_target < HARVEST_DISTANCE:
                reward += REACH_TARGET_BONUS
                probe['has_reached_current_target'] = True
                
            probe['distance_to_target_last_step'] = current_distance_to_target

        reward += self._handle_resource_collection(probe_id) # Resource collection
        
        probe['age'] += 1
        probe['energy'] = max(0, probe['energy'] - ENERGY_DECAY_RATE)

        # Energy Management Incentives
        if probe['energy'] > MAX_ENERGY * HIGH_ENERGY_THRESHOLD:
            reward += HIGH_ENERGY_REWARD_BONUS

        # Stronger 'Stay Alive' Signal
        if probe['energy'] > MAX_ENERGY * CRITICAL_ENERGY_THRESHOLD and probe['energy'] > 0:
            reward += STAY_ALIVE_REWARD_BONUS
        
        # --- DEBUG MODE: Energy Reset ---
        if DEBUG_MODE and probe['energy'] <= 0:
            probe['energy'] = DEBUG_ENERGY_RESET_VALUE
            # Optional: Could add a visual cue or print statement here if needed for debugging
            
        probe['total_reward'] += reward
        
        return reward
    
    def _handle_communication(self, probe_id: int) -> float:
        """Handle probe communication"""
        probe = self.probes[probe_id]
        
        # Find best resource to communicate
        best_resource = None
        best_dist = float('inf')
        
        for resource in self.resources:
            if resource.amount > 10:  # Only communicate about substantial resources
                dist = self._distance(probe['position'], resource.position)
                if dist < best_dist:
                    best_dist = dist
                    best_resource = resource
        
        if best_resource:
            message = Message(
                sender_id=probe_id,
                msg_type='RESOURCE_LOCATION',
                position=best_resource.position,
                resource_amount=best_resource.amount,
                timestamp=self.step_count
            )
            self.messages.append(message)
            return 2.0  # Communication reward
        
        return 0.0
    
    def _handle_replication(self, probe_id: int) -> float:
        """Handle probe replication"""
        if len(self.probes) >= MAX_PROBES:
            return 0.0
        
        probe = self.probes[probe_id]
        # Ensure replication cost doesn't take energy below 0, though REPLICATION_MIN_ENERGY check should prevent issues
        probe['energy'] = max(0, probe['energy'] - REPLICATION_COST)
        
        # Create new probe
        new_id = self.max_probe_id + 1
        new_pos = probe['position'] + np.random.normal(0, 20, 2)
        new_pos = self._wrap_position(new_pos)
        
        self.add_probe(new_id, new_pos, INITIAL_ENERGY, probe['generation'] + 1)
        
        return 20.0  # Large replication reward
    
    def _handle_resource_collection(self, probe_id: int) -> float:
        """Handle resource collection for a probe"""
        probe = self.probes[probe_id]
        reward = 0.0
        
        for resource in self.resources:
            dist = self._distance(probe['position'], resource.position)
            if dist < HARVEST_DISTANCE and resource.amount > 0:
                harvested = resource.harvest(HARVEST_RATE)
                self.total_resources_mined += harvested
                probe['energy'] = min(MAX_ENERGY, probe['energy'] + harvested)
                reward += harvested * 5.0  # Resource collection reward (base)
                
                # Sustained Mining Incentive
                # Check if the currently harvested resource is the probe's selected target
                selected_target_info = probe.get('selected_target_info')
                if selected_target_info and selected_target_info.get('type') == 'resource':
                    # Assuming resource objects themselves are not directly stored as IDs,
                    # we might need to compare positions or have a unique ID for resources if not using index.
                    # For now, let's assume selected_target_info['id'] is the index of the resource in self.resources
                    # This requires ensuring selected_target_info['id'] is indeed the index.
                    # A safer way if resource objects can change order or are not indexed directly:
                    # Compare resource.position with selected_target_info['world_pos']
                    if np.array_equal(resource.position, selected_target_info['world_pos']):
                         reward += SUSTAINED_MINING_REWARD_PER_STEP

                probe['is_mining_visual'] = True
                probe['mining_target_pos_visual'] = resource.position
        
        return reward
    
    def _update_physics(self):
        """Ultra-smooth physics with enhanced stability"""
        for probe_id, probe in self.probes.items():
            if not probe['alive']:
                continue

            # ENHANCED ROTATIONAL PHYSICS
            # Store previous angular velocity for smooth integration
            # prev_angular_vel = probe['angular_velocity'] # Not directly used in the provided replacement
            
            # Update angle with sub-step integration for smoothness
            dt = 1.0  # Time step
            probe['angle'] = (probe['angle'] + probe['angular_velocity'] * dt) % (2 * np.pi)
            
            # Progressive angular damping (stronger at higher speeds)
            speed_factor_angular = abs(probe['angular_velocity']) / MAX_ANGULAR_VELOCITY if MAX_ANGULAR_VELOCITY > 1e-6 else 0
            adaptive_damping = ANGULAR_DAMPING_FACTOR * (0.3 + 0.7 * speed_factor_angular)
            probe['angular_velocity'] *= (1 - adaptive_damping)
            
            # Smooth angular velocity limiting
            if abs(probe['angular_velocity']) > MAX_ANGULAR_VELOCITY:
                sign = np.sign(probe['angular_velocity'])
                excess = abs(probe['angular_velocity']) - MAX_ANGULAR_VELOCITY
                # Gently nudge back towards MAX_ANGULAR_VELOCITY instead of hard clip or proportional reduction
                probe['angular_velocity'] = sign * (MAX_ANGULAR_VELOCITY + excess * 0.1)

            # ENHANCED LINEAR PHYSICS
            # Store previous velocity for smoothing
            # prev_velocity = probe['velocity'].copy() # Not directly used in the provided replacement
            
            # Update position with enhanced integration
            probe['position'] += probe['velocity'] * dt
            probe['position'] = self._wrap_position(probe['position'])
            
            # ULTRA-SOFT VELOCITY LIMITING with exponential decay
            velocity_magnitude = np.linalg.norm(probe['velocity'])
            if velocity_magnitude > MAX_VELOCITY:
                if MAX_VELOCITY > 1e-6:
                    # Exponential approach to max velocity
                    excess_ratio = velocity_magnitude / MAX_VELOCITY
                    # Decay factor should be < 1 if excess_ratio > 1
                    decay_factor = 1.0 / (1.0 + (excess_ratio - 1.0) * 0.05)
                    probe['velocity'] *= decay_factor
            
            # ADAPTIVE SPACE FRICTION (varies with velocity)
            speed_factor_linear = velocity_magnitude / MAX_VELOCITY if MAX_VELOCITY > 1e-6 else 0
            adaptive_friction = 0.9998 + (0.9995 - 0.9998) * speed_factor_linear
            probe['velocity'] *= adaptive_friction
    
    def _regenerate_resources(self):
        """Regenerate resources over time"""
        for resource in self.resources:
            resource.regenerate()
    
    def _cleanup_old_messages(self):
        """Remove old messages"""
        self.messages = [msg for msg in self.messages 
                        if self.step_count - msg.timestamp < 50]
    
    def _distance(self, pos1, pos2):
        """Calculate distance between two positions with wraparound"""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        
        # Handle wraparound
        dx = min(dx, self.world_width - dx)
        dy = min(dy, self.world_height - dy)
        
        return np.sqrt(dx*dx + dy*dy)
    
    def _wrap_position(self, position: np.ndarray) -> np.ndarray:
        """Clamps the position to the world boundaries."""
        position[0] = np.clip(position[0], 0, self.world_width)
        position[1] = np.clip(position[1], 0, self.world_height)
        return position
    
    def reset(self):
        """Reset the environment"""
        self.probes = {}
        self.messages = []
        self.step_count = 0
        self.max_probe_id = 0
        self._generate_resources()
        
        # Add initial probes
        for i in range(INITIAL_PROBES):
            pos = (random.uniform(0, self.world_width),
                   random.uniform(0, self.world_height))
            self.add_probe(i, pos)
        
        # Return initial observations
        observations = {}
        for probe_id in self.probes:
            observations[probe_id] = self.get_observation(probe_id)
        
        return observations