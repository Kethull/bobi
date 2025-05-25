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
            'distance_to_target_last_step': float('inf') # For target proximity reward
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

        msg_idx = 6 + NUM_OBSERVED_RESOURCES_FOR_TARGETING * 3 + 2 * 2 # Should be 19
        obs[msg_idx] = min(len(recent_messages) / 5.0, 1.0)

        target_active_idx = msg_idx + 1 # Should be 20
        target_rel_pos_idx = msg_idx + 2 # Should be 21 (for x), 22 (for y) - wait, this is wrong.
                                         # If OBS_SIZE is 22, max index is 21.
                                         # target_active_idx = 19
                                         # target_rel_pos_idx_start = 20 (covers 20, 21)

        target_active_idx = 19 # Corrected index for OBS_SIZE = 22
        target_rel_pos_start_idx = 20 # Corrected index for OBS_SIZE = 22

        if probe.get('selected_target_info') and probe['selected_target_info'].get('world_pos') is not None:
            obs[target_active_idx] = 1.0
            target_world_pos = probe['selected_target_info']['world_pos']
            rel_pos_to_target = np.array(target_world_pos) - probe['position']
            rel_pos_to_target = self._wrap_position(rel_pos_to_target)
            obs[target_rel_pos_start_idx : target_rel_pos_start_idx+2] = rel_pos_to_target / np.array([self.world_width, self.world_height])
        else:
            obs[target_active_idx] = 0.0
            obs[target_rel_pos_start_idx : target_rel_pos_start_idx+2] = 0.0 # No target, zero out rel_pos
            
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
        """Process a single probe's action and return reward"""
        probe = self.probes[probe_id]
        reward = 0.0
        probe['is_thrusting_visual'] = False # Reset visual flag each step
        probe['thrust_power_visual'] = 0     # Reset visual thrust power
        probe['is_mining_visual'] = False    # Reset mining visual flag
        probe['mining_target_pos_visual'] = None # Reset mining target

        is_low_power = probe['energy'] <= 0

        if is_low_power:
            reward -= LOW_POWER_PENALTY # Apply penalty for being in low power mode
            # Probe cannot perform actions that cost energy if in low power mode
            thrust_dir = 0
            communicate = 0
            replicate = 0
            # Note: Resource collection can still happen passively below

        # Parse action (actions might have been overridden if low_power)
        # Action space now: [thrust_dir, thrust_power, communicate, replicate, target_select]
        thrust_dir_actual, thrust_power_actual, communicate_actual, replicate_actual, target_select_action = action
        
        if is_low_power:
            thrust_dir_actual = 0
            communicate_actual = 0
            replicate_actual = 0
            # target_select_action remains, but probe won't move towards it if low power.
            # Or, we can also nullify target selection if low power:
            # target_select_action = 0 # Optional: clear target if low power
        
        # --- Target Selection Logic ---
        if not is_low_power: # Allow target selection only if not in low power
            if target_select_action == 0:
                probe['selected_target_info'] = None
                probe['distance_to_target_last_step'] = float('inf')
            elif target_select_action > 0:
                # Find the (target_select_action)-th nearest resource
                # This re-queries nearest resources, could be optimized by passing from get_observation if needed
                observed_resource_distances = []
                for r_idx, resource in enumerate(self.resources):
                    if resource.amount > 0:
                        dist = self._distance(probe['position'], resource.position)
                        observed_resource_distances.append({'dist': dist, 'id': r_idx, 'world_pos': resource.position})
                
                observed_resource_distances.sort(key=lambda r: r['dist'])
                
                target_idx_in_observed_list = target_select_action - 1 # 1-based action to 0-based list index
                if target_idx_in_observed_list < len(observed_resource_distances) and target_idx_in_observed_list < NUM_OBSERVED_RESOURCES_FOR_TARGETING:
                    selected_res_info = observed_resource_distances[target_idx_in_observed_list]
                    probe['selected_target_info'] = {
                        'type': 'resource',
                        'id': selected_res_info['id'], # Actual index in self.resources
                        'world_pos': selected_res_info['world_pos']
                    }
                    probe['distance_to_target_last_step'] = selected_res_info['dist']
                    # print(f"Probe {probe_id} targeted resource {selected_res_info['id']} at {selected_res_info['world_pos']}") # Debug
                else:
                    # Invalid target selection (e.g., fewer than target_select_action resources available)
                    # Clear target if selection was invalid
                    probe['selected_target_info'] = None
                    probe['distance_to_target_last_step'] = float('inf')
        
        # Apply thrust (Newtonian physics)
        if thrust_dir_actual > 0:  # 0 is no thrust
            directions = [
                (0, 1), (1, 1), (1, 0), (1, -1),
                (0, -1), (-1, -1), (-1, 0), (-1, 1)
            ]
            direction_vector = np.array(directions[thrust_dir_actual - 1], dtype=np.float32)
            force_magnitude = THRUST_FORCE[thrust_power_actual] # Use THRUST_FORCE from config
            
            if force_magnitude > 0: # Should only happen if not is_low_power
                force_vector = direction_vector * force_magnitude
                
                acceleration_vector = force_vector / probe['mass']
                probe['velocity'] += acceleration_vector
                
                energy_cost = force_magnitude * THRUST_ENERGY_COST_FACTOR
                probe['energy'] = max(0, probe['energy'] - energy_cost) # Ensure energy doesn't go below 0
                reward -= energy_cost * 0.01
                
                probe['is_thrusting_visual'] = True # Set visual flag
                probe['thrust_power_visual'] = thrust_power_actual # Store thrust power for visual
        
        # Communication
        if communicate_actual > 0: # Should only happen if not is_low_power
            reward += self._handle_communication(probe_id)
        
        # Replication
        if replicate_actual > 0 and probe['energy'] > REPLICATION_MIN_ENERGY: # Check energy > REPLICATION_MIN_ENERGY
            reward += self._handle_replication(probe_id) # Energy cost handled within
        
        # Survival reward
        reward += 0.1
        
        # Exploration reward
        grid_pos = (int(probe['position'][0] // 50), int(probe['position'][1] // 50))
        if grid_pos not in probe['visited_positions']:
            probe['visited_positions'].add(grid_pos)
            reward += 1.0
        
        # Resource discovery reward
        for res_idx, resource_node in enumerate(self.resources):
            if resource_node.amount > 0 and res_idx not in probe['discovered_resources']:
                dist_to_resource = self._distance(probe['position'], resource_node.position)
                if dist_to_resource < DISCOVERY_RANGE:
                    probe['discovered_resources'].add(res_idx)
                    reward += RESOURCE_DISCOVERY_REWARD
        
        # Target Proximity Reward
        if probe.get('selected_target_info') and probe['selected_target_info'].get('world_pos') is not None:
            target_world_pos = probe['selected_target_info']['world_pos']
            current_distance_to_target = self._distance(probe['position'], target_world_pos)
            
            if probe['distance_to_target_last_step'] != float('inf'): # Avoid reward on first step of targeting
                distance_delta = probe['distance_to_target_last_step'] - current_distance_to_target
                if distance_delta > 0: # Got closer
                    reward += distance_delta * TARGET_PROXIMITY_REWARD_FACTOR
            
            probe['distance_to_target_last_step'] = current_distance_to_target

        # Resource collection
        reward += self._handle_resource_collection(probe_id)
        
        # Update probe state
        probe['age'] += 1
        probe['energy'] = max(0, probe['energy'] - ENERGY_DECAY_RATE) # Ensure energy doesn't go below 0 from decay
        probe['total_reward'] += reward
        
        # Probe no longer "dies" by having alive set to False or a large death penalty.
        # The penalty is now continuous if is_low_power.
        
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
                probe['energy'] = min(MAX_ENERGY, probe['energy'] + harvested)
                reward += harvested * 5.0  # Resource collection reward
                probe['is_mining_visual'] = True
                probe['mining_target_pos_visual'] = resource.position
        
        return reward
    
    def _update_physics(self):
        """Update probe positions based on velocity"""
        for probe in self.probes.values():
            if probe['alive']:
                probe['position'] += probe['velocity']
                probe['position'] = self._wrap_position(probe['position'])
                # probe['velocity'] *= 0.95  # Friction removed for Newtonian motion
                
                # Optional: Clamp velocity to the new MAX_VELOCITY as a safety/normalization cap
                probe['velocity'] = np.clip(probe['velocity'], -MAX_VELOCITY, MAX_VELOCITY)
    
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
    
    def _wrap_position(self, pos):
        """Wrap position within world boundaries"""
        return np.array([
            pos[0] % self.world_width,
            pos[1] % self.world_height
        ])
    
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