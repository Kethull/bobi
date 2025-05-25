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
            shape=(19,),  # pos(2) + vel(2) + energy(1) + age(1) + resources(9) + probes(4) + messages(1)
            dtype=np.float32
        )
        
        # Action space: [thrust_dir(9), thrust_power(3), communicate(2), replicate(2)]
        self.action_space = spaces.MultiDiscrete([9, 3, 2, 2])
        
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
            'alive': True
        }
        self.max_probe_id = max(self.max_probe_id, probe_id)
    
    def get_observation(self, probe_id: int) -> np.ndarray:
        """Get observation for a specific probe"""
        probe = self.probes[probe_id]
        obs = np.zeros(19, dtype=np.float32)
        
        # Own state (6 values)
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
        for i in range(3):
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
        for i in range(2):
            base_idx = 15 + i * 2
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
        obs[18] = min(len(recent_messages) / 5.0, 1.0)  # Normalize message count
        
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
        
        # Parse action
        thrust_dir, thrust_power, communicate, replicate = action
        
        # Apply thrust
        if thrust_dir > 0:  # 0 is no thrust
            directions = [
                (0, 1), (1, 1), (1, 0), (1, -1),
                (0, -1), (-1, -1), (-1, 0), (-1, 1)
            ]
            dx, dy = directions[thrust_dir - 1]
            power = THRUST_POWER[thrust_power]
            
            probe['velocity'] += np.array([dx * power, dy * power]) * 0.1
            probe['velocity'] = np.clip(probe['velocity'], -MAX_VELOCITY, MAX_VELOCITY)
            
            # Energy cost for movement
            energy_cost = power * 0.1
            probe['energy'] -= energy_cost
            reward -= energy_cost * 0.1  # Small penalty for energy use
        
        # Communication
        if communicate > 0:
            reward += self._handle_communication(probe_id)
        
        # Replication
        if replicate > 0 and probe['energy'] > REPLICATION_MIN_ENERGY:
            reward += self._handle_replication(probe_id)
        
        # Survival reward
        reward += 0.1
        
        # Exploration reward
        grid_pos = (int(probe['position'][0] // 50), int(probe['position'][1] // 50))
        if grid_pos not in probe['visited_positions']:
            probe['visited_positions'].add(grid_pos)
            reward += 1.0
        
        # Resource collection
        reward += self._handle_resource_collection(probe_id)
        
        # Update probe state
        probe['age'] += 1
        probe['energy'] -= ENERGY_DECAY_RATE
        probe['total_reward'] += reward
        
        # Check if probe dies
        if probe['energy'] <= 0:
            probe['alive'] = False
            reward -= 10  # Death penalty
        
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
        probe['energy'] -= REPLICATION_COST
        
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
        
        return reward
    
    def _update_physics(self):
        """Update probe positions based on velocity"""
        for probe in self.probes.values():
            if probe['alive']:
                probe['position'] += probe['velocity']
                probe['position'] = self._wrap_position(probe['position'])
                probe['velocity'] *= 0.95  # Friction
    
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