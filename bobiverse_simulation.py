# config.py
import numpy as np

# World Configuration
WORLD_WIDTH = 1000
WORLD_HEIGHT = 1000
MAX_PROBES = 20
INITIAL_PROBES = 3

# Resource Configuration
RESOURCE_COUNT = 15
RESOURCE_MIN_AMOUNT = 50
RESOURCE_MAX_AMOUNT = 200
RESOURCE_REGEN_RATE = 0.01  # units per step
HARVEST_RATE = 2.0
HARVEST_DISTANCE = 20

# Probe Configuration
MAX_ENERGY = 100
INITIAL_ENERGY = 50
REPLICATION_COST = 40
REPLICATION_MIN_ENERGY = 60
MAX_VELOCITY = 5.0
THRUST_POWER = [0, 1, 2]
ENERGY_DECAY_RATE = 0.05

# Communication
COMM_RANGE = 100
MESSAGE_TYPES = ['RESOURCE_LOCATION']

# Training Configuration
EPISODE_LENGTH = 5000
LEARNING_RATE = 3e-4
BATCH_SIZE = 64

# Visualization
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

#==============================================================================

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

#==============================================================================

# probe.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

class ProbeAgent:
    def __init__(self, probe_id: int, environment, parent_model=None):
        self.probe_id = probe_id
        self.environment = environment
        self.model = None
        self.generation = 0
        
        # Create individual environment wrapper for this probe
        self.vec_env = DummyVecEnv([lambda: ProbeEnvWrapper(environment, probe_id)])
        
        # Initialize RL model
        if parent_model is not None:
            # Inherit from parent with mutation
            self.model = self._inherit_model(parent_model)
            self.generation = parent_model.generation + 1
        else:
            # Create new model
            self.model = PPO(
                "MlpPolicy",
                self.vec_env,
                learning_rate=LEARNING_RATE,
                n_steps=2048,
                batch_size=BATCH_SIZE,
                verbose=0
            )
    
    def _inherit_model(self, parent_model):
        """Create a new model inheriting from parent with mutations"""
        # Create new model with same architecture
        new_model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=LEARNING_RATE,
            n_steps=2048,
            batch_size=BATCH_SIZE,
            verbose=0
        )
        
        # Copy parent weights with small mutations
        parent_params = parent_model.model.policy.state_dict()
        new_params = {}
        
        for name, param in parent_params.items():
            # Add small random mutations (1% of parameter value)
            mutation = torch.randn_like(param) * 0.01 * torch.abs(param)
            new_params[name] = param + mutation
        
        new_model.policy.load_state_dict(new_params)
        return new_model
    
    def predict(self, observation):
        """Get action prediction from the model"""
        action, _ = self.model.predict(observation, deterministic=False)
        return action
    
    def learn(self, total_timesteps=10000):
        """Train the agent"""
        self.model.learn(total_timesteps=total_timesteps)
    
    def save(self, path):
        """Save the model"""
        self.model.save(path)
    
    def load(self, path):
        """Load the model"""
        self.model = PPO.load(path, env=self.vec_env)


class ProbeEnvWrapper(gym.Env):
    """Wrapper to make multi-agent environment work with single-agent RL"""
    def __init__(self, multi_env, probe_id):
        super().__init__()
        self.multi_env = multi_env
        self.probe_id = probe_id
        self.observation_space = multi_env.observation_space
        self.action_space = multi_env.action_space
    
    def step(self, action):
        # Only send action for this probe
        actions = {self.probe_id: action}
        obs_dict, reward_dict, done_dict, info_dict = self.multi_env.step(actions)
        
        # Return single agent format
        obs = obs_dict.get(self.probe_id, np.zeros(self.observation_space.shape))
        reward = reward_dict.get(self.probe_id, 0.0)
        done = done_dict.get(self.probe_id, True)
        info = info_dict.get(self.probe_id, {})
        
        return obs, reward, done, info
    
    def reset(self):
        obs_dict = self.multi_env.reset()
        return obs_dict.get(self.probe_id, np.zeros(self.observation_space.shape))

#==============================================================================

# visualization.py
import pygame
import numpy as np
from typing import Dict, List
import math
from config import *

class Visualization:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Bobiverse RL Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 16)
        
        # Colors
        self.colors = {
            'background': (10, 10, 30),
            'resource': (0, 255, 0),
            'probe_base': (100, 150, 255),
            'communication': (255, 255, 0),
            'trail': (80, 80, 150),
            'ui_text': (255, 255, 255),
            'ui_bg': (50, 50, 50)
        }
        
        # Scaling for world to screen
        self.scale_x = (SCREEN_WIDTH - 200) / WORLD_WIDTH
        self.scale_y = SCREEN_HEIGHT / WORLD_HEIGHT
        
        # Trail storage
        self.probe_trails = {}
        
    def world_to_screen(self, world_pos):
        """Convert world coordinates to screen coordinates"""
        return (
            int(world_pos[0] * self.scale_x),
            int(world_pos[1] * self.scale_y)
        )
    
    def render(self, environment, probe_agents: Dict = None):
        """Render the current state of the simulation"""
        self.screen.fill(self.colors['background'])
        
        # Draw resources
        self._draw_resources(environment.resources)
        
        # Draw probes
        self._draw_probes(environment.probes, environment.messages)
        
        # Draw UI
        self._draw_ui(environment, probe_agents)
        
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def _draw_resources(self, resources):
        """Draw resource nodes"""
        for resource in resources:
            if resource.amount > 0:
                screen_pos = self.world_to_screen(resource.position)
                # Size based on remaining amount
                size = max(3, int(resource.amount / RESOURCE_MAX_AMOUNT * 15))
                pygame.draw.circle(self.screen, self.colors['resource'], 
                                 screen_pos, size)
                
                # Draw amount text
                if resource.amount > 10:
                    text = self.small_font.render(f"{int(resource.amount)}", 
                                                True, (255, 255, 255))
                    self.screen.blit(text, (screen_pos[0] + size + 2, 
                                          screen_pos[1] - 8))
    
    def _draw_probes(self, probes, messages):
        """Draw probes with trails and communication links"""
        generation_colors = [
            (100, 150, 255),  # Gen 0 - Blue
            (255, 100, 100),  # Gen 1 - Red  
            (100, 255, 100),  # Gen 2 - Green
            (255, 255, 100),  # Gen 3 - Yellow
            (255, 100, 255),  # Gen 4 - Magenta
            (100, 255, 255),  # Gen 5 - Cyan
        ]
        
        # Update and draw trails
        for probe_id, probe in probes.items():
            if not probe['alive']:
                continue
                
            screen_pos = self.world_to_screen(probe['position'])
            
            # Update trail
            if probe_id not in self.probe_trails:
                self.probe_trails[probe_id] = []
            
            self.probe_trails[probe_id].append(screen_pos)
            if len(self.probe_trails[probe_id]) > 50:
                self.probe_trails[probe_id].pop(0)
            
            # Draw trail
            if len(self.probe_trails[probe_id]) > 1:
                for i in range(1, len(self.probe_trails[probe_id])):
                    alpha = i / len(self.probe_trails[probe_id])
                    color = tuple(int(c * alpha) for c in self.colors['trail'])
                    if i > 0:
                        pygame.draw.line(self.screen, color,
                                       self.probe_trails[probe_id][i-1],
                                       self.probe_trails[probe_id][i], 1)
        
        # Draw communication links
        current_time = len(messages)  # Simplified timestamp
        for message in messages[-10:]:  # Show recent messages
            sender_pos = None
            for probe_id, probe in probes.items():
                if probe_id == message.sender_id and probe['alive']:
                    sender_pos = self.world_to_screen(probe['position'])
                    break
            
            if sender_pos:
                resource_pos = self.world_to_screen(message.position)
                pygame.draw.line(self.screen, self.colors['communication'],
                               sender_pos, resource_pos, 1)
        
        # Draw probes
        for probe_id, probe in probes.items():
            if not probe['alive']:
                continue
                
            screen_pos = self.world_to_screen(probe['position'])
            generation = min(probe['generation'], len(generation_colors) - 1)
            color = generation_colors[generation]
            
            # Probe size based on energy
            size = max(3, int(probe['energy'] / MAX_ENERGY * 8) + 3)
            pygame.draw.circle(self.screen, color, screen_pos, size)
            pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, size, 1)
            
            # Draw probe ID
            text = self.small_font.render(str(probe_id), True, (255, 255, 255))
            self.screen.blit(text, (screen_pos[0] + size + 2, screen_pos[1] - 8))
            
            # Draw energy bar
            bar_width = 20
            bar_height = 4
            bar_x = screen_pos[0] - bar_width // 2
            bar_y = screen_pos[1] - size - 8
            
            # Background
            pygame.draw.rect(self.screen, (100, 100, 100),
                           (bar_x, bar_y, bar_width, bar_height))
            
            # Energy level
            energy_width = int((probe['energy'] / MAX_ENERGY) * bar_width)
            energy_color = (255, 255, 0) if probe['energy'] > 30 else (255, 100, 100)
            pygame.draw.rect(self.screen, energy_color,
                           (bar_x, bar_y, energy_width, bar_height))
    
    def _draw_ui(self, environment, probe_agents):
        """Draw UI information panel"""
        ui_x = SCREEN_WIDTH - 190
        ui_y = 10
        
        # Background panel
        pygame.draw.rect(self.screen, self.colors['ui_bg'],
                        (ui_x - 5, ui_y - 5, 185, SCREEN_HEIGHT - 10))
        
        # Statistics
        alive_probes = sum(1 for probe in environment.probes.values() if probe['alive'])
        total_energy = sum(probe['energy'] for probe in environment.probes.values() if probe['alive'])
        avg_energy = total_energy / max(alive_probes, 1)
        
        # Generation statistics
        generations = {}
        for probe in environment.probes.values():
            if probe['alive']:
                gen = probe['generation']
                generations[gen] = generations.get(gen, 0) + 1
        
        stats = [
            f"Step: {environment.step_count}",
            f"Alive Probes: {alive_probes}",
            f"Total Probes: {len(environment.probes)}",
            f"Avg Energy: {avg_energy:.1f}",
            f"Resources: {len([r for r in environment.resources if r.amount > 0])}",
            f"Messages: {len(environment.messages)}",
            "",
            "Generations:"
        ]
        
        for gen, count in sorted(generations.items()):
            stats.append(f"  Gen {gen}: {count}")
        
        # Individual probe info
        stats.append("")
        stats.append("Active Probes:")
        for probe_id, probe in environment.probes.items():
            if probe['alive']:
                stats.append(f"  #{probe_id}: E{probe['energy']:.0f} G{probe['generation']}")
        
        # Render text
        y_offset = ui_y
        for stat in stats:
            text = self.small_font.render(stat, True, self.colors['ui_text'])
            self.screen.blit(text, (ui_x, y_offset))
            y_offset += 16
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

#==============================================================================

# main.py
import numpy as np
import random
import time
from typing import Dict
import pickle
import os

class BobiverseSimulation:
    def __init__(self):
        self.environment = SpaceEnvironment()
        self.visualization = Visualization()
        self.probe_agents = {}
        self.running = True
        self.training_mode = False
        self.episode_count = 0
        
    def initialize_agents(self):
        """Initialize RL agents for initial probes"""
        for probe_id in self.environment.probes.keys():
            self.probe_agents[probe_id] = ProbeAgent(probe_id, self.environment)
    
    def run_episode(self, max_steps=EPISODE_LENGTH, render=True, train=False):
        """Run a single episode of the simulation"""
        observations = self.environment.reset()
        self.initialize_agents()
        
        step_count = 0
        episode_rewards = {pid: 0 for pid in self.probe_agents.keys()}
        
        while step_count < max_steps and self.running:
            if render and not self.visualization.handle_events():
                self.running = False
                break
            
            # Get actions from all agents
            actions = {}
            for probe_id, agent in self.probe_agents.items():
                if probe_id in observations:
                    action = agent.predict(observations[probe_id])
                    actions[probe_id] = action
            
            # Execute environment step
            observations, rewards, dones, infos = self.environment.step(actions)
            
            # Handle new probes from replication
            self._handle_new_probes()
            
            # Accumulate rewards
            for probe_id, reward in rewards.items():
                if probe_id in episode_rewards:
                    episode_rewards[probe_id] += reward
            
            # Render if requested
            if render:
                self.visualization.render(self.environment, self.probe_agents)
                time.sleep(0.01)  # Control simulation speed
            
            step_count += 1
        
        # Training step for surviving agents
        if train:
            for agent in self.probe_agents.values():
                agent.learn(total_timesteps=1000)
        
        return episode_rewards, step_count
    
    def _handle_new_probes(self):
        """Create agents for newly replicated probes"""
        for probe_id in self.environment.probes.keys():
            if probe_id not in self.probe_agents:
                # Find parent probe (probe with highest generation in same family)
                parent_agent = None
                parent_generation = -1
                
                for existing_id, agent in self.probe_agents.items():
                    if (existing_id in self.environment.probes and 
                        self.environment.probes[existing_id]['generation'] < 
                        self.environment.probes[probe_id]['generation'] and
                        self.environment.probes[existing_id]['generation'] > parent_generation):
                        parent_agent = agent
                        parent_generation