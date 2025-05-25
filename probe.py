# probe.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
from config import * # Added import for config variables
import gym # Added import for gym.Env

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