# main.py
import numpy as np
import random
import time
from typing import Dict
import pickle
import os

from config import *
from environment import SpaceEnvironment
from probe import ProbeAgent
from visualization import Visualization

class BobiverseSimulation:
    def __init__(self):
        self.environment = SpaceEnvironment()
        self.visualization = Visualization()
        self.probe_agents: Dict[int, ProbeAgent] = {} # Added type hint
        self.running = True
        self.training_mode = False # Can be set to True to enable training
        self.episode_count = 0
        
    def initialize_agents(self):
        """Initialize RL agents for initial probes"""
        for probe_id in self.environment.probes.keys():
            if probe_id not in self.probe_agents: # Ensure agent is not re-initialized
                self.probe_agents[probe_id] = ProbeAgent(probe_id, self.environment)
    
    def run_episode(self, max_steps=EPISODE_LENGTH, render=True, train=False):
        """Run a single episode of the simulation"""
        observations = self.environment.reset()
        self.probe_agents = {} # Reset agents for new episode
        self.initialize_agents()
        
        step_count = 0
        episode_rewards = {pid: 0 for pid in self.probe_agents.keys()}
        
        while step_count < max_steps and self.running:
            if render and not self.visualization.handle_events():
                self.running = False
                break
            
            # Get actions from all agents
            actions = {}
            current_observations = {} # Store current observations for all live probes

            # First, get observations for all live probes
            for probe_id in list(self.probe_agents.keys()): # Iterate over a copy of keys
                if probe_id in self.environment.probes and self.environment.probes[probe_id]['alive']:
                    # Ensure observation is available from the environment's reset or step
                    if probe_id in observations:
                         current_observations[probe_id] = observations[probe_id]
                    else:
                        # This case should ideally not happen if observations are correctly passed
                        # For safety, try to get a fresh observation if missing
                        current_observations[probe_id] = self.environment.get_observation(probe_id)


            for probe_id, agent in self.probe_agents.items():
                if probe_id in current_observations: # Check if probe has a current observation
                    action = agent.predict(current_observations[probe_id])
                    actions[probe_id] = action
            
            # Execute environment step
            next_observations, rewards, dones, infos = self.environment.step(actions)
            observations = next_observations # Update observations for the next iteration
            
            # Handle new probes from replication
            self._handle_new_probes()
            
            # Accumulate rewards
            for probe_id, reward in rewards.items():
                if probe_id in episode_rewards:
                    episode_rewards[probe_id] += reward
                else: # For newly created probes
                    episode_rewards[probe_id] = reward

            # Remove agents for probes that are no longer alive
            for probe_id in list(self.probe_agents.keys()):
                if probe_id not in self.environment.probes or not self.environment.probes[probe_id]['alive']:
                    if probe_id in dones and dones[probe_id]: # Check if probe is done
                         del self.probe_agents[probe_id]
                         if probe_id in episode_rewards: # Clean up rewards dict
                             del episode_rewards[probe_id]


            # Render if requested
            if render:
                self.visualization.render(self.environment, self.probe_agents)
                time.sleep(0.01)  # Control simulation speed
            
            step_count += 1
        
        # Training step for surviving agents
        if train:
            for agent_id in list(self.probe_agents.keys()): # Iterate over a copy of keys
                if agent_id in self.probe_agents: # Check if agent still exists
                    self.probe_agents[agent_id].learn(total_timesteps=1000) # Reduced for faster episodes
        
        self.episode_count +=1
        print(f"Episode {self.episode_count} finished after {step_count} steps.")
        avg_reward = sum(episode_rewards.values()) / max(1, len(episode_rewards))
        print(f"Average reward: {avg_reward}")
        return episode_rewards, step_count
    
    def _handle_new_probes(self):
        """Create agents for newly replicated probes"""
        for probe_id, probe_data in self.environment.probes.items():
            if probe_id not in self.probe_agents and probe_data['alive']:
                parent_agent = None
                # Attempt to find a parent agent to inherit from
                # This logic might need refinement based on how parentage is tracked/desired
                # For now, new probes start with a fresh model or a default inherited one
                # Simplified: if there are other agents, pick one as a "parent" for model structure
                # A more robust way would be to pass parent_id during replication in environment
                if self.probe_agents:
                    # Heuristic: find the agent with the highest generation that is not the new probe itself
                    # This is a simplification. True parent tracking would be more complex.
                    potential_parents = [
                        ag for ag_id, ag in self.probe_agents.items()
                        if ag_id != probe_id and ag.generation < probe_data['generation']
                    ]
                    if potential_parents:
                         parent_agent = max(potential_parents, key=lambda ag: ag.generation, default=None)

                self.probe_agents[probe_id] = ProbeAgent(probe_id, self.environment, parent_model=parent_agent)
                print(f"Initialized new agent for probe {probe_id}, generation {probe_data['generation']}.")


if __name__ == "__main__":
    simulation = BobiverseSimulation()
    
    num_episodes = 5
    for i in range(num_episodes):
        print(f"\n--- Starting Episode {i+1}/{num_episodes} ---")
        # Set train=True if you want agents to learn
        rewards, steps = simulation.run_episode(max_steps=2000, render=True, train=simulation.training_mode) 
        
        if not simulation.running:
            print("Simulation terminated by user.")
            break
            
    print("\nSimulation finished.")
    # Optionally save models
    # save_dir = "saved_models"
    # os.makedirs(save_dir, exist_ok=True)
    # for probe_id, agent in simulation.probe_agents.items():
    #     agent.save(os.path.join(save_dir, f"probe_{probe_id}_gen{agent.generation}.zip"))
    # print(f"Models saved to {save_dir}")