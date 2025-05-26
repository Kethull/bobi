# main.py
import numpy as np
import random
import time
from typing import Dict
import pickle
import os
import psutil # For memory monitoring

from config import config # Use the global config instance
from environment import SolarSystemEnvironment
from probe import ProbeAgent
from visualization import Visualization

class BobiverseSimulation:
    def __init__(self):
        self.environment = SolarSystemEnvironment() # Changed from SpaceEnvironment
        self.visualization = Visualization()
        self.probe_agents: Dict[int, ProbeAgent] = {} # Added type hint
        self.running = True
        self.training_mode = True # Can be set to True to enable training
        self.episode_count = 0
        self.process = psutil.Process(os.getpid()) # Get current process for memory monitoring
        
    def initialize_agents(self):
        """Initialize RL agents for initial probes"""
        for probe_id in self.environment.probes.keys():
            if probe_id not in self.probe_agents: # Ensure agent is not re-initialized
                self.probe_agents[probe_id] = ProbeAgent(probe_id, self.environment)
    
    def run_episode(self, max_steps=None, render=True, train=False): # max_steps from config
        """Run a single episode of the simulation"""
        if max_steps is None:
            max_steps = config.RL.EPISODE_LENGTH_STEPS

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

            # First, get observations for all probes that are still in the environment
            # (The 'alive' flag is no longer used to determine if a probe is active for observation/action)
            for probe_id in list(self.probe_agents.keys()): # Iterate over a copy of keys
                if probe_id in self.environment.probes: # Check if probe exists in environment
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

            # Agents are no longer removed during an episode as probes don't "die"
            # They persist until the episode ends.
            # The 'dones' flag from environment.step() will signal episode end for all.
            # for probe_id in list(self.probe_agents.keys()):
            #     if probe_id not in self.environment.probes or not self.environment.probes[probe_id]['alive']:
            #         if probe_id in dones and dones[probe_id]:
            #              del self.probe_agents[probe_id]
            #              if probe_id in episode_rewards:
            #                  del episode_rewards[probe_id]


            # Render if requested
            if render:
                self.visualization.render(self.environment, self.probe_agents)
                time.sleep(0.01)  # Control simulation speed
            
            # Memory Usage Monitoring
            if step_count % config.Monitoring.MEMORY_CHECK_INTERVAL_STEPS == 0:
                memory_mb = self.process.memory_info().rss / (1024 * 1024)
                if memory_mb > config.Monitoring.MEMORY_USAGE_WARN_MB:
                    print(f"WARNING: High memory usage: {memory_mb:.2f} MB at step {step_count}")
                elif config.Debug.DEBUG_MODE: # Optional: print memory usage in debug mode
                    print(f"DEBUG: Memory usage: {memory_mb:.2f} MB at step {step_count}")

            step_count += 1
        
        # Training step for surviving agents
        if train:
            # All agents that existed throughout the episode (or were created) get a chance to learn
            # Probes in low power mode will have received different rewards, influencing their learning
            for agent_id, agent in self.probe_agents.items():
                # No need to check if agent_id in self.probe_agents again, as we iterate over items
                agent.learn(total_timesteps=1000) # Reduced for faster episodes
        
        self.episode_count +=1
        print(f"Episode {self.episode_count} finished after {step_count} steps.")
        avg_reward = sum(episode_rewards.values()) / max(1, len(episode_rewards))
        print(f"Average reward: {avg_reward}")
        return episode_rewards, step_count
    
    def _handle_new_probes(self):
        """Create agents for newly replicated probes"""
        for probe_id, probe_data in self.environment.probes.items():
            # Create agent if it's a new probe_id and doesn't have an agent yet.
            # The 'alive' flag is not the primary concern for agent creation anymore,
            # as probes are always "alive" in the environment dictionary.
            if probe_id not in self.probe_agents:
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
        rewards, steps = simulation.run_episode(render=True, train=simulation.training_mode) # max_steps now uses config by default
        
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