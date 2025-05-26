# main.py
import numpy as np
import random
import time
from typing import Dict
import pickle
import os
import psutil # For memory monitoring
import logging
import cProfile
import pstats
import argparse # For command line argument to enable profiling

from config import config, ConfigurationError # Use the global config instance
from environment import SolarSystemEnvironment
from probe import ProbeAgent
from visualization import Visualization

class BobiverseSimulation:
    """Manages the overall Bobiverse simulation lifecycle.

    This class orchestrates the simulation by initializing and coordinating core
    components: the `SolarSystemEnvironment` (managing physics, celestial bodies,
    resources), `ProbeAgent` instances (handling AI/RL logic for each probe),
    and the `Visualization` system (rendering and user input).

    It controls the main simulation loop, executing episodes step-by-step.
    Key responsibilities include:
    -   Initializing the environment and visualization based on `config`.
    -   Managing `ProbeAgent` creation, including handling new agents for
        replicated probes and potential model inheritance.
    -   Running simulation episodes: resetting the environment, collecting
        observations, dispatching actions from agents, stepping the environment,
        accumulating rewards, and optionally rendering.
    -   Integrating profiling (via `cProfile`) and memory monitoring (via `psutil`).
    -   Handling graceful shutdown and error logging.

    Attributes:
        environment (SolarSystemEnvironment): The simulation environment instance.
        visualization (Visualization): The visualization system instance.
        probe_agents (Dict[int, ProbeAgent]): A dictionary mapping probe IDs to
            their `ProbeAgent` instances.
        running (bool): Flag indicating if the main simulation loop should continue.
            Can be set to `False` by user input (e.g., closing window) or critical errors.
        training_mode (bool): If `True`, enables agent learning phases (e.g., at
            the end of an episode). Defaults to `True`.
        episode_count (int): Counter for the number of episodes run.
        process (psutil.Process): `psutil.Process` object for the current Python
            process, used for memory monitoring.
    """
    def __init__(self):
        """Initializes the `BobiverseSimulation`.

        This involves:
        1.  Creating instances of `SolarSystemEnvironment` and `Visualization`.
            These rely on the global `config` object for their parameters.
        2.  Initializing internal state:
            -   `probe_agents`: An empty dictionary to store `ProbeAgent` instances.
            -   `running`: Set to `True` to allow the simulation loop to start.
            -   `training_mode`: Set to `True` by default.
            -   `episode_count`: Set to 0.
            -   `process`: A `psutil.Process` object for the current process.

        Raises:
            ConfigurationError: If `SolarSystemEnvironment` or `Visualization`
                                (or the underlying `config` object) detect a critical
                                configuration issue that prevents initialization.
            Exception: For any other unexpected errors during the setup of these
                       core components. Such errors are logged as critical.
        """
        try:
            self.environment = SolarSystemEnvironment()
            self.visualization = Visualization()
        except ConfigurationError as e: # Propagated from config or component init
            logging.critical(f"Failed to initialize BobiverseSimulation due to ConfigurationError: {e}", exc_info=True)
            raise
        except Exception as e: # Other unexpected errors during component init
            logging.critical(f"An unexpected error occurred during BobiverseSimulation initialization: {e}", exc_info=True)
            raise
        
        self.probe_agents: Dict[int, ProbeAgent] = {}
        self.running = True
        self.training_mode = True # Default; can be changed before starting episodes
        self.episode_count = 0
        self.process = psutil.Process(os.getpid()) # For memory monitoring
        logging.info("BobiverseSimulation initialized successfully.")
        
    def initialize_agents(self):
        """Initializes `ProbeAgent` instances for all current probes in the environment.

        This method is typically called at the beginning of each simulation episode,
        after `self.environment.reset()` has populated `self.environment.probes`.
        It iterates through `self.environment.probes.keys()`:
        -   If a probe ID from the environment does not yet have a corresponding
            agent in `self.probe_agents`, a new `ProbeAgent` is created.
        -   The new agent is associated with its `probe_id` and given a reference
            to the `self.environment`.
        -   The new agent is stored in `self.probe_agents`.

        This ensures that every active probe at the start of an episode has an
        associated agent to control it. `_handle_new_probes` is used for probes
        created mid-episode (e.g., via replication).

        Error Handling:
            Logs an error if an exception occurs during the creation of any
            `ProbeAgent`, but attempts to continue initializing other agents.
        """
        try:
            current_probe_ids_in_env = list(self.environment.probes.keys())
            logging.debug(f"Initializing agents for probe IDs: {current_probe_ids_in_env}")
            
            # Clear any old agents that might not correspond to current probes
            # (though reset should handle environment's probes, this is a safeguard for self.probe_agents)
            self.probe_agents = {
                pid: agent for pid, agent in self.probe_agents.items() if pid in current_probe_ids_in_env
            }

            for probe_id in current_probe_ids_in_env:
                if probe_id not in self.probe_agents:
                    try:
                        self.probe_agents[probe_id] = ProbeAgent(probe_id, self.environment)
                        logging.info(f"Initialized ProbeAgent for probe_id: {probe_id}")
                    except Exception as e_agent:
                        logging.error(f"Failed to initialize ProbeAgent for probe_id {probe_id}: {e_agent}", exc_info=True)
        except Exception as e_outer:
            logging.error(f"Outer error during initialize_agents: {e_outer}", exc_info=True)

    def run_episode(self, max_steps: int = None, render: bool = True, train: bool = False):
        """Runs a single simulation episode.

        An episode consists of resetting the environment, then looping through
        simulation steps up to `max_steps` or until `self.running` becomes `False`.

        Core Loop (per step):
        1.  **Event Handling (if rendering)**: Processes Pygame events (e.g., quit).
        2.  **Observation Gathering**: Collects observations for all active probes.
        3.  **Action Selection**: Each `ProbeAgent` uses its observation to predict an action.
        4.  **Environment Step**: The environment executes the chosen actions, updates
            its state (celestial bodies, probes, resources), and returns new
            observations, rewards, 'done' flags, and info dictionaries.
        5.  **New Probe Handling**: Calls `_handle_new_probes()` to create agents for
            any newly replicated probes.
        6.  **Reward Accumulation**: Tracks total rewards for each probe.
        7.  **Rendering (if enabled)**: Updates the display via `self.visualization`.
        8.  **Monitoring**: Periodically checks memory usage.

        After the loop, if `train` is `True`, agents perform their learning updates.
        The episode concludes by logging summary statistics.

        Args:
            max_steps (int, optional): Maximum number of steps for this episode.
                Defaults to `config.RL.EPISODE_LENGTH_STEPS`.
            render (bool, optional): If `True`, the simulation is rendered graphically.
                Defaults to `True`.
            train (bool, optional): If `True`, agents' `learn()` methods are called
                at the end of the episode. Defaults to `False`.

        Returns:
            Tuple[Dict[int, float], int]:
                - `episode_rewards` (Dict[int, float]): Total reward per probe ID.
                - `step_count` (int): Number of steps executed in the episode.

        Raises:
            ConfigurationError: If a configuration issue prevents episode setup.
            Exception: For critical, unhandled errors during episode execution that
                       force termination. `self.running` will be set to `False`.
        """
        try:
            effective_max_steps = max_steps if max_steps is not None else config.RL.EPISODE_LENGTH_STEPS
            logging.info(f"Starting episode {self.episode_count + 1} with max_steps={effective_max_steps}, render={render}, train={train}")

            observations = self.environment.reset() # Resets env, returns initial obs for starting probes
            self.probe_agents.clear() # Clear agents from previous episode
            self.initialize_agents()  # Create agents for initial probes
            
            step_count = 0
            # Initialize episode_rewards for all agents active at the start of the episode
            episode_rewards: Dict[int, float] = {pid: 0.0 for pid in self.probe_agents.keys()}
            
            while step_count < effective_max_steps and self.running:
                try:
                    # 1. Event Handling (if rendering)
                    if render:
                        if not self.visualization.handle_events():
                            self.running = False
                            logging.info("Simulation stopped by user (visualization window closed).")
                            break # Exit step loop
                    
                    # 2. Observation Gathering & 3. Action Selection
                    actions: Dict[int, np.ndarray] = {}
                    current_observations_for_step: Dict[int, np.ndarray] = {}

                    active_agent_ids_this_step = list(self.probe_agents.keys())
                    for probe_id in active_agent_ids_this_step:
                        if probe_id in self.environment.probes and self.environment.probes[probe_id].get('alive', False):
                            obs_for_agent = observations.get(probe_id)
                            if obs_for_agent is None:
                                # This might happen if a probe was added mid-step by environment but not yet in 'observations'
                                # or if there's a logic flaw. Attempt to fetch directly as a fallback.
                                logging.warning(f"Observation for probe {probe_id} missing in 'observations' dict. Attempting direct fetch.")
                                obs_for_agent = self.environment.get_observation(probe_id)
                            
                            if obs_for_agent is not None:
                                current_observations_for_step[probe_id] = obs_for_agent
                                try:
                                    action = self.probe_agents[probe_id].predict(obs_for_agent)
                                    actions[probe_id] = action
                                except Exception as e_predict:
                                    logging.error(f"Error during agent {probe_id} prediction: {e_predict}", exc_info=True)
                                    actions[probe_id] = self.probe_agents[probe_id].get_default_action() # Fallback action
                            # else: Probe might exist but get_observation failed or returned None. Agent won't act.
                        # else: Probe associated with agent is no longer alive or in environment.

                    # 4. Environment Step
                    if not actions and active_agent_ids_this_step: # No actions could be generated, but agents exist
                        logging.warning(f"Step {step_count}: No valid actions generated for active agents. Stepping with empty actions dict.")
                    
                    next_observations, step_rewards, dones, infos = self.environment.step(actions)
                    observations = next_observations # Update for the next iteration's observation gathering
                    
                    # 5. New Probe Handling (agents for replicated probes)
                    self._handle_new_probes() # This might add to self.probe_agents
                    
                    # 6. Reward Accumulation
                    for probe_id, reward_value in step_rewards.items():
                        if probe_id in self.probe_agents: # Agent might have been created this step by _handle_new_probes
                            if probe_id not in episode_rewards: # Initialize if new agent
                                episode_rewards[probe_id] = 0.0
                            episode_rewards[probe_id] += reward_value
                    
                    # 7. Rendering (if enabled)
                    if render:
                        try:
                            self.visualization.render(self.environment, self.probe_agents)
                        except Exception as e_render:
                            logging.error(f"Error during visualization rendering: {e_render}", exc_info=True)
                            render = False # Disable rendering for the rest of the episode
                            logging.warning("Disabling rendering for the remainder of the episode due to an error.")
                        if render: # Check again in case it was just disabled
                            time.sleep(1.0 / config.Visualization.FPS if config.Visualization.FPS > 0 else 0.01)
                    
                    # 8. Monitoring
                    if step_count > 0 and step_count % config.Monitoring.MEMORY_CHECK_INTERVAL_STEPS == 0:
                        try:
                            memory_mb = self.process.memory_info().rss / (1024 * 1024)
                            if memory_mb > config.Monitoring.MEMORY_USAGE_WARN_MB:
                                logging.warning(f"High memory usage: {memory_mb:.2f} MB at step {step_count}")
                            elif config.Debug.DEBUG_MODE:
                                logging.debug(f"Memory usage: {memory_mb:.2f} MB at step {step_count}")
                        except psutil.Error as e_psutil:
                            logging.error(f"Could not retrieve memory usage: {e_psutil}", exc_info=True)
                            
                    step_count += 1
                    if all(dones.get(pid, False) for pid in self.probe_agents.keys() if self.probe_agents): # If all active agents are done
                        logging.info(f"All active agents are done at step {step_count}. Ending episode early.")
                        break


                except Exception as e_step_loop: # Catch-all for errors within a single step's logic
                    logging.error(f"Unhandled error in simulation step {step_count} loop: {e_step_loop}", exc_info=True)
                    self.running = False # Critical error, stop simulation
                    logging.critical("Critical error in simulation step loop. Terminating simulation run.")
                    break # Exit step loop

            # --- Post-episode ---
            if train and self.training_mode: # Check both explicit train flag and simulation's mode
                logging.info("Performing post-episode training for agents...")
                for agent_id, agent in self.probe_agents.items():
                    # Ensure agent's probe still exists and is alive, or if learning from buffer is always okay
                    if agent_id in self.environment.probes and self.environment.probes[agent_id].get('alive', True):
                        try:
                            # Assuming learn method handles its own data collection/buffering
                            agent.learn(total_timesteps=config.RL.BATCH_SIZE) # Example: learn from a batch
                        except Exception as e_learn:
                            logging.error(f"Error during agent {agent_id} learning phase: {e_learn}", exc_info=True)
            
            self.episode_count += 1
            logging.info(f"Episode {self.episode_count} finished after {step_count} steps.")
            if episode_rewards:
                 avg_reward_val = sum(episode_rewards.values()) / len(episode_rewards)
                 logging.info(f"Average total reward for episode {self.episode_count}: {avg_reward_val:.4f}")
                 if config.Debug.DEBUG_MODE: # Log individual rewards only in debug
                     for pid_log, rwd_log in episode_rewards.items():
                         logging.debug(f"Probe {pid_log} total reward for episode: {rwd_log:.4f}")
            else:
                logging.info(f"No rewards recorded for episode {self.episode_count} (no active agents or rewards processed).")

            return episode_rewards, step_count

        except ConfigurationError as e_config_episode:
            logging.critical(f"Episode {self.episode_count + 1} cannot run due to ConfigurationError: {e_config_episode}", exc_info=True)
            self.running = False # Stop further episodes
            return {}, 0
        except Exception as e_episode_setup:
            logging.critical(f"Critical error during setup or teardown of episode {self.episode_count + 1}: {e_episode_setup}", exc_info=True)
            self.running = False # Stop further episodes
            return {}, 0

    def _handle_new_probes(self):
        """Creates and initializes `ProbeAgent` instances for newly replicated probes.

        This method is called within the main simulation loop after each environment
        step. It checks `self.environment.probes` for any probe IDs that are not
        yet present in `self.probe_agents`, indicating a new probe (e.g., from replication).

        For each new probe:
        1.  A new `ProbeAgent` is instantiated.
        2.  **Model Inheritance (Simple Version)**: If other agents exist, it attempts
            to find a "parent" agent. A parent is an existing agent, ideally with a
            generation number less than the new probe's. If a parent is found, the
            new agent might inherit or copy the parent's model/policy. This is a
            basic mechanism for knowledge transfer. The current implementation picks
            the existing agent with the highest generation number that is still less
            than the new probe's generation as the parent.
        3.  The new agent is added to `self.probe_agents`.
        4.  An informational log message is recorded, noting the new agent's ID,
            its generation, and if a parent model was used.

        Assumptions:
            - `probe_data_from_env` (obtained from `self.environment.probes.items()`)
              is expected to have a `generation` attribute for new probes.
            - `ProbeAgent` constructor can accept a `parent_model` argument.

        Error Handling:
            If an exception occurs during the initialization of a specific new agent
            (e.g., issues with accessing `generation` attribute, parent model cloning,
            or the `ProbeAgent` constructor itself), an error is logged. The method
            attempts to continue processing other new probes.
        """
        try:
            current_env_probe_ids = set(self.environment.probes.keys())
            current_agent_ids = set(self.probe_agents.keys())
            new_probe_ids = current_env_probe_ids - current_agent_ids

            for probe_id in new_probe_ids:
                probe_data_from_env = self.environment.probes[probe_id] # Get data for the new probe
                parent_agent_for_model = None
                
                new_probe_generation = probe_data_from_env.get('generation', float('inf')) # Safely get generation

                if self.probe_agents: # Check if any agents exist to be parents
                    potential_parents = [
                        ag for ag_id, ag in self.probe_agents.items()
                        if ag.generation < new_probe_generation # Parent must be older generation
                    ]
                    if potential_parents:
                         # Prefer parent with highest generation number (closest ancestor still older)
                         parent_agent_for_model = max(potential_parents, key=lambda ag: ag.generation, default=None)

                try:
                    # Create the new agent, passing the parent_model if found
                    self.probe_agents[probe_id] = ProbeAgent(
                        probe_id,
                        self.environment,
                        parent_model=parent_agent_for_model.model if parent_agent_for_model and hasattr(parent_agent_for_model, 'model') else None
                    )
                    parent_info = f"from parent agent {parent_agent_for_model.probe_id} (gen {parent_agent_for_model.generation})" if parent_agent_for_model else "as new lineage (no suitable parent model found)"
                    logging.info(f"Initialized new agent for replicated probe {probe_id} (gen {new_probe_generation}) {parent_info}.")
                except Exception as e_agent_create:
                    logging.error(f"Failed to create ProbeAgent for new probe_id {probe_id}: {e_agent_create}", exc_info=True)

        except Exception as e_outer_handle: # Catch errors in the iteration logic itself
            logging.error(f"Error in _handle_new_probes outer loop: {e_outer_handle}", exc_info=True)

if __name__ == "__main__":
    """Main entry point for executing the Bobiverse Orbital Simulation.

    This script orchestrates the entire simulation lifecycle:
    1.  **Argument Parsing**: Handles command-line arguments.
        -   `--profile`: Enables `cProfile` for performance analysis, saving results
            to `simulation_profile.prof`.
    2.  **Profiling Setup**: If `--profile` is used, `cProfile.Profile` is initialized
        and enabled before the main simulation starts.
    3.  **Simulation Initialization**:
        -   An instance of `BobiverseSimulation` is created. This involves loading
          all configurations (from `config.py`), setting up the
          `SolarSystemEnvironment` (celestial bodies, physics), the `Visualization`
          (Pygame window), and preparing agent management systems.
        -   Critical `ConfigurationError` or other exceptions during this phase
          are caught, logged, and will terminate the script.
    4.  **Episode Loop**:
        -   The simulation runs for a configurable number of episodes (default: 5).
        -   For each episode, `simulation.run_episode()` is called. This method
          handles the detailed step-by-step execution of the simulation, including
          agent actions, environment updates, rendering, and optional training.
        -   The loop continues as long as `simulation.running` is `True` and the
          episode limit hasn't been reached. `simulation.running` can be set to
          `False` by user actions (e.g., closing the visualization window) or
          by critical errors within an episode.
    5.  **Error Handling**: Robust `try...except` blocks are used to catch:
        -   `ConfigurationError`: If the simulation cannot start due to invalid settings.
        -   General `Exception`: For any other unexpected critical errors during the
          main execution flow. These are logged, and a user-friendly message is printed.
    6.  **Profiling Teardown**: If profiling was enabled, the profiler is disabled
        after the simulation completes (or terminates due to error), and the
        collected statistics are dumped to `simulation_profile.prof`.
        Optional summary printing to logs is also available.
    7.  **Graceful Termination**: Logs are written at various stages, and a final
        termination message is logged.

    Execution Flow:
    `main.py` -> `BobiverseSimulation.__init__` -> `BobiverseSimulation.run_episode` (looped)
               -> `SolarSystemEnvironment` & `ProbeAgent` & `Visualization` interactions.
    """
    parser = argparse.ArgumentParser(description="Run the Bobiverse Orbital Simulation.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling for the simulation. Statistics will be saved to 'simulation_profile.prof'."
    )
    args = parser.parse_args()

    profiler = None
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        logging.info("cProfile profiling enabled. Output will be saved to simulation_profile.prof upon completion.")

    simulation_instance = None # To ensure it's in scope for finally block if init fails
    try:
        logging.info("Initializing BobiverseSimulation...")
        simulation_instance = BobiverseSimulation()
        
        num_episodes_to_run = 5 # Example: Run for 5 episodes
        logging.info(f"Starting Bobiverse Simulation for {num_episodes_to_run} episodes.")

        for episode_num in range(num_episodes_to_run):
            if not simulation_instance.running:
                logging.warning(f"Simulation.running is False before starting episode {episode_num + 1}. Terminating.")
                break
            
            logging.info(f"\n{'='*15} Starting Episode {episode_num + 1}/{num_episodes_to_run} {'='*15}")
            
            # Run the episode. Rendering is typically on, training can be controlled by simulation_instance.training_mode
            episode_rewards, steps_taken = simulation_instance.run_episode(
                render=True, # Control via config or another arg if needed
                train=simulation_instance.training_mode
            )
            
            if not simulation_instance.running:
                logging.info(f"Simulation run was terminated (e.g., by user or critical error) during episode {episode_num + 1}.")
                break # Exit episode loop
                
        logging.info(f"\n{'='*15} Simulation finished all {num_episodes_to_run} requested episodes or was terminated. {'='*15}")
        
        # Optional: Save models or perform other cleanup after all episodes
        if simulation_instance and simulation_instance.training_mode:
            save_dir = "saved_models_final"
            os.makedirs(save_dir, exist_ok=True)
            logging.info(f"Attempting to save final models to directory: {save_dir}")
            for probe_id, agent in simulation_instance.probe_agents.items():
                if hasattr(agent, 'save') and callable(agent.save):
                    try:
                        # Ensure agent and its model are in a savable state
                        agent_generation = agent.generation if hasattr(agent, 'generation') else 'unknown'
                        save_path = os.path.join(save_dir, f"probe_{probe_id}_gen{agent_generation}_final.zip")
                        agent.save(save_path)
                        logging.info(f"Saved model for probe {probe_id} to {save_path}")
                    except Exception as e_save:
                        logging.error(f"Failed to save final model for probe {probe_id}: {e_save}", exc_info=True)
            logging.info(f"Final model saving process complete for directory: {save_dir}")

    except ConfigurationError as e_config_main:
        logging.critical(f"BobiverseSimulation could not be initialized or run due to a ConfigurationError: {e_config_main}", exc_info=True)
        print(f"FATAL CONFIGURATION ERROR: {e_config_main}. Simulation cannot start. Check logs for details.")
    except Exception as e_main:
        logging.critical(f"An unexpected critical error occurred in the main simulation execution block: {e_main}", exc_info=True)
        print(f"FATAL UNEXPECTED ERROR: {e_main}. Simulation terminated. Check logs for details.")
    finally:
        if profiler:
            profiler.disable()
            stats_file = "simulation_profile.prof"
            try:
                profiler.dump_stats(stats_file)
                logging.info(f"Profiling data successfully saved to {stats_file}")
                
                # Optional: Print a brief summary of profiling stats to the log
                # import io
                # s = io.StringIO()
                # ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative') # Sort by cumulative time
                # ps.print_stats(20) # Print top 20 functions
                # logging.info(f"\n--- Top 20 Profiled Functions (Cumulative Time) ---\n{s.getvalue()}")
            except Exception as e_profile_dump:
                logging.error(f"Failed to save or process profiling data from {stats_file}: {e_profile_dump}", exc_info=True)
        
        logging.info("Bobiverse Simulation terminated.")