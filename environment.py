# environment.py
import gym
from gym import spaces
import numpy as np
import random
import math # Added for math.radians
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from config import config # Import the global config instance
from solarsystem import CelestialBody, OrbitalMechanics
from physics_utils import PhysicsError, safe_divide # Potentially useful for handling errors from orbital_mechanics

# Keep existing Message, Resource, SpaceEnvironment classes if they are still used as a base
# For brevity, assuming they are the same as before unless specified.
# If SolarSystemEnvironment is now the primary environment, SpaceEnvironment might be simplified or removed later.

@dataclass
class Message: # Copied from previous version if still needed
    sender_id: int
    msg_type: str
    position: Tuple[float, float]
    resource_amount: float
    timestamp: int

@dataclass
class Resource: # Copied from previous version if still needed
    position: Tuple[float, float]
    amount: float
    max_amount: float
    
    def harvest(self, rate: float) -> float:
        harvested = min(rate, self.amount)
        self.amount -= harvested
        return harvested
    
    def regenerate(self):
        self.amount = min(self.max_amount, self.amount + config.Resource.REGEN_RATE_PER_STEP) # Use config

class SpaceEnvironment(gym.Env): # Copied from previous version, might need pruning if not fully used
    def __init__(self):
        super().__init__()
        # Using config object for world dimensions
        self.world_width = config.World.WIDTH_SIM
        self.world_height = config.World.HEIGHT_SIM
        self.resources: List[Resource] = []
        self.probes: Dict[int, Dict] = {}
        self.messages: List[Message] = []
        self.step_count = 0
        self.max_probe_id = 0
        self.total_resources_mined = 0.0
        
        self._generate_resources()
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(config.RL.OBSERVATION_SPACE_SIZE,), # Use config
            dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(config.RL.ACTION_SPACE_DIMS) # Use config
        
    def _generate_resources(self):
        self.resources = []
        for _ in range(config.Resource.COUNT): # Use config
            pos = (random.uniform(0, self.world_width),
                   random.uniform(0, self.world_height))
            amount = random.uniform(config.Resource.MIN_AMOUNT, config.Resource.MAX_AMOUNT) # Use config
            self.resources.append(Resource(pos, amount, amount)) # Assuming Resource class takes max_amount
            
    def add_probe(self, probe_id: int, position: Tuple[float, float],
                  energy: float = config.Probe.INITIAL_ENERGY, generation: int = 0): # Use config
        # Simplified probe for now, can be expanded later
        self.probes[probe_id] = {
            'position': np.array(position, dtype=np.float32),
            'velocity': np.array([0.0, 0.0], dtype=np.float32), # Assuming probes are not part of n-body orbital mechanics for now
            'energy': energy,
            'alive': True,
            'trail': [], # Initialize an empty trail
            'max_trail_points': config.Visualization.MAX_PROBE_TRAIL_POINTS
        }
        self.max_probe_id = max(self.max_probe_id, probe_id)

    def get_observation(self, probe_id: int) -> np.ndarray:
        return np.zeros(config.RL.OBSERVATION_SPACE_SIZE, dtype=np.float32) # Use config

    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        for probe_id in list(self.probes.keys()):
            probe_data = self.probes.get(probe_id)
            if probe_data and probe_data['alive']:
                # --- Probe Physics Update (Example: simple movement based on action) ---
                # This section needs to be properly implemented based on how probe actions translate to movement.
                # For now, let's assume 'actions' might contain a target velocity or thrust.
                # Placeholder: If actions[probe_id] provides dx, dy
                # action_taken = actions.get(probe_id) # Get action for this specific probe
                # if action_taken is not None and len(action_taken) >= 2: # Example: action = [thrust_x, thrust_y, ...]
                #     # Apply thrust or set velocity based on action
                #     # For simplicity, let's assume action directly gives a small delta_pos for now
                #     # This is a placeholder for actual probe physics based on its RL actions
                #     delta_pos = np.array([action_taken[0]*0.1, action_taken[1]*0.1], dtype=np.float32)
                #     probe_data['position'] += delta_pos
                #     # Clamp position to world bounds (optional, depends on desired behavior)
                #     probe_data['position'][0] = np.clip(probe_data['position'][0], 0, self.world_width)
                #     probe_data['position'][1] = np.clip(probe_data['position'][1], 0, self.world_height)

                # --- Update Probe Trail ---
                probe_data['trail'].append(probe_data['position'].copy())
                if len(probe_data['trail']) > probe_data['max_trail_points']:
                    probe_data['trail'].pop(0)
                
                # --- Energy Decay & Death ---
                # probe_data['energy'] -= config.Probe.ENERGY_DECAY_RATE_PER_STEP # Example
                # if probe_data['energy'] <= 0:
                #    probe_data['alive'] = False
                #    dones[probe_id] = True # Probe is done if it runs out of energy

                observations[probe_id] = self.get_observation(probe_id)
                rewards[probe_id] = 0.0 # Placeholder reward
                # Probe-specific done condition might also come from energy or other factors
                dones[probe_id] = dones.get(probe_id, False) or self.step_count >= config.RL.EPISODE_LENGTH_STEPS
                infos[probe_id] = {}
            elif probe_data and not probe_data['alive']: # If probe died this step or was already dead
                dones[probe_id] = True # Ensure done is true for dead probes
                # Optionally remove dead probes from self.probes here or in a separate cleanup phase
                # For now, just mark as done.
                if probe_id not in observations: # If it died this step, might not have obs yet
                    observations[probe_id] = self.get_observation(probe_id) # Provide a final obs
                    rewards[probe_id] = rewards.get(probe_id, 0.0) # Ensure reward entry
                    infos[probe_id] = infos.get(probe_id, {})
        
        # Cleanup: Remove probes that are no longer alive
        probes_to_remove = [pid for pid, pdata in self.probes.items() if not pdata['alive'] and dones.get(pid, False)]
        for pid_to_remove in probes_to_remove:
            if config.Debug.DEBUG_MODE: # Or a new specific debug flag for probe lifecycle
                print(f"Step {self.step_count}: Removing dead probe {pid_to_remove}")
            del self.probes[pid_to_remove]

        self.step_count += 1
        return observations, rewards, dones, infos

    def reset(self):
        self.step_count = 0
        self.probes = {}
        self.messages = []
        self._generate_resources()
        # Add initial probes based on config
        for i in range(config.Probe.INITIAL_PROBES):
             # Distribute initial probes or place at center. For now, center.
            self.add_probe(i, (self.world_width / 2, self.world_height / 2))
        
        obs = {}
        for i in range(config.Probe.INITIAL_PROBES):
            if i in self.probes:
                 obs[i] = self.get_observation(i)
        return obs

# Your new SolarSystemEnvironment class:
class SolarSystemEnvironment(SpaceEnvironment): # Inherits from the (potentially simplified) SpaceEnvironment
    def __init__(self):
        super().__init__() # Calls SpaceEnvironment.__init__

        self.orbital_mechanics = OrbitalMechanics()
        self.sun: Optional[CelestialBody] = None
        self.planets: List[CelestialBody] = [] # Includes Earth's Moon and other moons
        self.initial_system_energy: Optional[float] = None # For energy conservation check

        self._create_celestial_bodies()
        # Calculate and store initial system energy
        if config.Debug.MONITOR_ENERGY_CONSERVATION:
            all_bodies_for_energy = []
            if self.sun: all_bodies_for_energy.append(self.sun)
            all_bodies_for_energy.extend(self.planets)
            if all_bodies_for_energy:
                self.initial_system_energy = self.orbital_mechanics.calculate_total_system_energy(all_bodies_for_energy)
                if config.Debug.ORBITAL_MECHANICS: # Or a more specific energy debug flag
                    print(f"DEBUG: Initial system energy: {self.initial_system_energy:.6e} Joules (kg*(km/s)^2)")


    def _create_celestial_bodies(self):
        self.planets = [] # Clear previous planets
        sun_config_data = config.SolarSystem.PLANET_DATA['Sun']
        self.sun = CelestialBody(
            name="Sun",
            mass_kg=sun_config_data['mass_kg'], # Already uses config.SolarSystem.SUN_MASS_KG via __init__
            radius_km=sun_config_data['radius_km'],
            display_radius_sim=sun_config_data['display_radius_sim'],
            color=sun_config_data['color'],
            a_au=sun_config_data['semi_major_axis_au'],
            e=sun_config_data['eccentricity'],
            i_deg=sun_config_data['inclination_deg'],
            omega_deg=sun_config_data['longitude_of_ascending_node_deg'],
            w_deg=sun_config_data['argument_of_perihelion_deg'],
            m0_deg=sun_config_data['mean_anomaly_at_epoch_deg'],
            position_sim=np.array(config.World.CENTER_SIM, dtype=np.float64), # Use World.CENTER_SIM
            velocity_sim_s=np.array([0.0, 0.0], dtype=np.float64), # Sun is initially static
            orbits_around=None
        )
        if config.Debug.ORBITAL_MECHANICS:
            print(f"DEBUG: Created Sun: Name={self.sun.name}, Pos_sim={self.sun.position_sim}, Vel_sim_s={self.sun.velocity_sim_s}")

        all_created_bodies: List[CelestialBody] = [self.sun]
        celestial_body_references: Dict[str, CelestialBody] = {self.sun.name: self.sun}

        # Create planets (orbiting Sun)
        planet_names = [
            "Mercury", "Venus", "Earth", "Mars",
            "Jupiter", "Saturn", "Uranus", "Neptune"
        ]
        for name in planet_names:
            data = config.SolarSystem.PLANET_DATA[name]
            planet_body = CelestialBody(
                name=name, mass_kg=data['mass_kg'], radius_km=data['radius_km'],
                display_radius_sim=data['display_radius_sim'], color=data['color'],
                a_au=data['semi_major_axis_au'], e=data['eccentricity'], i_deg=data['inclination_deg'],
                omega_deg=data['longitude_of_ascending_node_deg'], w_deg=data['argument_of_perihelion_deg'],
                m0_deg=data['mean_anomaly_at_epoch_deg'], orbits_around="Sun"
            )
            pos_rel_sun_sim, vel_rel_sun_sim_s = self.orbital_mechanics.calculate_initial_state_vector(
                planet_body, self.sun.mass_kg
            )
            planet_body.position_sim = self.sun.position_sim + pos_rel_sun_sim
            planet_body.velocity_sim_s = self.sun.velocity_sim_s + vel_rel_sun_sim_s # Add Sun's velocity if Sun moves

            self.planets.append(planet_body)
            all_created_bodies.append(planet_body)
            celestial_body_references[name] = planet_body
            
            if config.Debug.ORBITAL_MECHANICS:
                pos_au = planet_body.position_sim / config.AU_SCALE
                vel_au_s = planet_body.velocity_sim_s / config.AU_SCALE
                print(f"DEBUG: Created {name}: Pos_AU=[{pos_au[0]:.3f}, {pos_au[1]:.3f}], Vel_AU/s=[{vel_au_s[0]:.3f}, {vel_au_s[1]:.3f}]")

        # Create Moons (orbiting other planets)
        # Iterate through PLANET_DATA to find all bodies that orbit something other than None or "Sun"
        for body_name, body_data in config.SolarSystem.PLANET_DATA.items():
            primary_name = body_data.get('central_body')
            if primary_name and primary_name != "Sun" and primary_name in celestial_body_references:
                primary_body = celestial_body_references[primary_name]
                
                moon_body = CelestialBody(
                    name=body_name, mass_kg=body_data['mass_kg'], radius_km=body_data['radius_km'],
                    display_radius_sim=body_data['display_radius_sim'], color=body_data['color'],
                    a_au=body_data['semi_major_axis_au'], e=body_data['eccentricity'], i_deg=body_data['inclination_deg'],
                    omega_deg=body_data['longitude_of_ascending_node_deg'], w_deg=body_data['argument_of_perihelion_deg'],
                    m0_deg=body_data['mean_anomaly_at_epoch_deg'], orbits_around=primary_name
                )
                pos_rel_primary_sim, vel_rel_primary_sim_s = self.orbital_mechanics.calculate_initial_state_vector(
                    moon_body, primary_body.mass_kg
                )
                moon_body.position_sim = primary_body.position_sim + pos_rel_primary_sim
                moon_body.velocity_sim_s = primary_body.velocity_sim_s + vel_rel_primary_sim_s
                
                self.planets.append(moon_body) # Add to general list of planets/moons
                all_created_bodies.append(moon_body)
                celestial_body_references[body_name] = moon_body
                
                if config.Debug.ORBITAL_MECHANICS:
                    pos_au_m = moon_body.position_sim / config.AU_SCALE
                    vel_au_s_m = moon_body.velocity_sim_s / config.AU_SCALE
                    print(f"DEBUG: Created {body_name} (orbits {primary_name}): Pos_AU=[{pos_au_m[0]:.4f}, {pos_au_m[1]:.4f}], Vel_AU/s=[{vel_au_s_m[0]:.4f}, {vel_au_s_m[1]:.4f}]")
                    rel_pos_au_m = (moon_body.position_sim - primary_body.position_sim) / config.AU_SCALE
                    print(f"    Rel to {primary_name}: Pos_AU=[{rel_pos_au_m[0]:.4f}, {rel_pos_au_m[1]:.4f}], Dist={np.linalg.norm(rel_pos_au_m):.4f} AU (target ~{body_data['semi_major_axis_au']:.4f} AU)")

        # Initial acceleration calculation for Verlet integration
        # This must be done after all bodies are created and have initial positions/velocities
        for body_to_update in all_created_bodies:
            influencing_bodies = [other for other in all_created_bodies if other is not body_to_update]
            initial_accel_km_s2 = self.orbital_mechanics.calculate_gravitational_acceleration(
                body_to_update, influencing_bodies
            )
            body_to_update.previous_acceleration_sim_s2 = initial_accel_km_s2 * config.KM_SCALE
            if config.Debug.ORBITAL_MECHANICS and body_to_update.name in ["Earth", "Moon", "Sun"]:
                 print(f"DEBUG: Initial Accel for {body_to_update.name}: {body_to_update.previous_acceleration_sim_s2}")


    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        # Consolidate all bodies that participate in n-body simulation
        all_sim_bodies: List[CelestialBody] = []
        if self.sun:
            all_sim_bodies.append(self.sun)
        all_sim_bodies.extend(self.planets) # self.planets now includes planets and moons directly

        if all_sim_bodies: # Check if there are any bodies to update
            self.orbital_mechanics.propagate_orbits_verlet( # Call the new Verlet method
                all_sim_bodies,
                config.Time.SIM_SECONDS_PER_STEP # Use config for timestep
            )
        
        if config.Debug.ORBITAL_MECHANICS and self.step_count % 100 == 0 and all_sim_bodies:
            # Debug output for positions and velocities (can be extensive)
            temp_body_map_for_debug = {b.name: b for b in all_sim_bodies}

            for body in all_sim_bodies:
                pos_au_dbg = body.position_sim / config.AU_SCALE # Use global config.AU_SCALE
                vel_au_s_dbg = body.velocity_sim_s / config.AU_SCALE # Use global config.AU_SCALE
                
                log_msg = (f"Step {self.step_count}: {body.name} - "
                           f"Pos_AU: [{pos_au_dbg[0]:.3f}, {pos_au_dbg[1]:.3f}], "
                           f"Vel_AU/s: [{vel_au_s_dbg[0]:.4f}, {vel_au_s_dbg[1]:.4f}]")

                if body.orbits_around and body.orbits_around in temp_body_map_for_debug:
                    central_body_for_debug = temp_body_map_for_debug[body.orbits_around]
                    distance_vector_sim = body.position_sim - central_body_for_debug.position_sim
                    distance_au = np.linalg.norm(distance_vector_sim) / config.AU_SCALE # Use global config.AU_SCALE
                    log_msg += f", Dist_to_{central_body_for_debug.name}: {distance_au:.4f} AU (a={body.a_au:.4f})"
                print(log_msg)

        # Energy Conservation Check
        if config.Debug.MONITOR_ENERGY_CONSERVATION and \
           self.step_count % config.Monitoring.ENERGY_CHECK_INTERVAL_STEPS == 0 and \
           self.initial_system_energy is not None and all_sim_bodies: # Ensure all_sim_bodies is not empty
            current_energy = self.orbital_mechanics.calculate_total_system_energy(all_sim_bodies)
            energy_delta = current_energy - self.initial_system_energy
            # Use safe_divide for percentage calculation
            energy_delta_percent = safe_divide(energy_delta * 100.0, self.initial_system_energy, default_on_zero_denom=0.0)
            print(f"ENERGY CHECK (Step {self.step_count}): Current: {current_energy:.6e}, Initial: {self.initial_system_energy:.6e}, Delta: {energy_delta:.6e} ({energy_delta_percent:.6f}%)")
        
        observations, rewards, dones, infos = super().step(actions) # Handles probe logic
        return observations, rewards, dones, infos

    def reset(self) -> Dict[int, np.ndarray]:
        self._create_celestial_bodies()
        initial_observations = super().reset() # Call super().reset() after _create_celestial_bodies
        return initial_observations

    def get_celestial_bodies_data_for_render(self) -> List[Dict]:
        render_data = []
        if self.sun:
            render_data.append({
                'name': self.sun.name,
                'position': self.sun.position_sim.tolist(), 
                'radius_sim': self.sun.display_radius_sim, 
                'color': self.sun.color,
                'orbit_path': [p.tolist() for p in self.sun.orbit_path] 
            })
        for planet_body in self.planets: # Includes planets and moons
            render_data.append({
                'name': planet_body.name,
                'position': planet_body.position_sim.tolist(),
                'radius_sim': planet_body.display_radius_sim,
                'color': planet_body.color,
                'orbit_path': [p.tolist() for p in planet_body.orbit_path]
            })
        
        # Add probe data for rendering, including trails
        for probe_id, probe_data in self.probes.items():
            if probe_data['alive']:
                render_data.append({
                    'name': f"Probe_{probe_id}",
                    'type': 'probe', # Differentiate from celestial bodies
                    'position': probe_data['position'].tolist(),
                    'radius_sim': config.Visualization.PROBE_SIZE_PX / 2, # Example radius for rendering
                    'color': (255, 0, 255), # Example probe color
                    'trail': [p.tolist() for p in probe_data['trail']]
                })
        return render_data