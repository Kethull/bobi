# environment.py
import gym
from gym import spaces
import numpy as np
import random
import math # Added for math.radians
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from config import ( # Explicit imports for clarity and new constants
    PLANET_DATA, SIM_SECONDS_PER_STEP, AU_SCALE, KM_SCALE,
    SUN_POSITION_SIM, WORLD_SIZE_SIM, SUN_MASS_KG, DEBUG_ORBITAL_MECHANICS,
    OBSERVATION_SPACE_SIZE, ACTION_SPACE_DIMS, EPISODE_LENGTH, # For SpaceEnvironment
    RESOURCE_COUNT, RESOURCE_MIN_AMOUNT, RESOURCE_MAX_AMOUNT, RESOURCE_REGEN_RATE, # For SpaceEnvironment
    INITIAL_ENERGY # For SpaceEnvironment
)
from solarsystem import CelestialBody, OrbitalMechanics # AsteroidBelt is commented out in solarsystem.py

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
        self.amount = min(self.max_amount, self.amount + RESOURCE_REGEN_RATE)

class SpaceEnvironment(gym.Env): # Copied from previous version, might need pruning if not fully used
    def __init__(self):
        super().__init__()
        # Using WORLD_SIZE_SIM from new config for consistency
        self.world_width = WORLD_SIZE_SIM
        self.world_height = WORLD_SIZE_SIM
        self.resources: List[Resource] = [] # Ensure type hint
        self.probes: Dict[int, Dict] = {}    # Ensure type hint
        self.messages: List[Message] = []   # Ensure type hint
        self.step_count = 0
        self.max_probe_id = 0
        self.total_resources_mined = 0.0
        
        self._generate_resources()
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(OBSERVATION_SPACE_SIZE,),
            dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(ACTION_SPACE_DIMS)
        
    def _generate_resources(self): # Simplified, assuming resources are less critical for solar system focus
        self.resources = []
        # For now, fewer resources if any, as focus is on planets
        for _ in range(RESOURCE_COUNT if 'RESOURCE_COUNT' in globals() else 1): # Use RESOURCE_COUNT from config
            pos = (random.uniform(0, self.world_width),
                   random.uniform(0, self.world_height))
            amount = random.uniform(RESOURCE_MIN_AMOUNT if 'RESOURCE_MIN_AMOUNT' in globals() else 100,
                                    RESOURCE_MAX_AMOUNT if 'RESOURCE_MAX_AMOUNT' in globals() else 200)
            self.resources.append(Resource(pos, amount, amount))
            
    def add_probe(self, probe_id: int, position: Tuple[float, float], 
                  energy: float = INITIAL_ENERGY, generation: int = 0):
        # Simplified probe for now, can be expanded later
        self.probes[probe_id] = {
            'position': np.array(position, dtype=np.float32),
            'velocity': np.array([0.0, 0.0], dtype=np.float32), # Probes are not part of orbital mechanics here
            'energy': energy,
            'alive': True,
            # Add other necessary fields if SpaceEnvironment.step uses them
        }
        self.max_probe_id = max(self.max_probe_id, probe_id)

    def get_observation(self, probe_id: int) -> np.ndarray:
        # Dummy observation if probes are not the focus
        return np.zeros(OBSERVATION_SPACE_SIZE, dtype=np.float32)

    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        # Basic step for probes if any are active
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        # Minimal probe logic for now
        for probe_id in list(self.probes.keys()): # Iterate over a copy if probes can be removed
            if self.probes[probe_id]['alive']:
                # Process actions if any relevant probe actions exist
                # Update probe physics (e.g., simple movement, energy decay)
                # self.probes[probe_id]['energy'] -= 0.1 # Example decay
                # if self.probes[probe_id]['energy'] <= 0:
                #     self.probes[probe_id]['alive'] = False
                
                observations[probe_id] = self.get_observation(probe_id)
                rewards[probe_id] = 0.0 # Placeholder
                dones[probe_id] = self.step_count >= EPISODE_LENGTH # Placeholder
                infos[probe_id] = {}
            else: # Remove dead probes
                # del self.probes[probe_id] # Be careful if iterating and modifying
                pass

        self.step_count += 1
        return observations, rewards, dones, infos

    def reset(self):
        self.step_count = 0
        self.probes = {}
        self.messages = []
        self._generate_resources()
        # Add initial probes if needed by the simulation logic
        self.add_probe(0, (self.world_width / 2, self.world_height / 2)) # Assuming INITIAL_PROBES is 1 for now
        obs = {}
        if 0 in self.probes:
           obs[0] = self.get_observation(0)
        return obs

# Your new SolarSystemEnvironment class:
class SolarSystemEnvironment(SpaceEnvironment): # Inherits from the (potentially simplified) SpaceEnvironment
    def __init__(self):
        super().__init__() # Calls SpaceEnvironment.__init__

        self.orbital_mechanics = OrbitalMechanics()
        self.sun: Optional[CelestialBody] = None
        self.planets: List[CelestialBody] = [] # Includes Earth's Moon and other moons
        # self.asteroid_belt: Optional[AsteroidBelt] = None # Commented out for now

        self._create_celestial_bodies()
        # if self.orbital_mechanics: # AsteroidBelt commented out
        #     self.asteroid_belt = AsteroidBelt(self.orbital_mechanics)

    def _create_celestial_bodies(self):
        self.planets = [] # Clear previous planets
        sun_config_data = PLANET_DATA['Sun']
        self.sun = CelestialBody(
            name="Sun",
            mass_kg=sun_config_data['mass_kg'],
            radius_km=sun_config_data['radius_km'],
            display_radius_sim=sun_config_data['display_radius_sim'],
            color=sun_config_data['color'],
            a_au=sun_config_data['semi_major_axis_au'],
            e=sun_config_data['eccentricity'],
            i_deg=sun_config_data['inclination_deg'],
            omega_deg=sun_config_data['longitude_of_ascending_node_deg'],
            w_deg=sun_config_data['argument_of_perihelion_deg'],
            m0_deg=sun_config_data['mean_anomaly_at_epoch_deg'],
            position_sim=np.array(SUN_POSITION_SIM, dtype=np.float64),
            velocity_sim_s=np.array([0.0, 0.0], dtype=np.float64),
            orbits_around=None # Sun doesn't orbit anything in this model
        )
        if DEBUG_ORBITAL_MECHANICS:
            print(f"DEBUG: Created Sun: Name={self.sun.name}, Pos_sim={self.sun.position_sim}, Vel_sim_s={self.sun.velocity_sim_s}")

        # Create planets (Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune)
        planet_names = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
        # Store references to planets that have moons - defined inside the method
        celestial_body_references: Dict[str, CelestialBody] = {} 
        celestial_body_references[self.sun.name] = self.sun # Add Sun for reference

        for name in planet_names:
            data = PLANET_DATA[name]
            planet_body = CelestialBody(
                name=name,
                mass_kg=data['mass_kg'],
                radius_km=data['radius_km'],
                display_radius_sim=data['display_radius_sim'],
                color=data['color'],
                a_au=data['semi_major_axis_au'],
                e=data['eccentricity'],
                i_deg=data['inclination_deg'],
                omega_deg=data['longitude_of_ascending_node_deg'],
                w_deg=data['argument_of_perihelion_deg'],
                m0_deg=data['mean_anomaly_at_epoch_deg'],
                orbits_around="Sun" # Explicitly orbits Sun
            )
            
            pos_rel_sun_sim, vel_rel_sun_sim_s = self.orbital_mechanics.calculate_initial_state_vector(
                planet_body, self.sun.mass_kg
            )
            
            planet_body.position_sim = np.array(SUN_POSITION_SIM, dtype=np.float64) + pos_rel_sun_sim
            planet_body.velocity_sim_s = vel_rel_sun_sim_s # Sun's velocity is [0,0]

            self.planets.append(planet_body)
            celestial_body_references[name] = planet_body # Store reference for moons
            
            if DEBUG_ORBITAL_MECHANICS:
                pos_au = planet_body.position_sim / AU_SCALE
                vel_au_s = planet_body.velocity_sim_s / AU_SCALE
                print(f"DEBUG: Created {name}: Pos_sim=[{pos_au[0]:.3f}, {pos_au[1]:.3f}] AU, Vel_sim_s=[{vel_au_s[0]:.3f}, {vel_au_s[1]:.3f}] AU/s")

        # Create Moons
        moon_definitions = {
            "Moon": "Earth",
            "Io": "Jupiter", "Europa": "Jupiter", "Ganymede": "Jupiter", "Callisto": "Jupiter",
            "Titan": "Saturn"
            # Add other moons here if defined in PLANET_DATA
        }

        for moon_name, primary_name in moon_definitions.items():
            if primary_name in celestial_body_references and moon_name in PLANET_DATA:
                primary_body = celestial_body_references[primary_name]
                moon_data = PLANET_DATA[moon_name]
                
                moon_body = CelestialBody(
                    name=moon_name,
                    mass_kg=moon_data['mass_kg'],
                    radius_km=moon_data['radius_km'],
                    display_radius_sim=moon_data['display_radius_sim'],
                    color=moon_data['color'],
                    a_au=moon_data['semi_major_axis_au'], # Relative to its primary
                    e=moon_data['eccentricity'],
                    i_deg=moon_data['inclination_deg'],
                    omega_deg=moon_data['longitude_of_ascending_node_deg'],
                    w_deg=moon_data['argument_of_perihelion_deg'],
                    m0_deg=moon_data['mean_anomaly_at_epoch_deg'],
                    orbits_around=primary_name
                )

                pos_rel_primary_sim, vel_rel_primary_sim_s = self.orbital_mechanics.calculate_initial_state_vector(
                    moon_body, primary_body.mass_kg # Moon orbits its primary
                )

                # Convert Moon's relative state vector to absolute (world) coordinates
                moon_body.position_sim = primary_body.position_sim + pos_rel_primary_sim
                moon_body.velocity_sim_s = primary_body.velocity_sim_s + vel_rel_primary_sim_s
                
                self.planets.append(moon_body) # Add Moon to the list of bodies to be updated and rendered
                celestial_body_references[moon_name] = moon_body # Store moon reference
                
                if DEBUG_ORBITAL_MECHANICS:
                    pos_au_moon = moon_body.position_sim / AU_SCALE
                    vel_au_s_moon = moon_body.velocity_sim_s / AU_SCALE
                    print(f"DEBUG: Created {moon_name} (orbits {primary_name}): Pos_sim=[{pos_au_moon[0]:.4f}, {pos_au_moon[1]:.4f}] AU, Vel_sim_s=[{vel_au_s_moon[0]:.4f}, {vel_au_s_moon[1]:.4f}] AU/s")
                    rel_pos_au = (moon_body.position_sim - primary_body.position_sim) / AU_SCALE
                    print(f"DEBUG: {moon_name} initial pos rel {primary_name}: [{rel_pos_au[0]:.4f}, {rel_pos_au[1]:.4f}] AU, dist: {np.linalg.norm(rel_pos_au):.4f} AU (target ~{moon_data['semi_major_axis_au']:.4f} AU)")

    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        if self.sun:
            self.orbital_mechanics.update_celestial_body_positions(
                self.planets, 
                self.sun,
                SIM_SECONDS_PER_STEP
            )
        
        if DEBUG_ORBITAL_MECHANICS and self.step_count % 100 == 0 and self.sun:
            # Create a temporary map for debug reference within this scope
            temp_body_map_for_debug = {p.name: p for p in self.planets}
            temp_body_map_for_debug[self.sun.name] = self.sun 

            for body in self.planets + [self.sun]: 
                if body.name == "Sun":
                    pos_au_sun_debug = body.position_sim / AU_SCALE
                    print(f"Step {self.step_count}: Sun - Pos: [{pos_au_sun_debug[0]:.3f}, {pos_au_sun_debug[1]:.3f}] AU")
                    continue

                central_body_for_debug = None
                if body.orbits_around == "Sun":
                    central_body_for_debug = self.sun
                elif body.orbits_around is not None: 
                    central_body_for_debug = temp_body_map_for_debug.get(body.orbits_around)
                
                if central_body_for_debug:
                    distance_vector_sim = body.position_sim - central_body_for_debug.position_sim
                    distance_au = np.linalg.norm(distance_vector_sim) / AU_SCALE
                    speed_au_s = np.linalg.norm(body.velocity_sim_s) / AU_SCALE
                    
                    pos_au_step_debug = body.position_sim / AU_SCALE
                    print(f"Step {self.step_count}: {body.name} - "
                          f"Pos_AU: [{pos_au_step_debug[0]:.3f}, {pos_au_step_debug[1]:.3f}], "
                          f"Dist_to_{central_body_for_debug.name}: {distance_au:.4f} AU (a={body.a_au:.4f}), "
                          f"Speed_AU/s: {speed_au_s:.4f}")
        
        observations, rewards, dones, infos = super().step(actions)
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
        for planet_body in self.planets: # Includes moons
            render_data.append({
                'name': planet_body.name,
                'position': planet_body.position_sim.tolist(), 
                'radius_sim': planet_body.display_radius_sim, 
                'color': planet_body.color,
                'orbit_path': [p.tolist() for p in planet_body.orbit_path] 
            })
        return render_data