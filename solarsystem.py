# solarsystem.py
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Tuple, List
from config import (
    AU_SCALE, SUN_POSITION_SIM, PLANET_DATA, SIM_SECONDS_PER_STEP,
    GRAVITATIONAL_CONSTANT_KM3_KG_S2, AU_KM, KM_SCALE, DEBUG_ORBITAL_MECHANICS
)

@dataclass
class CelestialBody:
    name: str
    mass_kg: float
    radius_km: float
    display_radius_sim: float
    color: Tuple[int, int, int]
    
    # Orbital elements (J2000.0 epoch)
    a_au: float  # Semi-major axis in AU
    e: float  # Eccentricity
    i_deg: float  # Inclination in degrees
    omega_deg: float  # Longitude of ascending node in degrees (Ω)
    w_deg: float  # Argument of perihelion in degrees (ω)
    m0_deg: float  # Mean anomaly at epoch in degrees (M0)
    
    # State vectors in simulation units
    # position_sim is relative to the central body if not the Sun, otherwise world coords.
    # For planets, this will be updated to world coords after initial calculation.
    position_sim: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0], dtype=np.float64)) # [x, y] in simulation units
    velocity_sim_s: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0], dtype=np.float64)) # [vx, vy] in simulation units per second

    # For Moon, this would be its primary (e.g., Earth)
    orbits_around: str = None # Name of the body it orbits, None for Sun or bodies orbiting Sun

    def __post_init__(self):
        if not isinstance(self.position_sim, np.ndarray):
            self.position_sim = np.array(self.position_sim, dtype=np.float64)
        if not isinstance(self.velocity_sim_s, np.ndarray):
            self.velocity_sim_s = np.array(self.velocity_sim_s, dtype=np.float64)

class OrbitalMechanics:
    def __init__(self):
        self.G_KM3_KG_S2 = GRAVITATIONAL_CONSTANT_KM3_KG_S2
        self.KM_SCALE = KM_SCALE
        self.AU_KM = AU_KM

    def calculate_initial_state_vector(self, celestial_body: CelestialBody, central_body_mass_kg: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the initial 2D position and velocity vectors for an orbiting body
        relative to its central body.
        Position is in simulation units, velocity is in simulation units per second.
        """
        if celestial_body.a_au <= 0: # For the Sun itself or invalid data
            return np.array([0.0, 0.0]), np.array([0.0, 0.0])

        # 1. Convert semi-major axis from AU to km
        a_km = celestial_body.a_au * self.AU_KM

        # 2. Calculate mean motion n (radians/sec)
        # mu = G * M_central
        mu_km3_s2 = self.G_KM3_KG_S2 * central_body_mass_kg
        if a_km**3 <= 0: # Avoid division by zero or sqrt of negative
             return np.array([0.0,0.0]), np.array([0.0,0.0]) # Should not happen with a_au > 0
        n_rad_s = math.sqrt(mu_km3_s2 / a_km**3)

        # 3. Calculate mean anomaly M at epoch (t=0)
        M_rad = np.radians(celestial_body.m0_deg)
        e = celestial_body.e

        # 4. Solve Kepler's Equation for eccentric anomaly E: M = E - e * sin(E)
        # Use Newton-Raphson: E_next = E - (E - e * sin(E) - M) / (1 - e * cos(E))
        E_rad = M_rad # Initial guess for E
        if e > 0.8: # For high eccentricities, M is not a good first guess.
            E_rad = np.pi
            
        for _ in range(100): # Max 100 iterations for convergence
            f_E = E_rad - e * np.sin(E_rad) - M_rad
            f_prime_E = 1 - e * np.cos(E_rad)
            if abs(f_prime_E) < 1e-10: # Avoid division by zero if derivative is too small
                break
            E_next = E_rad - f_E / f_prime_E
            if abs(E_next - E_rad) < 1e-9: # Convergence criterion
                E_rad = E_next
                break
            E_rad = E_next
        else: # Did not converge
            if DEBUG_ORBITAL_MECHANICS:
                print(f"Warning: Kepler's eq. for {celestial_body.name} did not converge. M={M_rad}, e={e}, Last E={E_rad}")


        # 5. Calculate true anomaly nu (ν)
        cos_E = np.cos(E_rad)
        sin_E = np.sin(E_rad)
        
        # Robust atan2 version for nu
        nu_rad = np.arctan2(np.sqrt(1 - e**2) * sin_E, cos_E - e)

        # 6. Calculate distance to central body r_km
        r_km = a_km * (1 - e * cos_E)

        # 7. Calculate position in the orbital plane (perifocal coordinates)
        x_orb_km = r_km * np.cos(nu_rad)
        y_orb_km = r_km * np.sin(nu_rad)

        # 8. Calculate velocity in the orbital plane (km/s)
        # Using vis-viva equation for speed, then components
        # v_sq = mu_km3_s2 * (2/r_km - 1/a_km)
        # orbital_speed_km_s = math.sqrt(v_sq)
        # More direct components:
        # vx_orb_km_s = - (n_rad_s * a_km**2 / r_km) * np.sin(E_rad) # This is one form
        # vy_orb_km_s = (n_rad_s * a_km**2 / r_km) * np.sqrt(1 - e**2) * np.cos(E_rad) # This is one form
        
        # Alternative formulation for velocity components in orbital frame:
        # (sqrt(G * M * a) / r) * sin(E) and (sqrt(G * M * a * (1-e^2)) / r) * cos(E)
        # The negative sign for vx_orb_km_s depends on the coordinate system definition (e.g. x towards periapsis)
        # Let's use the common formulation:
        # vx' = -sqrt(mu*a)/r * sin(E)
        # vy' =  sqrt(mu*a*(1-e^2))/r * cos(E)
        # where mu = G * M_central
        
        if r_km == 0: # Avoid division by zero
            vx_orb_km_s = 0.0
            vy_orb_km_s = 0.0
        else:
            vx_orb_km_s = -(math.sqrt(mu_km3_s2 * a_km) / r_km) * sin_E
            vy_orb_km_s = (math.sqrt(mu_km3_s2 * a_km * (1 - e**2)) / r_km) * cos_E


        # 9. Rotation to Ecliptic/Simulation Plane (simplified 2D: i=0)
        # For 2D, we rotate by (argument of perihelion ω + longitude of ascending node Ω)
        # Let ang_rot_rad = ω + Ω
        w_rad = np.radians(celestial_body.w_deg)
        omega_rad = np.radians(celestial_body.omega_deg)
        # For 2D projection assuming i=0, the rotation is effectively by (w_rad + omega_rad)
        # If i_deg is non-zero but we are projecting to 2D, the full 3D rotation equations
        # would be used, and then z components ignored.
        # For this task, simplified: i=0 for projection.
        # P1 = cos(w)cos(omega) - sin(w)sin(omega)cos(i)
        # P2 = -sin(w)cos(omega) - cos(w)sin(omega)cos(i)
        # Q1 = cos(w)sin(omega) + sin(w)cos(omega)cos(i)
        # Q2 = -sin(w)sin(omega) + cos(w)cos(omega)cos(i)
        # With i=0, cos(i)=1:
        # P1 = cos(w)cos(omega) - sin(w)sin(omega) = cos(w+omega)
        # P2 = -sin(w)cos(omega) - cos(w)sin(omega) = -sin(w+omega)
        # Q1 = cos(w)sin(omega) + sin(w)cos(omega) = sin(w+omega)
        # Q2 = -sin(w)sin(omega) + cos(w)cos(omega) = cos(w+omega)
        
        # Note: inclination (i_deg) is stored but for 2D, we project as if i=0 for calculations.
        # The true rotation involves i_deg. For a strict 2D view on the ecliptic,
        # if the body's inclination is non-zero, its projected path is complex.
        # The instruction "assume i=0 for the projection" simplifies this.
        # The angle for 2D rotation in the plane is effectively the sum of longitude of ascending node (omega_deg)
        # and argument of perihelion (w_deg).
        
        angle_sum_rad = w_rad + omega_rad
        cos_angle_sum = np.cos(angle_sum_rad)
        sin_angle_sum = np.sin(angle_sum_rad)

        # x_ecl_km = (cos(w)cos(omega) - sin(w)sin(omega)cos(i)) * x_orb_km + (-sin(w)cos(omega) - cos(w)sin(omega)cos(i)) * y_orb_km
        # y_ecl_km = (cos(w)sin(omega) + sin(w)cos(omega)cos(i)) * x_orb_km + (-sin(w)sin(omega) + cos(w)cos(omega)cos(i)) * y_orb_km
        # Simplified with i=0:
        # x_ecl_km = cos(w+omega) * x_orb_km - sin(w+omega) * y_orb_km
        # y_ecl_km = sin(w+omega) * x_orb_km + cos(w+omega) * y_orb_km
        
        x_ecl_km = cos_angle_sum * x_orb_km - sin_angle_sum * y_orb_km
        y_ecl_km = sin_angle_sum * x_orb_km + cos_angle_sum * y_orb_km

        vx_ecl_km_s = cos_angle_sum * vx_orb_km_s - sin_angle_sum * vy_orb_km_s
        vy_ecl_km_s = sin_angle_sum * vx_orb_km_s + cos_angle_sum * vy_orb_km_s
        
        # 10. Convert final position to simulation units
        position_sim = np.array([x_ecl_km, y_ecl_km]) * self.KM_SCALE

        # 11. Convert final velocity to simulation units per second
        velocity_sim_s = np.array([vx_ecl_km_s, vy_ecl_km_s]) * self.KM_SCALE
        
        return position_sim, velocity_sim_s

    def calculate_gravitational_acceleration(self, target_body: CelestialBody, other_bodies: List[CelestialBody]) -> np.ndarray:
        """
        Calculates the total gravitational acceleration on target_body due to all other_bodies.
        Returns acceleration in km/s^2.
        """
        total_accel_km_s2 = np.array([0.0, 0.0], dtype=np.float64)

        for other_body in other_bodies:
            if other_body is target_body: # Cannot exert force on itself
                continue

            # Vector from target_body to other_body in simulation units
            dist_vector_sim = other_body.position_sim - target_body.position_sim
            
            # Convert distance vector to km
            dist_vector_km = dist_vector_sim / self.KM_SCALE
            
            dist_sq_km = np.sum(dist_vector_km**2) # Squared distance in km^2
            
            if dist_sq_km == 0: # Avoid division by zero if bodies are at the same point
                continue
                
            dist_km = np.sqrt(dist_sq_km) # Distance in km

            # Gravitational force magnitude (Newton's law of gravitation)
            # F = G * m1 * m2 / r^2
            # This G is GRAVITATIONAL_CONSTANT_KM3_KG_S2
            # force_magnitude_N_equivalent = (self.G_KM3_KG_S2 * target_body.mass_kg * other_body.mass_kg) / dist_sq_km
            
            # Acceleration = F / m_target = G * m_other / r^2
            # The direction is from target_body towards other_body
            accel_magnitude_km_s2 = (self.G_KM3_KG_S2 * other_body.mass_kg) / dist_sq_km
            
            # Acceleration vector in km/s^2
            # Unit vector in direction of force (from target to other)
            direction_vector = dist_vector_km / dist_km
            accel_vector_km_s2 = accel_magnitude_km_s2 * direction_vector
            
            total_accel_km_s2 += accel_vector_km_s2
            
        return total_accel_km_s2

    def update_celestial_body_positions(self, all_bodies: List[CelestialBody], sun: CelestialBody, dt_seconds: float):
        """
        Updates positions and velocities of all celestial bodies using Euler integration.
        Handles n-body interactions for planets.
        Moon's orbit around Earth needs special handling (simplified for now or hierarchical).
        """
        
        # For bodies orbiting the Sun (planets)
        planets = [body for body in all_bodies if body.name != "Sun" and (body.orbits_around is None or body.orbits_around == "Sun")]
        
        for body in planets:
            # Bodies to consider for gravitational pull on 'body'
            # Includes the Sun and all other planets
            influencing_bodies = [sun] + [p for p in planets if p is not body]
            
            # If Moon is present and orbits Earth, Earth's calculation should include Moon's pull if desired for high accuracy,
            # but for now, we assume planets are dominant for each other and Sun.
            # Moon will be handled separately.

            # 1. Calculate net gravitational acceleration on body
            accel_km_s2 = self.calculate_gravitational_acceleration(body, influencing_bodies)
            
            # 2. Convert accel_km_s2 to accel_sim_s2
            accel_sim_s2 = accel_km_s2 * self.KM_SCALE
            
            # 3. Update velocity (Euler method)
            body.velocity_sim_s += accel_sim_s2 * dt_seconds
            
            # 4. Update position (Euler method)
            # Position is relative to Sun if calculated initially that way, then made absolute.
            # Here, position_sim is already in world coordinates.
            body.position_sim += body.velocity_sim_s * dt_seconds

            if DEBUG_ORBITAL_MECHANICS and body.name in ["Mercury", "Earth", "Mars"]:
                 print(f"DEBUG: {body.name} Pos={body.position_sim/AU_SCALE} AU, Vel={body.velocity_sim_s/AU_SCALE} AU/s, Accel_sim={accel_sim_s2}")


        # Handle Moon (orbiting Earth) - This is a simplified approach
        # For a more accurate model, Earth-Moon barycenter would orbit the Sun,
        # and Moon would orbit that barycenter (or Earth).
        # Here, Moon orbits Earth, which itself orbits the Sun.
        moon = next((b for b in all_bodies if b.name == "Moon" and b.orbits_around == "Earth"), None)
        earth = next((b for b in all_bodies if b.name == "Earth"), None)

        if moon and earth:
            # Influencing bodies for Moon: Earth (primary), Sun (perturbation), other planets (minor perturbations)
            # For this phase, let's simplify: Moon orbits Earth, Earth's position is already updated.
            # Acceleration on Moon due to Earth:
            accel_moon_due_to_earth_km_s2 = self.calculate_gravitational_acceleration(moon, [earth])
            
            # Acceleration on Moon due to Sun (perturbation)
            accel_moon_due_to_sun_km_s2 = self.calculate_gravitational_acceleration(moon, [sun])

            # Total acceleration on Moon
            total_accel_moon_km_s2 = accel_moon_due_to_earth_km_s2 + accel_moon_due_to_sun_km_s2
            # Could add other planets for more accuracy, but this is a start.

            accel_moon_sim_s2 = total_accel_moon_km_s2 * self.KM_SCALE

            moon.velocity_sim_s += accel_moon_sim_s2 * dt_seconds
            # Moon's position is relative to Earth's center of mass in its initial state vector calculation.
            # Here, its position_sim should be world coordinates.
            # If moon.position_sim was relative to Earth, it would be:
            # moon.position_sim += moon.velocity_sim_s * dt_seconds
            # And then its world position would be earth.position_sim + moon.position_sim (if moon.position_sim is relative)
            # However, CelestialBody.position_sim is intended to be world coordinates after initialization.
            # So, the update is direct.
            moon.position_sim += moon.velocity_sim_s * dt_seconds
            
            if DEBUG_ORBITAL_MECHANICS:
                print(f"DEBUG: Moon Pos={moon.position_sim/AU_SCALE} AU, Vel={moon.velocity_sim_s/AU_SCALE} AU/s (rel to Sun)")
                print(f"DEBUG: Earth Pos={earth.position_sim/AU_SCALE} AU")
                relative_pos_moon_earth_au = (moon.position_sim - earth.position_sim) / AU_SCALE
                print(f"DEBUG: Moon Pos rel Earth={relative_pos_moon_earth_au} AU, Dist={np.linalg.norm(relative_pos_moon_earth_au)} AU")


# Commenting out AsteroidBelt for now as it's not part of the immediate refactoring
# and relies on the old mechanics.
"""
class AsteroidBelt:
    def __init__(self, orbital_mechanics_instance: 'OrbitalMechanics'): # Forward reference OrbitalMechanics
        self.orbital_mechanics = orbital_mechanics_instance
        # Ensure these constants are available from config.py when this is instantiated
        from config import AU_SCALE, ASTEROID_BELT_INNER_AU, ASTEROID_BELT_OUTER_AU, \
                           ASTEROID_COUNT, ASTEROID_MIN_RADIUS_SIM, ASTEROID_MAX_RADIUS_SIM, \
                           ASTEROID_DEFAULT_COLOR, ASTEROID_MASS_KG_MIN, ASTEROID_MASS_KG_MAX, \
                           SUN_POSITION_SIM, PLANET_DATA # Use SUN_POSITION_SIM

        self.inner_radius_sim = ASTEROID_BELT_INNER_AU * AU_SCALE
        self.outer_radius_sim = ASTEROID_BELT_OUTER_AU * AU_SCALE
        self.asteroids: List[CelestialBody] = [] # Type hint for clarity
        self.generate_asteroids()

    def generate_asteroids(self):
        # Ensure these constants are available from config.py
        from config import AU_SCALE, ASTEROID_BELT_INNER_AU, ASTEROID_BELT_OUTER_AU, \
                           ASTEROID_COUNT, ASTEROID_MIN_RADIUS_SIM, ASTEROID_MAX_RADIUS_SIM, \
                           ASTEROID_DEFAULT_COLOR, ASTEROID_MASS_KG_MIN, ASTEROID_MASS_KG_MAX, \
                           SUN_POSITION_SIM, PLANET_DATA # Use SUN_POSITION_SIM
                           
        sun_mass_kg = PLANET_DATA['Sun']['mass_kg']
        self.asteroids = [] # Clear previous asteroids if regenerating

        for i in range(ASTEROID_COUNT):
            # This needs to be updated to use the new CelestialBody structure and initial state calculation
            # For now, this will break.
            # Example of what would need to change:
            # Asteroid data would need to be defined with all orbital elements similar to planets.
            # Then call self.orbital_mechanics.calculate_initial_state_vector(asteroid_body, sun_mass_kg)
            
            # Placeholder:
            # a_au = np.random.uniform(ASTEROID_BELT_INNER_AU, ASTEROID_BELT_OUTER_AU)
            # ecc = np.random.uniform(0.0, 0.25)
            # m0_deg_val = np.random.uniform(0, 360)
            # # ... other orbital elements (i, omega, w) would need to be defined or randomized appropriately
            # asteroid_definition = PLANET_DATA['Sun'].copy() # Just as a template for structure
            # asteroid_definition.update({
            #     'mass_kg': np.random.uniform(ASTEROID_MASS_KG_MIN, ASTEROID_MASS_KG_MAX),
            #     'radius_km': 1.0, # Placeholder
            #     'display_radius_sim': np.random.uniform(ASTEROID_MIN_RADIUS_SIM, ASTEROID_MAX_RADIUS_SIM),
            #     'color': ASTEROID_DEFAULT_COLOR,
            #     'a_au': a_au,
            #     'e': ecc,
            #     'i_deg': np.random.uniform(0,5), # Example inclination
            #     'omega_deg': np.random.uniform(0,360),
            #     'w_deg': np.random.uniform(0,360),
            #     'm0_deg': m0_deg_val
            # })
            # temp_body = CelestialBody(name=f"Asteroid_{i}", **asteroid_definition)
            # initial_pos_sim_rel_sun, initial_vel_sim_s_rel_sun = self.orbital_mechanics.calculate_initial_state_vector(
            #     temp_body, sun_mass_kg
            # )
            # pos_sim_world = np.array(SUN_POSITION_SIM, dtype=np.float64) + initial_pos_sim_rel_sun
            # temp_body.position_sim = pos_sim_world
            # temp_body.velocity_sim_s = initial_vel_sim_s_rel_sun
            # self.asteroids.append(temp_body)
            pass # End of placeholder

# Ensure imports for CelestialBody and OrbitalMechanics are correctly handled if this is at the top.
# It might be better to place this class after CelestialBody and OrbitalMechanics definitions.
# For now, assuming it's appended or prepended and Python handles resolution.
# The provided spec had it as a separate block.
"""