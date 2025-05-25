# solarsystem.py
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List
from config import AU_SCALE, SUN_POSITION, PLANET_DATA # Assuming these are available after config update

@dataclass
class CelestialBody:
    name: str
    mass_kg: float
    position: np.ndarray # [x, y] in simulation units
    velocity: np.ndarray # [vx, vy] in simulation units per second
    radius_sim: float # For display
    color: Tuple[int, int, int]
    # Orbital parameters (optional, can be stored if needed for direct updates or info)
    semi_major_axis_au: float = 0.0
    eccentricity: float = 0.0
    orbital_period_days: float = 0.0
    inclination_deg: float = 0.0

    def __post_init__(self):
        # Ensure position and velocity are numpy arrays
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float64)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=np.float64)

class OrbitalMechanics:
    def __init__(self):
        self.G_SI = 6.67430e-11 # m^3 kg^-1 s^-2
        # SIM_UNIT_TO_METER needs AU_SCALE from config.py
        # Ensure AU_SCALE is loaded before this class is instantiated if it's a global.
        # It's better to pass AU_SCALE or calculate SIM_UNIT_TO_METER during init if possible,
        # or ensure config.py is fully processed. For now, direct import.
        if AU_SCALE == 0: # Avoid division by zero if config not loaded as expected
            raise ValueError("AU_SCALE in config.py is not initialized or is zero.")
        self.SIM_UNIT_TO_METER = 1.496e11 / AU_SCALE

    def calculate_gravitational_force(self, body1_mass_kg: float, body2_mass_kg: float, dist_vector_sim_units: np.ndarray) -> np.ndarray:
        # dist_vector_sim_units is from body2 to body1 (force on body1 by body2)
        dist_vector_meters = dist_vector_sim_units * self.SIM_UNIT_TO_METER
        dist_sq_meters = np.sum(dist_vector_meters**2)
        
        if dist_sq_meters == 0:
            return np.array([0.0, 0.0], dtype=np.float64)
            
        force_magnitude_newtons = (self.G_SI * body1_mass_kg * body2_mass_kg) / dist_sq_meters
        
        # Normalized direction vector (from body2 towards body1)
        # Check for zero distance again before normalization to avoid division by zero if somehow dist_sq_meters was non-zero but sqrt is zero (highly unlikely with floats)
        dist_abs_meters = np.sqrt(dist_sq_meters)
        if dist_abs_meters == 0:
             return np.array([0.0, 0.0], dtype=np.float64)

        # Force should be attractive, so in the opposite direction of dist_vector_meters (which is Sun to Planet)
        force_vector_newtons = -force_magnitude_newtons * (dist_vector_meters / dist_abs_meters)
        return force_vector_newtons # This is F in Newtons (m*kg/s^2)

    def update_celestial_body_positions(self, celestial_bodies: List[CelestialBody], sun_mass_kg: float, dt_seconds: float):
        # For Phase 1, focus on 2-body problem for each planet with the Sun.
        # dt_seconds is the time step for this update (e.g., SIM_SECONDS_PER_STEP).
        
        # SUN_POSITION is assumed to be from config and in simulation units
        # It's used here to calculate the vector from the planet to the sun.
        # The Sun itself is considered stationary in this phase.

        for body in celestial_bodies:
            if body.name == "Sun": # Sun is stationary for this phase
                continue

            # Vector from body to Sun (for force direction on body)
            # dist_vector_to_sun_sim = np.array(SUN_POSITION) - body.position # This is vector Planet -> Sun
            # The force calculation expects vector from attracting body (Sun) to attracted body (Planet)
            # So, vector_sun_to_planet_sim = body.position - np.array(SUN_POSITION)

            vector_sun_to_planet_sim = body.position - np.array(SUN_POSITION, dtype=np.float64)
            
            # Force on planet from Sun.
            # calculate_gravitational_force expects dist_vector_sim_units from body2 (Sun) to body1 (planet)
            force_on_body_newtons = self.calculate_gravitational_force(
                body.mass_kg, sun_mass_kg, vector_sun_to_planet_sim
            ) 
            
            acceleration_m_s2 = force_on_body_newtons / body.mass_kg
            
            # Convert acceleration from m/s^2 to sim_units/s^2
            acceleration_sim_units_s2 = acceleration_m_s2 / self.SIM_UNIT_TO_METER
            
            # Update velocity (symplectic Euler: v_new = v_old + a*dt)
            body.velocity += acceleration_sim_units_s2 * dt_seconds
            # Update position (p_new = p_old + v_new*dt)
            body.position += body.velocity * dt_seconds

    def calculate_initial_state_vector(self, semi_major_axis_au: float, eccentricity: float, true_anomaly_rad: float, central_body_mass_kg: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the initial 2D position (AU) and velocity (AU/second) vectors for an orbiting body.
        Position and velocity are relative to the central body.
        Assumes true_anomaly_rad is the angle from the x-axis in the orbital plane.
        """
        AU_METERS = 1.496e11 # Astronomical unit in meters

        if semi_major_axis_au <= 0:
            return np.array([0.0, 0.0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)
        if eccentricity < 0 or eccentricity >= 1: # Basic check for elliptical orbits
            # For simplicity, fallback to circular if eccentricity is problematic, or raise error
            # For now, let's assume valid eccentricities are passed.
            # If e=0, formulas should simplify to circular.
            pass

        a_meters = semi_major_axis_au * AU_METERS
        
        # Distance from central body (r)
        if np.isclose(eccentricity, 0.0): # Handle purely circular case to avoid 1-e^2 = 1 in denominator if e is tiny but not zero
            r_meters = a_meters
        else:
            r_meters = a_meters * (1 - eccentricity**2) / (1 + eccentricity * np.cos(true_anomaly_rad))

        if r_meters <= 0: # Should not happen with valid a_meters and e < 1
             return np.array([0.0, 0.0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)

        # Position vector components (AU)
        pos_x_au = (r_meters / AU_METERS) * np.cos(true_anomaly_rad)
        pos_y_au = (r_meters / AU_METERS) * np.sin(true_anomaly_rad)
        initial_pos_au = np.array([pos_x_au, pos_y_au], dtype=np.float64)

        # Velocity components (m/s)
        # Semi-latus rectum (p) in meters
        p_meters = a_meters * (1 - eccentricity**2)
        if p_meters <= 0 and not np.isclose(eccentricity, 1.0): # p_meters can be 0 if e=1 (parabolic)
             # This indicates an issue, likely with a_meters or e
             return initial_pos_au, np.array([0.0, 0.0], dtype=np.float64) # Return current pos with zero vel

        # Standard gravitational parameter mu = G * M
        mu_SI = self.G_SI * central_body_mass_kg

        if np.isclose(p_meters, 0.0) and np.isclose(eccentricity, 0.0): # Circular orbit, r_meters = a_meters = p_meters
            # Simplified circular velocity
            v_circular_m_s = np.sqrt(mu_SI / r_meters)
            vel_x_m_s = -v_circular_m_s * np.sin(true_anomaly_rad) # Tangential
            vel_y_m_s = v_circular_m_s * np.cos(true_anomaly_rad)  # Tangential
        elif np.isclose(p_meters, 0.0) and eccentricity >= 1.0: # Parabolic/Hyperbolic edge case, not handled here
            vel_x_m_s = 0.0
            vel_y_m_s = 0.0
        else:
            # Radial velocity (m/s)
            v_r_m_s = np.sqrt(mu_SI / p_meters) * eccentricity * np.sin(true_anomaly_rad)
            # Tangential velocity (m/s)
            v_t_m_s = np.sqrt(mu_SI / p_meters) * (1 + eccentricity * np.cos(true_anomaly_rad))

            # Convert radial and tangential velocities to Cartesian components (m/s)
            vel_x_m_s = v_r_m_s * np.cos(true_anomaly_rad) - v_t_m_s * np.sin(true_anomaly_rad)
            vel_y_m_s = v_r_m_s * np.sin(true_anomaly_rad) + v_t_m_s * np.cos(true_anomaly_rad)
        
        initial_vel_m_s = np.array([vel_x_m_s, vel_y_m_s], dtype=np.float64)
        
        # Convert velocity from m/s to AU/s
        initial_vel_au_s = initial_vel_m_s / AU_METERS
        
        return initial_pos_au, initial_vel_au_s