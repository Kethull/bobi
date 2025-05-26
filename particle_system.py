import pygame
import numpy as np
import math
import random
from config import config # Use the global config instance
from spatial_partition import Rectangle, Quadtree # Import Quadtree components

class Particle:
    """
    Represents a single particle in the particle system.

    Particles have properties like position, velocity, lifespan, size, color,
    and type, which influences their behavior and appearance. They can be
    pooled and reused for efficiency.

    Attributes:
        pos (np.ndarray): Current 2D position [x, y] of the particle.
        vel (np.ndarray): Current 2D velocity [vx, vy] of the particle.
        life (int): Remaining lifespan in simulation steps.
        max_life (float): Initial lifespan, used for alpha calculations.
        size (float): Current size of the particle.
        color (Tuple[int, int, int]): Base color of the particle.
        particle_type (str): Type of particle (e.g., 'exhaust', 'spark', 'energy').
                             Affects update logic and rendering.
        temperature (float): Temperature of the particle (primarily for 'exhaust' type), in Kelvin.
        gravity_affected (bool): Whether the particle is affected by gravity (not currently implemented).
        active (bool): True if the particle is currently active in the simulation,
                       False if it's in the pool or has expired.
    """
    def __init__(self, pos: np.ndarray, vel: np.ndarray, life: int, size: float, color: Tuple[int, int, int], particle_type: str = 'default'):
        """
        Initializes a new particle.

        Args:
            pos (np.ndarray): Initial position [x, y].
            vel (np.ndarray): Initial velocity [vx, vy].
            life (int): Initial lifespan in steps.
            size (float): Initial size.
            color (Tuple[int, int, int]): Base color (R, G, B).
            particle_type (str, optional): Type of particle. Defaults to 'default'.
        """
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.life = life
        self.max_life = float(life)
        self.size = size
        self.color = color
        self.particle_type = particle_type
        self.temperature = 3000  # Default for exhaust, K
        self.gravity_affected = False
        self.active = True

    def reset(self, pos: np.ndarray, vel: np.ndarray, life: int, size: float, color: Tuple[int, int, int], particle_type: str = 'default', temperature: float = 3000, gravity_affected: bool = False) -> 'Particle':
        """
        Re-initializes an existing particle, typically from an object pool.

        Args:
            pos (np.ndarray): New initial position.
            vel (np.ndarray): New initial velocity.
            life (int): New initial lifespan.
            size (float): New initial size.
            color (Tuple[int, int, int]): New base color.
            particle_type (str, optional): New particle type. Defaults to 'default'.
            temperature (float, optional): New temperature. Defaults to 3000.
            gravity_affected (bool, optional): New gravity affected status. Defaults to False.

        Returns:
            Particle: The re-initialized particle instance (self).
        """
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.life = life
        self.max_life = float(life)
        self.size = size
        self.color = color
        self.particle_type = particle_type
        self.temperature = temperature
        self.gravity_affected = gravity_affected
        self.active = True
        return self
        
    def update(self) -> bool:
        """
        Updates the particle's state for one simulation step.

        Moves the particle, decreases its lifespan, and applies type-specific
        behavior (e.g., velocity damping, size reduction, temperature change).
        Sets `active` to False if lifespan reaches zero.

        Returns:
            bool: True if the particle is still active after the update, False otherwise.
        """
        if not self.active:
            return False

        self.pos += self.vel
        self.life -= 1
        
        if self.particle_type == 'exhaust':
            self.vel *= 0.98
            self.temperature *= 0.92
            self.size *= 0.99
            if self.size < 0.5: self.life = 0
        elif self.particle_type == 'spark':
            self.vel *= 0.95
            self.size *= 0.97
            if self.size < 0.5: self.life = 0
        elif self.particle_type == 'energy' or self.particle_type == 'elegant_exhaust': # elegant_exhaust shares some energy behavior
            self.size *= 0.98
            if self.size < 0.3: self.life = 0
        
        if self.life <= 0:
            self.active = False
            return False
        return True
    
    def get_render_color_and_alpha(self) -> Tuple[Tuple[int, int, int], int]:
        """
        Calculates the render color and alpha for the particle based on its type and state.

        For 'exhaust' particles, color and alpha depend on temperature and remaining life.
        For 'energy' particles, color may pulse and alpha is generally lower.
        For 'spark' particles, color is maintained, and alpha fades with life.

        Returns:
            Tuple[Tuple[int, int, int], int]: A tuple containing the (R, G, B) render color
                                              and an alpha value (0-255).
        """
        life_ratio = self.life / self.max_life if self.max_life > 0 else 0.0
        
        render_color = self.color
        alpha = int(255 * life_ratio)

        if self.particle_type == 'exhaust' or self.particle_type == 'elegant_exhaust':
            temp_norm = self.temperature / 3000.0 # Normalize temperature for color mapping
            if self.temperature > 2500:
                render_color = (255, int(220 * temp_norm), int(180 * temp_norm**2))
            elif self.temperature > 1500:
                render_color = (255, int(165 * temp_norm), int(80 * temp_norm**2))
            else:
                render_color = (200, int(80 * temp_norm), int(40 * temp_norm**2))
            alpha = int(255 * life_ratio * (temp_norm**0.5))
        
        elif self.particle_type == 'spark':
            # Color is already set, alpha handles fade based on life_ratio
            pass

        elif self.particle_type == 'energy':
            pulse = (math.sin(self.life * 0.5 + self.pos[0] * 0.1) + 1) / 2 # Add spatial variation to pulse
            render_color = tuple(min(255, int(c * (0.6 + 0.4 * pulse))) for c in self.color)
            alpha = int(180 * life_ratio**0.7) # Slightly different fade for energy

        # Ensure color components are valid
        render_color = tuple(max(0, min(255, int(c))) for c in render_color)
        return render_color, max(0, min(255, alpha))


class AdvancedParticleSystem:
    """
    Manages a collection of particles for various visual effects.

    This system handles the emission, update, rendering, and pooling of particles.
    It uses a Quadtree for efficient spatial querying of particles, although this
    is not heavily utilized in the current rendering logic but is available for
    potential future interactions or culling.

    Attributes:
        particles (List[Particle]): List of currently active particles.
        particle_pool (List[Particle]): List of inactive particles available for reuse.
        max_particles (int): Maximum number of active particles allowed,
                             defined by `config.Visualization.MAX_PARTICLES_PER_SYSTEM`.
        quadtree (Quadtree): Quadtree for spatial partitioning of active particles.
                             Its boundary is based on `config.World` dimensions.
    """
    def __init__(self):
        """
        Initializes the AdvancedParticleSystem.

        Sets up the particle lists, pool, maximum particle limit from config,
        and initializes the Quadtree with boundaries matching the simulation world.
        """
        self.particles = []
        self.particle_pool = []
        self.max_particles = config.Visualization.MAX_PARTICLES_PER_SYSTEM

        world_boundary = Rectangle(
            config.World.CENTER_SIM[0],
            config.World.CENTER_SIM[1],
            config.World.WIDTH_SIM / 2, # Half-width for Rectangle constructor
            config.World.HEIGHT_SIM / 2  # Half-height for Rectangle constructor
        )
        self.quadtree = Quadtree(
            world_boundary,
            config.SpatialPartitioning.QUADTREE_CAPACITY,
            config.SpatialPartitioning.QUADTREE_MAX_DEPTH
        )
        
    def _get_or_create_particle(self, pos: np.ndarray, vel: np.ndarray, life: int, size: float, color: Tuple[int, int, int], particle_type: str = 'default', temperature: float = 3000, gravity_affected: bool = False) -> Particle:
        """
        Retrieves a particle from the pool or creates a new one if the pool is empty.

        Args:
            pos (np.ndarray): Initial position.
            vel (np.ndarray): Initial velocity.
            life (int): Initial lifespan.
            size (float): Initial size.
            color (Tuple[int, int, int]): Base color.
            particle_type (str, optional): Particle type. Defaults to 'default'.
            temperature (float, optional): Temperature. Defaults to 3000.
            gravity_affected (bool, optional): Gravity status. Defaults to False.

        Returns:
            Particle: An initialized or re-initialized particle.
        """
        if self.particle_pool:
            particle = self.particle_pool.pop()
            particle.reset(pos, vel, life, size, color, particle_type, temperature, gravity_affected)
        else:
            # Create a new particle if pool is empty and we are below max_particles overall
            # Note: add_particle handles the max_particles logic for the active list.
            # This method just focuses on object creation/reuse.
            particle = Particle(pos, vel, life, size, color, particle_type)
            particle.temperature = temperature # Ensure these are set if not in Particle.__init__
            particle.gravity_affected = gravity_affected
        return particle

    def emit_thruster_exhaust(self, pos: np.ndarray, angle: float, thrust_power_level: float, thrust_ramp_ratio: float = 1.0):
        """
        Emits particles simulating thruster exhaust.

        The number, velocity, size, and temperature of particles depend on
        `thrust_power_level` and `thrust_ramp_ratio`. Particle properties are
        randomized for a more natural look. Uses `config.Probe.THRUST_FORCE_MAGNITUDES`
        and `config.Visualization.PROBE_SIZE_PX`.

        Args:
            pos (np.ndarray): Position of the thruster.
            angle (float): Angle of the probe/thruster (radians), exhaust goes opposite.
            thrust_power_level (float): Current power level of the thruster (e.g., index or normalized value).
            thrust_ramp_ratio (float, optional): Ramping factor (0.0 to 1.0). Defaults to 1.0.
        """
        if thrust_power_level <= 0 or thrust_ramp_ratio <= 0.01:
            return
            
        exhaust_angle = angle + math.pi # Exhaust goes opposite to probe's facing angle
        
        probe_thrust_forces = config.Probe.THRUST_FORCE_MAGNITUDES
        probe_size_px = config.Visualization.PROBE_SIZE_PX # Used for offset

        # Normalize thrust_power_level if it's an index
        # Assuming thrust_power_level is 0 for no thrust, 1 for first level, etc.
        # If it's already a normalized factor, this logic might need adjustment.
        max_thrust_level_idx = len(probe_thrust_forces) - 1 if len(probe_thrust_forces) > 1 else 1
        normalized_power = thrust_power_level / max_thrust_level_idx if max_thrust_level_idx > 0 else 0
        base_intensity_factor = normalized_power * thrust_ramp_ratio
        
        particle_count = int(base_intensity_factor * 20) + 2 # Min 2 particles if any thrust
        base_velocity_magnitude = base_intensity_factor * 4.0 + 1.0 # Base speed of particles
        
        for _ in range(particle_count):
            if len(self.particles) >= self.max_particles and not self.particle_pool: continue # Optimization

            spread = math.radians(random.uniform(-20, 20) * (1.0 - base_intensity_factor * 0.5)) # Narrower cone at high thrust
            p_angle = exhaust_angle + spread
            p_vel_mag = base_velocity_magnitude * random.uniform(0.7, 1.3)
            vel = np.array([math.cos(p_angle) * p_vel_mag, math.sin(p_angle) * p_vel_mag])
            
            # Offset particle start position slightly behind the probe based on its angle
            offset_distance = probe_size_px * 0.3 # Adjust as needed
            start_pos_offset = np.array([math.cos(angle) * offset_distance, math.sin(angle) * offset_distance])
            
            p_pos = pos - start_pos_offset # Start behind the emitter
            p_life = random.randint(20, int(45 * (1 + base_intensity_factor))) # Longer life for stronger thrust
            p_size = random.uniform(1.0, 3.0) * base_intensity_factor + 0.5
            p_color = (255,180,100) # Base color, will be modified by temperature in get_render_color_and_alpha
            p_temp = random.uniform(2200, 3800) * base_intensity_factor

            particle = self._get_or_create_particle(p_pos, vel, p_life, p_size, p_color, 'exhaust', temperature=p_temp)
            self.add_particle(particle)
        
    def emit_mining_sparks(self, impact_pos: np.ndarray, surface_normal_angle: Optional[float] = None, intensity: float = 1.0):
        """
        Emits particles simulating sparks from a mining operation or impact.

        Args:
            impact_pos (np.ndarray): Position of the impact/mining.
            surface_normal_angle (Optional[float], optional): Angle of the surface normal
                at the impact point (radians). If provided, sparks emit away from it.
                Defaults to None (omnidirectional sparks).
            intensity (float, optional): Intensity of the effect (0.0 to 1.0+). Defaults to 1.0.
        """
        spark_count = int(intensity * 10) + 3
        colors = [(255,255,150), (255,200,100), (255,180,50)] # Yellow/orange sparks

        for _ in range(spark_count):
            if len(self.particles) >= self.max_particles and not self.particle_pool: continue

            if surface_normal_angle is not None:
                # Sparks generally reflect off the surface
                base_angle = surface_normal_angle # Start with normal, then add spread
                angle_spread = math.radians(random.uniform(-70, 70)) # Wider spread for sparks
                p_angle = base_angle + angle_spread
            else:
                p_angle = random.uniform(0, 2 * math.pi) # Omnidirectional if no normal
                
            speed = random.uniform(1.5, 5.0) * intensity
            vel = np.array([math.cos(p_angle) * speed, math.sin(p_angle) * speed])
            
            p_pos = impact_pos + np.array([random.gauss(0,1), random.gauss(0,1)]) # Slight position jitter
            p_life = random.randint(15, 30)
            p_size = random.uniform(0.8, 2.2)
            p_color = random.choice(colors)

            particle = self._get_or_create_particle(p_pos, vel, p_life, p_size, p_color, 'spark')
            self.add_particle(particle)
            
    def emit_communication_pulse(self, start_pos: np.ndarray, end_pos: np.ndarray, pulse_speed: float = 4.0):
        """
        Emits particles simulating a communication pulse traveling between two points.

        Args:
            start_pos (np.ndarray): Starting position of the pulse.
            end_pos (np.ndarray): Target position of the pulse.
            pulse_speed (float, optional): Speed of the pulse particles. Defaults to 4.0.
        """
        direction_vec = np.array(end_pos) - np.array(start_pos)
        distance = np.linalg.norm(direction_vec)
        if distance < 1e-3: return # Avoid division by zero if start and end are too close
        
        norm_direction = direction_vec / distance
        num_segments = max(3, int(distance / 15)) # Create segments along the path

        for i in range(num_segments):
            if len(self.particles) >= self.max_particles and not self.particle_pool: continue

            pos_offset = norm_direction * (distance * (i / float(num_segments)))
            current_pos = np.array(start_pos) + pos_offset
            vel = norm_direction * pulse_speed
            p_life = int(distance / pulse_speed) + 5 # Lifespan to roughly reach target
            p_size = random.uniform(1.5, 2.5)
            p_color = (100, 180, 255) # Bluish energy color

            particle = self._get_or_create_particle(current_pos, vel, p_life, p_size, p_color, 'energy')
            self.add_particle(particle)

    def emit_energy_field(self, center_pos: np.ndarray, radius: float, energy_intensity_factor: float):
        """
        Emits particles simulating a diffuse energy field.

        Args:
            center_pos (np.ndarray): Center position of the energy field.
            radius (float): Radius of the field.
            energy_intensity_factor (float): Intensity factor (0.0 to 1.0),
                                             affecting particle count and behavior.
        """
        if energy_intensity_factor < 0.1: return
        particle_count = int(energy_intensity_factor * 15)

        for _ in range(particle_count):
            if len(self.particles) >= self.max_particles and not self.particle_pool: continue

            angle = random.uniform(0, 2 * math.pi)
            dist_from_center = radius * random.uniform(0.5, 1.0) # Particles within the radius
            pos_offset = np.array([math.cos(angle) * dist_from_center, math.sin(angle) * dist_from_center])
            
            # Gentle swirling motion
            vel_tangential = np.array([-pos_offset[1], pos_offset[0]]) * 0.005 * energy_intensity_factor
            vel_radial = pos_offset * random.uniform(-0.002, 0.002) # Slight in/out movement
            
            p_pos = center_pos + pos_offset
            p_vel = vel_tangential + vel_radial
            p_life = random.randint(40, 80)
            p_size = random.uniform(0.8, 1.8)
            p_color = (50, 100, 200) # Cool energy color

            particle = self._get_or_create_particle(p_pos, p_vel, p_life, p_size, p_color, 'energy')
            self.add_particle(particle)

    def emit_organic_thruster_exhaust(self, pos: np.ndarray, angle: float, thrust_power: float, thrust_ramp: float = 1.0):
        """
        Emits particles for an "organic" or "elegant" thruster style.

        Uses `config.Visualization.ORGANIC_EXHAUST_PARTICLE_SCALE`.

        Args:
            pos (np.ndarray): Position of the thruster.
            angle (float): Angle of the probe/thruster (radians).
            thrust_power (float): Power of the thrust (affects particle count and velocity).
            thrust_ramp (float, optional): Ramping factor (0.0 to 1.0). Defaults to 1.0.
        """
        if thrust_power <= 0: return
        particle_count = int(thrust_power * thrust_ramp * 8)
        
        organic_scale = config.Visualization.ORGANIC_EXHAUST_PARTICLE_SCALE

        for _ in range(particle_count):
            if len(self.particles) >= self.max_particles and not self.particle_pool: continue

            spread_angle = random.gauss(0, 0.15) # Gaussian spread for a softer look
            particle_angle = angle + math.pi + spread_angle # Exhaust opposite to probe angle
            base_velocity = thrust_power * 2.5 * thrust_ramp
            velocity_variation = base_velocity * 0.15
            velocity_magnitude = base_velocity + random.gauss(0, velocity_variation)
            vel = np.array([math.cos(particle_angle) * velocity_magnitude, math.sin(particle_angle) * velocity_magnitude])
            
            p_pos = pos + np.array([random.gauss(0, 1.5 * organic_scale), random.gauss(0, 1.5 * organic_scale)]) # Softer origin spread
            p_life = random.randint(20, 40)
            p_size = random.uniform(1.0, 2.2) * organic_scale
            p_color = (200, 220, 255) # Light, ethereal color
            p_temp = random.uniform(1800, 2800) # Cooler temperature range

            particle = self._get_or_create_particle(p_pos, vel, p_life, p_size, p_color, 'elegant_exhaust', temperature=p_temp)
            self.add_particle(particle)

    def add_particle(self, particle_to_add: Particle):
        """
        Adds a particle to the active list, managing the maximum particle limit.

        If the active particle list is at `max_particles`, the oldest particle
        is removed, marked inactive, and added to the `particle_pool` before
        the new particle is added.

        Args:
            particle_to_add (Particle): The particle instance to add.
        """
        if len(self.particles) >= self.max_particles:
            if self.particles: # Ensure list is not empty before pop
                oldest_particle = self.particles.pop(0) # FIFO for removal
                oldest_particle.active = False
                self.particle_pool.append(oldest_particle)
            # If pool was also full, this new particle might be dropped if not handled by emitter
            # However, current logic is to always add, potentially exceeding max_particles if pool is also full.
            # For strict max_particles, might need to check pool size or simply not add if oldest_particle couldn't be pooled.
            # Current behavior: always adds the new one after making space.

        self.particles.append(particle_to_add)
        # Quadtree insertion is handled in update() after all particles are processed for the frame.


    def update(self):
        """
        Updates all active particles and rebuilds the Quadtree.

        Iterates through active particles, calls their `update()` method.
        Particles that become inactive (lifespan ends) are moved to the
        `particle_pool`. The Quadtree is cleared and then repopulated with
        all currently active particles.
        """
        self.quadtree.clear()
        
        active_particles_after_update = []
        for particle in self.particles:
            if particle.update():
                active_particles_after_update.append(particle)
                # Defer Quadtree insertion to a single pass after all updates if preferred,
                # or insert here. Current Quadtree implementation might be fine with per-particle insert.
                # For simplicity with current Quadtree, inserting here is okay.
                # If Quadtree was more complex (e.g. dynamic boundaries), a rebuild pass might be better.
                self.quadtree.insert(particle)
            else:
                particle.active = False # Ensure it's marked inactive
                self.particle_pool.append(particle)
        
        self.particles = active_particles_after_update
    
    def query_particles_in_region(self, query_rect: Rectangle) -> List[Particle]:
        """
        Queries the Quadtree for active particles within a given rectangular region.

        Args:
            query_rect (Rectangle): The rectangular area to query.

        Returns:
            List[Particle]: A list of `Particle` objects found within the region.
        """
        found_particles: List[Particle] = []
        if self.quadtree: # Ensure quadtree is initialized
            self.quadtree.query(query_rect, found_particles)
        return found_particles

    def render(self, surface: pygame.Surface):
        """
        Renders all active particles onto the given Pygame surface.

        Calculates each particle's render color and alpha. Uses alpha blending
        for certain particle types (exhaust, energy) often with `BLEND_RGBA_ADD`
        for a glowing effect.

        Args:
            surface (pygame.Surface): The Pygame surface to draw on.
        """
        if not surface: return

        for p in self.particles:
            if not p.active: continue # Should not happen if update() correctly manages self.particles

            color, alpha = p.get_render_color_and_alpha()
            if alpha == 0: continue

            pos_int = p.pos.astype(int)
            size_int = max(1, int(p.size)) # Ensure size is at least 1 for drawing

            try:
                if p.particle_type in ['exhaust', 'energy', 'elegant_exhaust']:
                    temp_surf_size = size_int * 4 # Surface large enough for glow
                    if temp_surf_size <= 0: continue # Avoid creating zero-size surface
                    
                    temp_surf = pygame.Surface((temp_surf_size, temp_surf_size), pygame.SRCALPHA)
                    temp_surf.fill((0,0,0,0)) # Ensure fully transparent background
                    
                    center_on_temp = (temp_surf_size // 2, temp_surf_size // 2)
                    
                    pygame.draw.circle(temp_surf, (*color, alpha), center_on_temp, size_int)
                    
                    if p.particle_type in ['exhaust', 'elegant_exhaust']:
                        glow_alpha = int(alpha * 0.3)
                        if glow_alpha > 10:
                             pygame.draw.circle(temp_surf, (*color, glow_alpha), center_on_temp, size_int + 2)
                    
                    # Blit onto main surface with additive blending for glow
                    surface.blit(temp_surf, (pos_int[0] - center_on_temp[0], pos_int[1] - center_on_temp[1]), special_flags=pygame.BLEND_RGBA_ADD)

                elif p.particle_type == 'spark':
                    # Sparks are solid, bright, no special blending needed beyond their color/alpha
                    # If alpha is less than 255, it implies fading, but draw.circle doesn't use per-pixel alpha directly.
                    # For true alpha on sparks, would need temp surface like above, but without BLEND_ADD.
                    # Current get_render_color_and_alpha for sparks just passes base color and life-based alpha.
                    # To make sparks fade, they'd need a temp surface or a color that incorporates alpha.
                    # For simplicity, drawing solid if alpha > some threshold, or skip.
                    if alpha > 20: # Only draw reasonably visible sparks
                        pygame.draw.circle(surface, color, pos_int, size_int)
                else: # Default rendering for other types
                    # This also won't have true alpha unless color tuple has 4 elements and surface supports it.
                    # pygame.draw.circle uses the surface's colorkey or global alpha if set on surface.
                    # For per-pixel alpha on default particles, a temp surface is needed.
                    # Assuming for 'default' type, we just want a simple circle.
                    pygame.draw.circle(surface, color, pos_int, size_int)
            except pygame.error as e:
                logging.error(f"Pygame error rendering particle type {p.particle_type} at {pos_int}: {e}", exc_info=True)
            except Exception as e:
                logging.error(f"Unexpected error rendering particle: {e}", exc_info=True)