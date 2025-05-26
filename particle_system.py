import pygame
import numpy as np
import math
import random
from config import config # Use the global config instance

class Particle:
    def __init__(self, pos, vel, life, size, color, particle_type='default'):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.life = life
        self.max_life = float(life) # Ensure max_life is float for division
        self.size = size
        self.color = color
        self.particle_type = particle_type
        self.temperature = 3000  # For exhaust particles, K
        self.gravity_affected = False # Default, can be overridden
        self.active = True # To manage in pool

    def reset(self, pos, vel, life, size, color, particle_type='default', temperature=3000, gravity_affected=False):
        """Re-initializes a particle from the pool."""
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
        
    def update(self):
        """Update particle physics. Returns True if still alive, False otherwise."""
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
        elif self.particle_type == 'energy':
            self.size *= 0.98
            if self.size < 0.3: self.life = 0
        
        if self.life <= 0:
            self.active = False
            return False
        return True
    
    def get_render_color_and_alpha(self):
        """Get color and alpha based on particle properties"""
        life_ratio = self.life / self.max_life if self.max_life > 0 else 0
        
        render_color = self.color
        alpha = int(255 * life_ratio)

        if self.particle_type == 'exhaust':
            if self.temperature > 2500: # Hot
                render_color = (255, 220, 180) # Bright yellow-white
            elif self.temperature > 1500: # Medium
                render_color = (255, 165, 80)  # Orange
            else: # Cool
                render_color = (200, 80, 40)   # Dim red/brown
            alpha = int(255 * life_ratio * (self.temperature / 3000)**0.5) # Alpha also depends on temp
        
        elif self.particle_type == 'spark':
            # Sparks maintain bright color but fade
            pass # Color is already set, alpha handles fade

        elif self.particle_type == 'energy':
            # Energy particles might pulse or change color
            pulse = (math.sin(self.life * 0.5) + 1) / 2 # Simple pulse effect
            render_color = tuple(min(255, int(c * (0.7 + 0.3 * pulse))) for c in self.color)
            alpha = int(150 * life_ratio) # Energy particles are more translucent

        return render_color, max(0, min(255, alpha))


class AdvancedParticleSystem:
    def __init__(self):
        self.particles = [] # Active particles
        self.particle_pool = [] # Pooled, inactive particles
        self.max_particles = config.Visualization.MAX_PARTICLES_PER_SYSTEM
        
    def _get_or_create_particle(self, pos, vel, life, size, color, particle_type='default', temperature=3000, gravity_affected=False):
        if self.particle_pool:
            particle = self.particle_pool.pop()
            particle.reset(pos, vel, life, size, color, particle_type, temperature, gravity_affected)
        else:
            particle = Particle(pos, vel, life, size, color, particle_type)
            # These might need to be set after init if not in Particle.__init__ or if reset doesn't cover all
            particle.temperature = temperature
            particle.gravity_affected = gravity_affected
        return particle

    def emit_thruster_exhaust(self, pos, angle, thrust_power_level, thrust_ramp_ratio=1.0):
        if thrust_power_level <= 0 or thrust_ramp_ratio <= 0.01:
            return
            
        exhaust_angle = angle + math.pi
        # Use config for THRUST_FORCE and SPACESHIP_SIZE if they are moved to config
        # For now, assuming they are global or accessible (e.g. from config.Probe if defined there)
        # This part might need adjustment based on where THRUST_FORCE and SPACESHIP_SIZE are defined.
        # Let's assume they are accessible via `config.Probe.THRUST_FORCE_MAGNITUDES` and `config.Visualization.PROBE_SIZE_PX`
        
        probe_thrust_forces = config.Probe.THRUST_FORCE_MAGNITUDES
        probe_size_px = config.Visualization.PROBE_SIZE_PX

        base_intensity_factor = (thrust_power_level / (len(probe_thrust_forces) -1 if len(probe_thrust_forces) > 1 else 1)) * thrust_ramp_ratio
        
        particle_count = int(base_intensity_factor * 20) + 2
        base_velocity_magnitude = base_intensity_factor * 4.0 + 1.0
        
        for _ in range(particle_count):
            spread = math.radians(random.uniform(-20, 20) * (1.0 - base_intensity_factor * 0.5))
            p_angle = exhaust_angle + spread
            p_vel_mag = base_velocity_magnitude * random.uniform(0.7, 1.3)
            vel = np.array([math.cos(p_angle) * p_vel_mag, math.sin(p_angle) * p_vel_mag])
            
            offset_distance = probe_size_px * 0.5
            start_pos_offset = np.array([math.cos(angle) * offset_distance, math.sin(angle) * offset_distance])
            
            p_pos = pos - start_pos_offset
            p_life = random.randint(20, 45)
            p_size = random.uniform(1.0, 3.0) * base_intensity_factor + 0.5
            p_color = (255,180,100)
            p_temp = random.uniform(2200, 3800) * base_intensity_factor

            particle = self._get_or_create_particle(p_pos, vel, p_life, p_size, p_color, 'exhaust', temperature=p_temp)
            self.add_particle(particle)
        
    def emit_mining_sparks(self, impact_pos, surface_normal_angle=None, intensity=1.0):
        spark_count = int(intensity * 10) + 3
        colors = [(255,255,150), (255,200,100), (255,180,50)]

        for _ in range(spark_count):
            if surface_normal_angle is not None:
                base_angle = surface_normal_angle + math.pi
                angle_spread = math.radians(random.uniform(-60, 60))
                p_angle = base_angle + angle_spread
            else:
                p_angle = random.uniform(0, 2 * math.pi)
                
            speed = random.uniform(1.5, 5.0) * intensity
            vel = np.array([math.cos(p_angle) * speed, math.sin(p_angle) * speed])
            
            p_pos = impact_pos + np.array([random.gauss(0,1), random.gauss(0,1)])
            p_life = random.randint(15, 30)
            p_size = random.uniform(0.8, 2.2)
            p_color = random.choice(colors)

            particle = self._get_or_create_particle(p_pos, vel, p_life, p_size, p_color, 'spark')
            self.add_particle(particle)
            
    def emit_communication_pulse(self, start_pos, end_pos, pulse_speed=4.0):
        direction_vec = np.array(end_pos) - np.array(start_pos)
        distance = np.linalg.norm(direction_vec)
        if distance < 1e-3: return
        
        norm_direction = direction_vec / distance
        num_segments = max(3, int(distance / 15))

        for i in range(num_segments):
            pos_offset = norm_direction * (distance * (i / float(num_segments)))
            current_pos = np.array(start_pos) + pos_offset
            vel = norm_direction * pulse_speed
            p_life = int(distance / pulse_speed) + 5
            p_size = random.uniform(1.5, 2.5)
            p_color = (100, 180, 255)

            particle = self._get_or_create_particle(current_pos, vel, p_life, p_size, p_color, 'energy')
            self.add_particle(particle)

    def emit_energy_field(self, center_pos, radius, energy_intensity_factor): # Factor 0-1
        if energy_intensity_factor < 0.1: return
        particle_count = int(energy_intensity_factor * 15)

        for _ in range(particle_count):
            angle = random.uniform(0, 2 * math.pi)
            dist_from_center = radius * random.uniform(0.5, 1.0)
            pos_offset = np.array([math.cos(angle) * dist_from_center, math.sin(angle) * dist_from_center])
            vel_tangential = np.array([-pos_offset[1], pos_offset[0]]) * 0.005 * energy_intensity_factor
            vel_radial = pos_offset * random.uniform(-0.002, 0.002)
            
            p_pos = center_pos + pos_offset
            p_vel = vel_tangential + vel_radial
            p_life = random.randint(40, 80)
            p_size = random.uniform(0.8, 1.8)
            p_color = (50, 100, 200)

            particle = self._get_or_create_particle(p_pos, p_vel, p_life, p_size, p_color, 'energy')
            self.add_particle(particle)

    def emit_organic_thruster_exhaust(self, pos, angle, thrust_power, thrust_ramp=1.0):
            if thrust_power <= 0: return
            particle_count = int(thrust_power * thrust_ramp * 8)
            
            # Assuming ORGANIC_EXHAUST_PARTICLE_SCALE is in config.Visualization
            organic_scale = config.Visualization.ORGANIC_EXHAUST_PARTICLE_SCALE

            for _ in range(particle_count):
                spread_angle = random.gauss(0, 0.15)
                particle_angle = angle + math.pi + spread_angle
                base_velocity = thrust_power * 2.5 * thrust_ramp
                velocity_variation = base_velocity * 0.15
                velocity_magnitude = base_velocity + random.gauss(0, velocity_variation)
                vel = np.array([math.cos(particle_angle) * velocity_magnitude, math.sin(particle_angle) * velocity_magnitude])
                
                p_pos = pos + np.array([random.gauss(0, 1.5), random.gauss(0, 1.5)])
                p_life = random.randint(20, 40)
                p_size = random.uniform(1.0, 2.2) * organic_scale
                p_color = (200, 220, 255)
                p_temp = random.uniform(1800, 2800)

                particle = self._get_or_create_particle(p_pos, vel, p_life, p_size, p_color, 'elegant_exhaust', temperature=p_temp)
                self.add_particle(particle) # Changed from self.particles.append(particle)

    def add_particle(self, particle_to_add: Particle):
        if len(self.particles) >= self.max_particles:
            # Remove the oldest particle and add it to the pool
            oldest_particle = self.particles.pop(0)
            oldest_particle.active = False # Mark as inactive before pooling
            self.particle_pool.append(oldest_particle)
        self.particles.append(particle_to_add)


    def update(self):
        # Iterate backwards for safe removal or moving to pool
        i = len(self.particles) - 1
        while i >= 0:
            particle = self.particles[i]
            if not particle.update(): # particle.update() now returns False if dead and sets self.active = False
                # Move dead particle to pool
                self.particle_pool.append(self.particles.pop(i))
            i -= 1
    
    def render(self, surface):
        for p in self.particles:
            color, alpha = p.get_render_color_and_alpha()
            if alpha == 0: continue

            pos_int = p.pos.astype(int)
            size_int = max(1, int(p.size))

            if p.particle_type == 'exhaust' or p.particle_type == 'energy' or p.particle_type == 'elegant_exhaust': # Added elegant_exhaust
                # Use a temporary surface for alpha blending and glow
                temp_surf_size = size_int * 4
                if temp_surf_size == 0: continue
                temp_surf = pygame.Surface((temp_surf_size, temp_surf_size), pygame.SRCALPHA)
                
                # Draw particle center
                pygame.draw.circle(temp_surf, (*color, alpha),
                                 (temp_surf_size // 2, temp_surf_size // 2), size_int)
                
                # Simple glow for exhaust/energy
                if p.particle_type == 'exhaust' or p.particle_type == 'elegant_exhaust': # Added elegant_exhaust
                    glow_alpha = int(alpha * 0.3)
                    if glow_alpha > 10:
                         pygame.draw.circle(temp_surf, (*color, glow_alpha),
                                         (temp_surf_size // 2, temp_surf_size // 2), size_int + 2)
                
                surface.blit(temp_surf, (pos_int[0] - temp_surf_size // 2, pos_int[1] - temp_surf_size // 2), special_flags=pygame.BLEND_RGBA_ADD)

            elif p.particle_type == 'spark':
                # Sparks are solid, bright
                pygame.draw.circle(surface, color, pos_int, size_int)
            else: # Default rendering
                pygame.draw.circle(surface, (*color, alpha), pos_int, size_int)

# The _render_glowing_particle and _render_soft_particle methods from the prompt
# were integrated into the main render loop with simplifications.
# The prompt's version of these created new surfaces per particle per glow layer,
# which can be very slow. BLEND_ADD is a good way to achieve glow.