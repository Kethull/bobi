import pygame
import numpy as np
import math
import random
from config import * # Assuming config.py might have relevant constants later

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
        
    def update(self):
        """Update particle physics"""
        self.pos += self.vel
        self.life -= 1
        
        if self.particle_type == 'exhaust':
            self.vel *= 0.98  # Exhaust slows down due to expansion/drag
            self.temperature *= 0.92  # Cools over time
            self.size *= 0.99  # Shrinks slightly as it disperses
            if self.size < 0.5: self.life = 0 # Fade out if too small
        elif self.particle_type == 'spark':
            # self.vel[1] += 0.05  # Simple gravity, adjust as needed if sparks are heavy
            self.vel *= 0.95 # Sparks lose energy
            self.size *= 0.97
            if self.size < 0.5: self.life = 0
        elif self.particle_type == 'energy':
            # Energy particles might maintain velocity or have specific behaviors
            self.size *= 0.98 # Energy particles might dissipate
            if self.size < 0.3: self.life = 0
        
        return self.life > 0
    
    def get_render_color_and_alpha(self): # Renamed for clarity
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
        self.particles = []
        self.max_particles = 1000 # Default, can be overridden
        
    def emit_thruster_exhaust(self, pos, angle, thrust_power_level, thrust_ramp_ratio=1.0):
        """Create realistic thruster exhaust plume.
        thrust_power_level: 0-3 (index from THRUST_FORCE)
        thrust_ramp_ratio: 0.0-1.0 (how much of target thrust is active)
        """
        if thrust_power_level <= 0 or thrust_ramp_ratio <= 0.01:
            return
            
        exhaust_angle = angle + math.pi # Exhaust goes opposite to ship's angle
        
        # Intensity based on power level and ramp
        # THRUST_FORCE = [0.0, 0.08, 0.18, 0.32]
        # Max actual force is THRUST_FORCE[3]
        base_intensity_factor = (thrust_power_level / (len(THRUST_FORCE) -1 if len(THRUST_FORCE) > 1 else 1)) * thrust_ramp_ratio
        
        particle_count = int(base_intensity_factor * 20) + 2 # Min 2 particles
        base_velocity_magnitude = base_intensity_factor * 4.0 + 1.0 # Base speed of particles
        
        for _ in range(particle_count):
            spread = math.radians(random.uniform(-20, 20) * (1.0 - base_intensity_factor * 0.5)) # Narrower cone at high thrust
            p_angle = exhaust_angle + spread
            
            p_vel_mag = base_velocity_magnitude * random.uniform(0.7, 1.3)
            vel = np.array([math.cos(p_angle) * p_vel_mag, math.sin(p_angle) * p_vel_mag])
            
            # Offset particles slightly from emission point to avoid clustering at ship center
            offset_distance = SPACESHIP_SIZE * 0.5 # Approx half ship size
            start_pos_offset = np.array([math.cos(angle) * offset_distance, math.sin(angle) * offset_distance])
            
            particle = Particle(
                pos=pos - start_pos_offset, # Emit from "behind" the center based on ship angle
                vel=vel,
                life=random.randint(20, 45),
                size=random.uniform(1.0, 3.0) * base_intensity_factor + 0.5,
                color=(255,180,100), # Base color, will be modified by temperature
                particle_type='exhaust'
            )
            particle.temperature = random.uniform(2200, 3800) * base_intensity_factor
            self.add_particle(particle)
        
    def emit_mining_sparks(self, impact_pos, surface_normal_angle=None, intensity=1.0):
        spark_count = int(intensity * 10) + 3
        
        for _ in range(spark_count):
            if surface_normal_angle is not None:
                # Sparks reflect off surface
                base_angle = surface_normal_angle + math.pi # Opposite to normal
                angle_spread = math.radians(random.uniform(-60, 60))
                p_angle = base_angle + angle_spread
            else:
                # Sparks radiate if no surface normal
                p_angle = random.uniform(0, 2 * math.pi)
                
            speed = random.uniform(1.5, 5.0) * intensity
            vel = np.array([math.cos(p_angle) * speed, math.sin(p_angle) * speed])
            
            colors = [(255,255,150), (255,200,100), (255,180,50)]
            particle = Particle(
                pos=impact_pos + np.array([random.gauss(0,1), random.gauss(0,1)]),
                vel=vel,
                life=random.randint(15, 30),
                size=random.uniform(0.8, 2.2),
                color=random.choice(colors),
                particle_type='spark'
            )
            self.add_particle(particle)
            
    def emit_communication_pulse(self, start_pos, end_pos, pulse_speed=4.0):
        direction_vec = np.array(end_pos) - np.array(start_pos)
        distance = np.linalg.norm(direction_vec)
        if distance < 1e-3: return
        
        norm_direction = direction_vec / distance
        num_segments = max(3, int(distance / 15)) # More particles for longer beams

        for i in range(num_segments):
            # Stagger particle creation for a "traveling pulse" effect if called over multiple frames
            # For a single call, they appear along the line
            pos_offset = norm_direction * (distance * (i / float(num_segments)))
            current_pos = np.array(start_pos) + pos_offset
            
            vel = norm_direction * pulse_speed
            
            particle = Particle(
                pos=current_pos,
                vel=vel,
                life=int(distance / pulse_speed) + 5, # Life depends on travel time
                size=random.uniform(1.5, 2.5),
                color=(100, 180, 255), # Cyan-blue
                particle_type='energy'
            )
            self.add_particle(particle)

    def emit_energy_field(self, center_pos, radius, energy_intensity_factor): # Factor 0-1
        if energy_intensity_factor < 0.1: return
        particle_count = int(energy_intensity_factor * 15)

        for _ in range(particle_count):
            angle = random.uniform(0, 2 * math.pi)
            dist_from_center = radius * random.uniform(0.5, 1.0)
            
            pos_offset = np.array([math.cos(angle) * dist_from_center, math.sin(angle) * dist_from_center])
            
            # Slow orbital/drifting motion
            vel_tangential = np.array([-pos_offset[1], pos_offset[0]]) * 0.005 * energy_intensity_factor
            vel_radial = pos_offset * random.uniform(-0.002, 0.002) # Slight in/out drift
            
            particle = Particle(
                pos=center_pos + pos_offset,
                vel=vel_tangential + vel_radial,
                life=random.randint(40, 80),
                size=random.uniform(0.8, 1.8),
                color=(50, 100, 200), # Deep blue
                particle_type='energy'
            )
            self.add_particle(particle)

    def emit_organic_thruster_exhaust(self, pos, angle, thrust_power, thrust_ramp=1.0):
            """Create elegant exhaust particles that match organic ship design"""
            if thrust_power <= 0:
                return
                
            # More refined particle emission
            particle_count = int(thrust_power * thrust_ramp * 8)  # Fewer, higher quality particles
            
            for _ in range(particle_count):
                # Exhaust flows more smoothly
                spread_angle = random.gauss(0, 0.15)  # Tighter spread for elegance
                particle_angle = angle + math.pi + spread_angle
                
                # More controlled velocity distribution
                base_velocity = thrust_power * 2.5 * thrust_ramp
                velocity_variation = base_velocity * 0.15  # Less chaos
                velocity_magnitude = base_velocity + random.gauss(0, velocity_variation)
                
                vel = np.array([
                    math.cos(particle_angle) * velocity_magnitude,
                    math.sin(particle_angle) * velocity_magnitude
                ])
                
                # Elegant particle properties
                particle = Particle(
                    pos=pos + np.array([random.gauss(0, 1.5), random.gauss(0, 1.5)]),
                    vel=vel,
                    life=random.randint(20, 40),  # Longer life for elegance
                    size=random.uniform(1.0, 2.2) * ORGANIC_EXHAUST_PARTICLE_SCALE, # Apply scale
                    color=(200, 220, 255),  # Cooler, more elegant flame color
                    particle_type='elegant_exhaust'
                )
                particle.temperature = random.uniform(1800, 2800)  # Cooler temperatures
                
                self.particles.append(particle)

    def add_particle(self, particle):
        if len(self.particles) < self.max_particles:
            self.particles.append(particle)
        # Optional: replace oldest if max_particles is reached
        # else:
        #     self.particles.pop(0)
        #     self.particles.append(particle)


    def update(self):
        self.particles = [p for p in self.particles if p.update()]
    
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