import pygame
import numpy as np
import math
import random # Added for Starfield
from config import *

class VisualEffects:
    def __init__(self):
        self.glow_cache = {}  # Cache for glow surfaces
        
    def draw_mining_beam_advanced(self, surface, start_pos, end_pos, intensity=1.0, 
                                 pulse_phase=0):
        """Draw advanced mining laser with multiple layers"""
        start_pos_np = np.array(start_pos)
        end_pos_np = np.array(end_pos)

        if np.array_equal(start_pos_np, end_pos_np):
            return
            
        # Calculate beam properties
        beam_vector = end_pos_np - start_pos_np
        beam_length = np.linalg.norm(beam_vector)
        if beam_length == 0: return # Avoid division by zero
        beam_angle_rad = math.atan2(beam_vector[1], beam_vector[0])
        beam_angle_deg = math.degrees(-beam_angle_rad) # Pygame rotates counter-clockwise
        
        # Pulsing intensity
        pulse = (math.sin(pulse_phase * 0.1) + 1) / 2 # pulse_phase is time in ticks
        current_intensity = intensity * (0.7 + 0.3 * pulse)
        
        # Multiple beam layers for depth
        beam_layers = [
            {'width': int(max(1, 6 * current_intensity)), 'color': (0, 255, 150), 'alpha': int(200 * current_intensity)}, # Core beam
            {'width': int(max(1, 10 * current_intensity)), 'color': (100, 255, 200), 'alpha': int(100 * current_intensity)}, # Inner glow
            {'width': int(max(1, 16 * current_intensity)), 'color': (150, 255, 220), 'alpha': int(50 * current_intensity)}  # Outer glow
        ]
        
        for layer in beam_layers:
            if layer['alpha'] > 10 and layer['width'] > 0 and beam_length > 0:
                # Create beam surface for alpha blending
                # Surface needs to be large enough to contain the rotated beam.
                # Max dimension for surface is beam_length + layer['width']
                surf_width = int(beam_length)
                surf_height = layer['width']
                if surf_width <=0 or surf_height <=0: continue

                beam_surf = pygame.Surface((surf_width, surf_height), pygame.SRCALPHA)
                beam_color_with_alpha = (*layer['color'], layer['alpha'])
                
                # Draw beam on surface (horizontal line)
                pygame.draw.line(beam_surf, beam_color_with_alpha, 
                               (0, surf_height // 2), 
                               (surf_width, surf_height // 2), 
                               layer['width'])
                
                rotated_beam = pygame.transform.rotate(beam_surf, beam_angle_deg)
                
                rect = rotated_beam.get_rect(center = (start_pos_np + beam_vector / 2).astype(int) )
                surface.blit(rotated_beam, rect, special_flags=pygame.BLEND_RGBA_ADD)
        
        self.draw_impact_glow(surface, end_pos_np, current_intensity * 15, (0, 255, 150))
    
    def draw_impact_glow(self, surface, pos, radius, color):
        if radius <= 1: # Min radius for visibility
            return
            
        for i in range(3): # Number of glow layers
            layer_radius = radius * (1.0 - i * 0.25) # Decrease radius for inner layers
            layer_alpha = int(120 * (1.0 - i * 0.3)) # Decrease alpha for inner layers
            
            if layer_radius > 1 and layer_alpha > 10:
                # Optimized: Use one surface for all layers if possible, or draw directly if opaque
                glow_color_with_alpha = (*color, layer_alpha)
                
                # Create a surface for each layer for smooth alpha blending
                temp_surface_size = int(layer_radius * 2)
                if temp_surface_size <=0: continue
                glow_surf = pygame.Surface((temp_surface_size, temp_surface_size), pygame.SRCALPHA)
                
                pygame.draw.circle(glow_surf, glow_color_with_alpha, 
                                 (int(layer_radius), int(layer_radius)), 
                                 int(layer_radius))
                
                surface.blit(glow_surf, 
                           (int(pos[0] - layer_radius), int(pos[1] - layer_radius)),
                           special_flags=pygame.BLEND_RGBA_ADD) # Use RGBA_ADD for additive glow
    
    def draw_communication_beam(self, surface, start_pos, end_pos, intensity=1.0):
        start_pos_np = np.array(start_pos)
        end_pos_np = np.array(end_pos)
        if np.array_equal(start_pos_np, end_pos_np): return
            
        time_ms = pygame.time.get_ticks()
        packet_speed_factor = 0.002  # Adjusted speed
        
        direction_vec = end_pos_np - start_pos_np
        distance = np.linalg.norm(direction_vec)
        if distance < 1e-3: return
            
        norm_direction = direction_vec / distance
        
        # Draw base beam (very faint, static)
        beam_color_static = (100, 150, 255, 20) # Very transparent
        pygame.draw.line(surface, beam_color_static, start_pos_np.astype(int), end_pos_np.astype(int), 1)
        
        # Animated data packets
        num_packets = 3
        for i in range(num_packets):
            # Stagger packets along the beam, make them loop
            packet_phase = (time_ms * packet_speed_factor + (i / float(num_packets))) % 1.0
            current_packet_pos = start_pos_np + norm_direction * distance * packet_phase
            
            self.draw_data_packet(surface, current_packet_pos, intensity)
    
    def draw_data_packet(self, surface, pos, intensity):
        base_packet_size = 2 + intensity * 2 # Scale size with intensity
        packet_colors_alphas = [
            ((180, 220, 255), int(220 * intensity)), # Core
            ((120, 180, 255), int(150 * intensity)), # Mid glow
            ((80, 120, 200), int(80 * intensity))    # Outer glow
        ]
        
        for i, (color, alpha) in enumerate(packet_colors_alphas):
            current_size = base_packet_size * (1.0 - i * 0.3)
            if current_size >= 1 and alpha > 10:
                # Draw directly with alpha if surface supports it, or use temp surface
                # For simplicity and performance, draw directly if surface is main screen
                final_color = (*color, alpha)
                # pygame.draw.circle(surface, final_color, pos.astype(int), int(current_size)) # This won't blend well
                
                # Use a small temp surface for each layer for proper blending
                temp_surf_size = int(current_size * 2)
                if temp_surf_size <= 0: continue
                temp_surf = pygame.Surface((temp_surf_size, temp_surf_size), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, final_color, (int(current_size), int(current_size)), int(current_size))
                surface.blit(temp_surf, (int(pos[0]-current_size), int(pos[1]-current_size)), special_flags=pygame.BLEND_RGBA_ADD)


    def draw_energy_field_distortion(self, surface, center_pos, intensity_factor, radius=50):
        if intensity_factor < 0.1: # Min intensity to show effect
            return
            
        time_ms = pygame.time.get_ticks()
        distortion_count = int(intensity_factor * 6) + 2 # Min 2 distortions
        
        for i in range(distortion_count):
            # Angle determines position on circle, also slowly rotates
            angle = (i * 2 * math.pi / distortion_count) + (time_ms * 0.0005 * (i%2 * 2 -1)) # Slow rotation, alternating direction
            
            # distance_factor makes them appear to pulse in and out
            distance_factor = 0.6 + 0.4 * math.sin(time_ms * 0.0015 + i * math.pi/3)
            
            current_radius = radius * distance_factor
            distortion_pos = center_pos + np.array([
                math.cos(angle) * current_radius,
                math.sin(angle) * current_radius
            ])
            
            distortion_alpha = int(40 * intensity_factor * ((0.5 + distance_factor*0.5))) # Alpha also pulses
            if distortion_alpha > 5:
                distortion_color_with_alpha = (50, 120, 220, distortion_alpha) # Cool blue
                distortion_size = 1.5 + intensity_factor * 2.0
                
                # Draw directly for this subtle effect, BLEND_ADD will make it glowish
                # A small temp surface for each particle is better for smooth alpha
                temp_surf_size = int(distortion_size * 2)
                if temp_surf_size <=0: continue
                temp_surf = pygame.Surface((temp_surf_size, temp_surf_size), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, distortion_color_with_alpha, 
                                 (int(distortion_size), int(distortion_size)),
                                 int(distortion_size))
                surface.blit(temp_surf, (int(distortion_pos[0]-distortion_size), int(distortion_pos[1]-distortion_size)),
                             special_flags=pygame.BLEND_RGBA_ADD)

    def draw_resource_glow(self, surface, pos, radius, intensity_factor):
            """Draw a pulsating glow effect for resources."""
            if radius <= 1 or intensity_factor < 0.05:
                return

            time_ms = pygame.time.get_ticks()
            # Pulsating effect for radius and alpha
            pulse = (math.sin(time_ms * 0.002 + pos[0] * 0.1) + 1) / 2 # 0 to 1, vary phase by pos
            current_radius_factor = 0.8 + 0.2 * pulse
            current_alpha_factor = 0.5 + 0.5 * pulse * intensity_factor

            base_color = (0, 200, 50) # Greenish glow for resources

            for i in range(2): # Two layers for a softer glow
                layer_radius = radius * current_radius_factor * (1.0 - i * 0.3)
                layer_alpha = int(100 * current_alpha_factor * (1.0 - i * 0.4))

                if layer_radius > 1 and layer_alpha > 10:
                    glow_color_with_alpha = (*base_color, layer_alpha)
                    
                    temp_surface_size = int(layer_radius * 2)
                    if temp_surface_size <= 0: continue
                    
                    glow_surf = pygame.Surface((temp_surface_size, temp_surface_size), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surf, glow_color_with_alpha,
                                     (int(layer_radius), int(layer_radius)),
                                     int(layer_radius))
                    
                    surface.blit(glow_surf,
                               (int(pos[0] - layer_radius), int(pos[1] - layer_radius)),
                               special_flags=pygame.BLEND_RGBA_ADD)
class StarField:
    def __init__(self, width, height, star_count=200):
        self.width = width
        self.height = height
        
        self.star_layers = [
            {'stars': [], 'speed_factor': 0.05, 'base_brightness': 0.4, 'size': 1, 'count': int(star_count * 0.5)},  # Farthest
            {'stars': [], 'speed_factor': 0.15, 'base_brightness': 0.7, 'size': 1, 'count': int(star_count * 0.3)}, # Mid
            {'stars': [], 'speed_factor': 0.30, 'base_brightness': 1.0, 'size': 2, 'count': int(star_count * 0.2)},  # Near
        ]
        
        for layer in self.star_layers:
            for _ in range(layer['count']):
                star = {
                    'pos': np.array([random.uniform(0, width), random.uniform(0, height)]),
                    'brightness_mod': random.uniform(0.5, 1.0), # Individual star brightness variation
                    'twinkle_phase': random.uniform(0, 2 * math.pi),
                    'twinkle_speed': random.uniform(0.002, 0.008) # Slower twinkle
                }
                layer['stars'].append(star)
    
    def draw(self, surface, camera_offset_world=np.array([0.0, 0.0])):
        time_ms = pygame.time.get_ticks()
        
        for layer in self.star_layers:
            # Parallax: camera_offset_world is how much the world view has shifted.
            # Stars shift less based on their layer's speed_factor.
            parallax_shift = camera_offset_world * layer['speed_factor']
            
            for star in layer['stars']:
                # Star's base position + parallax shift, then wrap around screen
                screen_pos_x = (star['pos'][0] - parallax_shift[0]) % self.width
                screen_pos_y = (star['pos'][1] - parallax_shift[1]) % self.height
                screen_pos_int = (int(screen_pos_x), int(screen_pos_y))
                
                # Twinkling effect
                twinkle_val = (math.sin(time_ms * star['twinkle_speed'] + star['twinkle_phase']) + 1) / 2 # 0 to 1
                current_brightness = layer['base_brightness'] * star['brightness_mod'] * (0.6 + 0.4 * twinkle_val)
                
                # Star color (mostly white, slight blue tint for some)
                blue_tint_factor = random.uniform(0.8, 1.0) if random.random() < 0.1 else 1.0
                star_rgb_val = int(255 * current_brightness)
                star_color = (
                    min(255, star_rgb_val),
                    min(255, star_rgb_val),  
                    min(255, int(star_rgb_val * blue_tint_factor * 1.05)) # Slight blue boost
                )
                
                if star_rgb_val < 20: continue # Skip very dim stars

                if layer['size'] <= 1:
                    surface.set_at(screen_pos_int, star_color)
                else:
                    # For larger stars, draw a small circle, maybe with a bit of alpha if possible
                    # Direct drawing for performance, assuming main screen surface
                    pygame.draw.circle(surface, star_color, screen_pos_int, layer['size'])