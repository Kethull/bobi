import pygame
import numpy as np
import math
from config import config

class OrganicShipRenderer:
    def __init__(self):
        self.ship_colors = {
            # Primary hull colors with metallic tones
            'hull_primary': (85, 95, 110),      # Cool metallic blue-gray
            'hull_secondary': (105, 115, 130),   # Lighter metallic
            'hull_accent': (125, 135, 150),     # Highlight metallic
            'hull_shadow': (65, 75, 90),        # Shadow areas
            
            # Engine and technical elements
            'engine_core': (150, 180, 255),     # Bright blue engine glow
            'engine_ring': (100, 130, 200),     # Engine accent
            'thruster_glow': (200, 220, 255),   # Thruster output
            
            # Panel and detail colors
            'panel_lines': (45, 55, 70),        # Subtle panel definition
            'window_glow': (180, 200, 255),     # Cockpit/sensor windows
            'status_lights': (0, 255, 150),     # Status indicators
            
            # Energy states
            'energy_high': (120, 255, 200),     # High energy glow
            'energy_medium': (255, 200, 100),   # Medium energy
            'energy_low': (255, 100, 100),      # Low energy warning
        }
        
        # Ship proportions (scaled to SPACESHIP_SIZE)
        self.ship_length = config.Visualization.PROBE_SIZE_PX * 2.5   # Much more elongated
        self.ship_width = config.Visualization.PROBE_SIZE_PX * 0.8
        self.engine_length = config.Visualization.PROBE_SIZE_PX * 0.6
        
    def draw_organic_ship(self, surface, probe, screen_pos, scale=1.0):
        """Draw elegant organic spacecraft"""
        angle = probe.get('angle', 0)
        energy_ratio = probe['energy'] / config.Probe.MAX_ENERGY
        is_low_power = probe['energy'] <= 0
        
        # Scale dimensions
        length = self.ship_length * scale
        width = self.ship_width * scale
        
        # Drawing order: back to front
        self._draw_engine_section(surface, screen_pos, angle, energy_ratio, scale)
        self._draw_main_hull(surface, screen_pos, angle, energy_ratio, scale)
        self._draw_forward_section(surface, screen_pos, angle, energy_ratio, scale)
        self._draw_surface_details(surface, screen_pos, angle, scale)
        self._draw_energy_indicators(surface, screen_pos, angle, energy_ratio, is_low_power, scale)
        
        return screen_pos
    
    def _draw_main_hull(self, surface, pos, angle, energy_ratio, scale):
        """Draw the main elongated hull with organic curves"""
        length = self.ship_length * scale
        width = self.ship_width * scale
        
        # Create smooth hull outline using multiple curve segments
        hull_points = self._generate_hull_curve(pos, angle, length, width)
        
        # Color based on energy state
        if energy_ratio > 0.7:
            hull_color = self.ship_colors['hull_primary']
        elif energy_ratio > 0.3:
            hull_color = tuple(int(c * 0.85) for c in self.ship_colors['hull_primary'])
        else:
            hull_color = tuple(int(c * 0.7) for c in self.ship_colors['hull_primary'])
        
        # Draw main hull
        if len(hull_points) > 2:
            pygame.draw.polygon(surface, hull_color, hull_points)
            
            # Add metallic sheen effect
            self._add_metallic_sheen(surface, hull_points, angle)
            
            # Hull outline
            pygame.draw.polygon(surface, self.ship_colors['hull_accent'], hull_points, 1)
    
    def _generate_hull_curve(self, center, angle, length, width):
        """Generate smooth curved hull outline"""
        points = []
        num_segments = 20  # Higher for smoother curves
        
        # Top curve (starboard side)
        for i in range(num_segments + 1):
            t = i / num_segments
            
            # Parametric curve for organic shape
            # Forward section tapers to a point
            if t < 0.7:  # Main body
                x = (t - 0.5) * length
                y = -width * 0.5 * math.sin(math.pi * t) * (1 - t * 0.3)
            else:  # Forward taper
                progress = (t - 0.7) / 0.3
                x = (t - 0.5) * length
                y = -width * 0.5 * (1 - progress) * math.sin(math.pi * 0.7) * (1 - 0.7 * 0.3)
            
            # Apply rotation and translation
            rotated_point = self._rotate_point(np.array([x, y]), angle)
            points.append(center + rotated_point)
        
        # Bottom curve (port side) - mirror of top
        for i in range(num_segments, -1, -1):
            t = i / num_segments
            
            if t < 0.7:
                x = (t - 0.5) * length
                y = width * 0.5 * math.sin(math.pi * t) * (1 - t * 0.3)
            else:
                progress = (t - 0.7) / 0.3
                x = (t - 0.5) * length
                y = width * 0.5 * (1 - progress) * math.sin(math.pi * 0.7) * (1 - 0.7 * 0.3)
            
            rotated_point = self._rotate_point(np.array([x, y]), angle)
            points.append(center + rotated_point)
        
        return points
    
    def _draw_engine_section(self, surface, pos, angle, energy_ratio, scale):
        """Draw organic engine section with glowing elements"""
        engine_length = self.engine_length * scale
        engine_width = self.ship_width * scale * 0.9
        
        # Engine housing - curved trapezoid shape
        engine_points = []
        
        # Create engine outline
        engine_coords = [
            # Rear (wide end)
            (-self.ship_length * 0.5 * scale, -engine_width * 0.4),
            (-self.ship_length * 0.5 * scale, engine_width * 0.4),
            # Transition to main hull
            (-self.ship_length * 0.2 * scale, engine_width * 0.45),
            (-self.ship_length * 0.2 * scale, -engine_width * 0.45),
        ]
        
        for coord in engine_coords:
            rotated_point = self._rotate_point(np.array(coord), angle)
            engine_points.append(pos + rotated_point)
        
        # Draw engine housing
        engine_color = tuple(int(c * 0.8) for c in self.ship_colors['hull_secondary'])
        pygame.draw.polygon(surface, engine_color, engine_points)
        
        # Engine glow effects
        if energy_ratio > 0.1:
            self._draw_engine_glow(surface, pos, angle, energy_ratio, scale)
        
        # Engine details
        self._draw_engine_details(surface, pos, angle, scale)
    
    def _draw_engine_glow(self, surface, pos, angle, energy_ratio, scale):
        """Draw glowing engine effects"""
        # Engine center position
        engine_center = pos + self._rotate_point(
            np.array([-self.ship_length * 0.5 * scale, 0]), angle
        )
        
        # Main engine glow
        glow_intensity = energy_ratio
        glow_radius = int(self.ship_width * 0.3 * scale * glow_intensity)
        
        if glow_radius > 1:
            # Multiple glow layers for depth
            glow_colors = [
                (*self.ship_colors['engine_core'], int(150 * glow_intensity)),
                (*self.ship_colors['engine_ring'], int(100 * glow_intensity)),
                (*self.ship_colors['thruster_glow'], int(50 * glow_intensity))
            ]
            
            for i, (r, g, b, a) in enumerate(glow_colors):
                if a > 10:
                    layer_radius = glow_radius + i * 3
                    glow_surf = pygame.Surface((layer_radius*2, layer_radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surf, (r, g, b, a), 
                                     (layer_radius, layer_radius), layer_radius)
                    
                    surface.blit(glow_surf, 
                               (engine_center[0] - layer_radius, 
                                engine_center[1] - layer_radius),
                               special_flags=pygame.BLEND_ADD)
    
    def _draw_forward_section(self, surface, pos, angle, energy_ratio, scale):
        """Draw elegant forward section with sensor arrays"""
        # Forward section is integrated into main hull curve
        # Add forward sensor cluster
        forward_tip = pos + self._rotate_point(
            np.array([self.ship_length * 0.5 * scale, 0]), angle
        )
        
        # Sensor array at the tip
        sensor_size = max(2, int(3 * scale))
        sensor_color = self.ship_colors['window_glow'] if energy_ratio > 0.3 else self.ship_colors['hull_shadow']
        
        pygame.draw.circle(surface, sensor_color, forward_tip.astype(int), sensor_size)
        
        # Navigation lights
        nav_light_positions = [
            np.array([self.ship_length * 0.4 * scale, -self.ship_width * 0.2 * scale]),
            np.array([self.ship_length * 0.4 * scale, self.ship_width * 0.2 * scale])
        ]
        
        for nav_pos in nav_light_positions:
            light_pos = pos + self._rotate_point(nav_pos, angle)
            light_color = self.ship_colors['status_lights'] if energy_ratio > 0.2 else self.ship_colors['hull_shadow']
            pygame.draw.circle(surface, light_color, light_pos.astype(int), max(1, int(2 * scale)))
    
    def _draw_surface_details(self, surface, pos, angle, scale):
        """Draw subtle surface details and panel lines"""
        # Flowing panel lines that follow the ship's curves
        detail_color = self.ship_colors['panel_lines']
        
        # Longitudinal flow lines
        for y_offset in [-0.15, 0, 0.15]:
            line_points = []
            for i in range(15):
                t = i / 14.0
                x = (t - 0.5) * self.ship_length * scale * 0.8
                y = y_offset * self.ship_width * scale
                
                # Add subtle curve to follow hull shape
                y *= math.sin(math.pi * t) * (1 - t * 0.2) if t < 0.7 else (1 - (t - 0.7) / 0.3)
                
                local_point = np.array([x, y])
                world_point = pos + self._rotate_point(local_point, angle)
                line_points.append(world_point)
            
            if len(line_points) > 1:
                pygame.draw.lines(surface, detail_color, False, line_points, 1)
        
        # Subtle geometric accents
        self._draw_hull_accents(surface, pos, angle, scale)
    
    def _draw_hull_accents(self, surface, pos, angle, scale):
        """Draw subtle geometric accent lines"""
        accent_color = self.ship_colors['hull_accent']
        
        # Side accent lines
        accent_positions = [
            # Port side accent
            [np.array([self.ship_length * 0.1 * scale, self.ship_width * 0.25 * scale]),
             np.array([self.ship_length * 0.3 * scale, self.ship_width * 0.15 * scale])],
            # Starboard side accent  
            [np.array([self.ship_length * 0.1 * scale, -self.ship_width * 0.25 * scale]),
             np.array([self.ship_length * 0.3 * scale, -self.ship_width * 0.15 * scale])]
        ]
        
        for accent_line in accent_positions:
            world_points = [pos + self._rotate_point(point, angle) for point in accent_line]
            pygame.draw.line(surface, accent_color, world_points[0], world_points[1], 1)
    
    def _draw_energy_indicators(self, surface, pos, angle, energy_ratio, is_low_power, scale):
        """Draw energy status indicators integrated into hull"""
        # Status indicator positions along the hull
        indicator_positions = [
            np.array([self.ship_length * 0.2 * scale, 0]),
            np.array([0, -self.ship_width * 0.3 * scale]),
            np.array([0, self.ship_width * 0.3 * scale]),
        ]
        
        # Choose indicator color based on energy state
        if is_low_power:
            indicator_color = self.ship_colors['energy_low']
        elif energy_ratio > 0.7:
            indicator_color = self.ship_colors['energy_high']
        else:
            indicator_color = self.ship_colors['energy_medium']
        
        # Pulsing effect
        pulse = (math.sin(pygame.time.get_ticks() * 0.008) + 1) / 2
        pulse_alpha = int(255 * (0.4 + 0.6 * pulse))
        pulsing_color = (*indicator_color[:3], pulse_alpha)
        
        for indicator_pos in indicator_positions:
            world_pos = pos + self._rotate_point(indicator_pos, angle)
            indicator_size = max(1, int(2 * scale))
            
            # Draw pulsing indicator with glow
            self._draw_glowing_point(surface, world_pos, pulsing_color, indicator_size)
    
    def _add_metallic_sheen(self, surface, hull_points, angle):
        """Add metallic sheen effect to hull"""
        if len(hull_points) < 3:
            return
            
        # Calculate "light direction" for sheen effect
        light_angle = angle + math.pi * 0.25  # Light comes from upper-left relative to ship
        
        # Create highlight line along the hull
        # Find topmost points for highlight
        hull_array = np.array(hull_points)
        center_y = np.mean(hull_array[:, 1])
        
        highlight_points = []
        for point in hull_points:
            if point[1] < center_y:  # Upper half of hull
                highlight_points.append(point)
        
        if len(highlight_points) > 1:
            # Draw subtle highlight line
            highlight_color = self.ship_colors['hull_accent']
            pygame.draw.lines(surface, highlight_color, False, highlight_points, 1)
    
    def _draw_glowing_point(self, surface, pos, color, size):
        """Draw a glowing point with soft falloff"""
        # Main point
        pygame.draw.circle(surface, color[:3], pos.astype(int), size)
        
        # Glow effect
        if len(color) > 3 and color[3] > 20:  # Has alpha and is visible
            glow_surf = pygame.Surface((size*6, size*6), pygame.SRCALPHA)
            glow_color = (*color[:3], color[3] // 3)
            pygame.draw.circle(glow_surf, glow_color, (size*3, size*3), size*3)
            
            surface.blit(glow_surf, 
                        (pos[0] - size*3, pos[1] - size*3),
                        special_flags=pygame.BLEND_ADD)
    
    def _draw_engine_details(self, surface, pos, angle, scale):
        """Draw detailed engine components"""
        engine_center = pos + self._rotate_point(
            np.array([-self.ship_length * 0.35 * scale, 0]), angle
        )
        
        # Engine ring details
        ring_radius = int(self.ship_width * 0.25 * scale)
        if ring_radius > 2:
            pygame.draw.circle(surface, self.ship_colors['engine_ring'], 
                             engine_center.astype(int), ring_radius, 1)
            pygame.draw.circle(surface, self.ship_colors['hull_accent'], 
                             engine_center.astype(int), ring_radius - 2, 1)
    
    def _rotate_point(self, point, angle):
        """Rotate a 2D point by given angle"""
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return np.array([
            point[0] * cos_a - point[1] * sin_a,
            point[0] * sin_a + point[1] * cos_a
        ])

    def draw_enhanced_thruster_effects(self, surface, pos, angle, thrust_power, thrust_ramp):
        """Draw organic thruster effects that match the ship design"""
        if thrust_power <= 0 or thrust_ramp <= 0.01:
            return
            
        # Thruster position at rear of ship
        thruster_pos = pos + self._rotate_point(
            np.array([-self.ship_length * 0.5, 0]), angle
        )
        
        # Organic flame shape - more flowing than angular
        flame_length = self.ship_length * 0.4 * thrust_ramp
        flame_width = self.ship_width * 0.6 * thrust_ramp
        
        # Create organic flame outline
        flame_points = []
        num_points = 12
        
        for i in range(num_points):
            t = i / (num_points - 1)
            
            if i < num_points // 2:  # Top side
                x = -flame_length * (t ** 0.7)  # Curved taper
                y = -flame_width * 0.5 * math.sin(math.pi * t * 0.8)
            else:  # Bottom side  
                mirror_t = 1 - (i - num_points // 2) / (num_points // 2)
                x = -flame_length * (mirror_t ** 0.7)
                y = flame_width * 0.5 * math.sin(math.pi * mirror_t * 0.8)
            
            local_point = np.array([x, y])
            world_point = thruster_pos + self._rotate_point(local_point, angle)
            flame_points.append(world_point)
        
        # Draw flame with gradient effect
        if len(flame_points) > 2:
            # Outer flame (cooler)
            outer_color = (255, 150, 50, int(100 * thrust_ramp))
            pygame.draw.polygon(surface, outer_color[:3], flame_points)
            
            # Inner flame (hotter) - smaller
            inner_points = []
            for point in flame_points:
                direction_to_center = thruster_pos - point
                inner_point = point + direction_to_center * 0.3
                inner_points.append(inner_point)
            
            if len(inner_points) > 2:
                inner_color = (200, 220, 255)
                pygame.draw.polygon(surface, inner_color, inner_points)