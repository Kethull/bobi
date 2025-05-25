import pygame
import numpy as np
import math
from config import *

class DetailedProbeRenderer:
    def __init__(self):
        self.metallic_colors = {
            'hull_primary': (120, 130, 140),
            'hull_secondary': (90, 100, 110),
            'hull_accent': (150, 160, 170),
            'panel_lines': (60, 70, 80),
            'dish_white': (240, 245, 250),
            'dish_rim': (180, 185, 190),
            'solar_active': (20, 40, 80),
            'solar_inactive': (40, 40, 40),
            'status_green': (0, 255, 100),
            'status_yellow': (255, 200, 0),
            'status_red': (255, 50, 50)
        }
    
    def draw_bobiverse_probe(self, surface, probe, screen_pos, scale=1.0):
        """Draw detailed Bobiverse probe with all authentic components"""
        angle = probe.get('angle', 0)
        energy_ratio = probe['energy'] / MAX_ENERGY if MAX_ENERGY > 0 else 0
        is_low_power = probe['energy'] <= 0
        
        # Component rendering order (back to front)
        self._draw_solar_panels(surface, screen_pos, angle, energy_ratio, scale)
        self._draw_main_hull(surface, screen_pos, angle, energy_ratio, scale)
        self._draw_communication_dish(surface, screen_pos, angle, scale)
        self._draw_sensor_arrays(surface, screen_pos, angle, scale)
        self._draw_rcs_thrusters(surface, screen_pos, angle, scale)
        self._draw_status_indicators(surface, screen_pos, angle, energy_ratio, is_low_power, scale)
        
        return screen_pos  # Return for effect positioning
    
    def _draw_main_hull(self, surface, pos, angle, energy_ratio, scale):
        """Draw cylindrical main body with panel details"""
        hull_size = SPACESHIP_SIZE * scale
        
        # Main hull outline (elongated hexagon for cylindrical appearance)
        # This was an 8-sided polygon, let's try to make it look more cylindrical
        # by adjusting radii and y_offsets based on angle to simulate perspective.
        # For simplicity, we'll stick to the provided 8-point approach first.
        
        hull_points_local = []
        # Simplified: using a base shape and rotating. The original prompt's logic for hull_points
        # seemed to try to build perspective directly, which is complex.
        # Let's use a simpler base shape that can be rotated.
        # A stretched octagon might work.
        # Top front, mid front, bottom front, bottom back, mid back, top back
        
        # Re-interpreting the 8-point logic from the prompt for a more consistent shape
        # The prompt's logic for hull_points was a bit convoluted with angle_offset inside the loop.
        # Let's define a base shape and then rotate it.
        
        # Base points for an elongated octagonal cylinder (local coordinates, angle=0 means front is to the right)
        # This is a simplified interpretation. The prompt's original hull logic was complex.
        # Let's use a simpler approach: a main body and a slightly tapered nose.
        
        # Main body (rectangle)
        body_width = hull_size * 0.7
        body_length = hull_size * 1.5
        
        body_rect_points_local = [
            np.array([-body_length * 0.6, -body_width * 0.5]), # Top-left rear
            np.array([ body_length * 0.4, -body_width * 0.5]), # Top-right front
            np.array([ body_length * 0.4,  body_width * 0.5]), # Bottom-right front
            np.array([-body_length * 0.6,  body_width * 0.5]), # Bottom-left rear
        ]
        
        # Nose (triangle)
        nose_length = hull_size * 0.5
        nose_points_local = [
            np.array([body_length * 0.4, -body_width * 0.3]), # Connects to top-right body
            np.array([body_length * 0.4 + nose_length, 0]),   # Nose tip
            np.array([body_length * 0.4,  body_width * 0.3]), # Connects to bottom-right body
        ]

        # Rotate and translate points
        all_hull_points_world = [self._rotate_point(p, angle) + pos for p in body_rect_points_local]
        nose_points_world = [self._rotate_point(p, angle) + pos for p in nose_points_local]

        # Draw hull with gradient effect (simulate lighting)
        base_color = self.metallic_colors['hull_primary']
        if energy_ratio > 0.7:
            hull_color = base_color
        elif energy_ratio > 0.3:
            hull_color = tuple(int(c * 0.85) for c in base_color) # Slightly brighter than 0.8
        else:
            hull_color = tuple(int(c * 0.7) for c in base_color) # Slightly brighter than 0.6
        
        pygame.draw.polygon(surface, hull_color, all_hull_points_world)
        pygame.draw.polygon(surface, self.metallic_colors['hull_accent'], all_hull_points_world, 2)
        
        pygame.draw.polygon(surface, hull_color, nose_points_world) # Nose same color as body
        pygame.draw.polygon(surface, self.metallic_colors['hull_accent'], nose_points_world, 2)

        # Panel line details (simplified for now, original was complex)
        self._draw_hull_panel_lines(surface, pos, angle, hull_size, body_length, body_width)

    def _draw_hull_panel_lines(self, surface, pos, angle, hull_total_size, body_length, body_width):
        """Draw realistic panel lines and details on hull"""
        panel_color = self.metallic_colors['panel_lines']
        
        # Longitudinal panel lines on main body
        for offset_factor in [-0.3, 0, 0.3]: # Relative to body_width
            start_local = np.array([-body_length * 0.6, offset_factor * body_width * 0.5])
            end_local   = np.array([ body_length * 0.4, offset_factor * body_width * 0.5])
            
            start_world = self._rotate_point(start_local, angle) + pos
            end_world   = self._rotate_point(end_local, angle) + pos
            pygame.draw.line(surface, panel_color, start_world, end_world, 1)

        # Circumferential panel lines on main body
        for x_offset_factor in [-0.3, 0.1]: # Relative to body_length
            start_local = np.array([x_offset_factor * body_length, -body_width * 0.5])
            end_local   = np.array([x_offset_factor * body_length,  body_width * 0.5])

            start_world = self._rotate_point(start_local, angle) + pos
            end_world   = self._rotate_point(end_local, angle) + pos
            pygame.draw.line(surface, panel_color, start_world, end_world, 1)

    def _draw_communication_dish(self, surface, pos, angle, scale):
        """Draw large parabolic communication dish - most prominent feature"""
        dish_base_size = SPACESHIP_SIZE * 1.1 * scale # Slightly smaller than prompt for balance
        
        # Dish center position (mounted on top/front of probe)
        # Angle 0 is right, so dish should be pointing "forward" (positive X in local rotated space)
        dish_offset_local = np.array([SPACESHIP_SIZE * 0.5 * scale, 0]) 
        dish_center = pos + self._rotate_point(dish_offset_local, angle)
        
        # Main dish surface (multiple concentric circles for depth)
        dish_colors = [
            self.metallic_colors['dish_white'],
            (220, 225, 230),
            (200, 205, 210),
        ]
        
        for ring_idx, color in enumerate(dish_colors):
            ring_radius = int(dish_base_size * (1.0 - ring_idx * 0.2)) # Adjusted spacing
            if ring_radius > 0:
                pygame.draw.circle(surface, color, dish_center.astype(int), ring_radius)
                if ring_idx < len(dish_colors) -1: # Draw rim for inner circles
                     pygame.draw.circle(surface, self.metallic_colors['dish_rim'], 
                                     dish_center.astype(int), ring_radius, 1)
        
        # Central feed horn
        feed_horn_size = int(dish_base_size * 0.15) # Slightly larger
        pygame.draw.circle(surface, self.metallic_colors['hull_secondary'], 
                          dish_center.astype(int), feed_horn_size)
        pygame.draw.circle(surface, self.metallic_colors['panel_lines'], 
                          dish_center.astype(int), feed_horn_size, 1) # Outline feed horn

        # Dish support struts (simplified)
        # Struts from near base of dish to main hull body
        # Assuming hull center is 'pos'
        for i in range(3): # Three supports
            strut_angle_offset = (i * 2 * math.pi / 3) - (math.pi / 6) # Spread them out
            
            # Connection on dish (approximate rim)
            dish_connection_local = np.array([math.cos(strut_angle_offset) * dish_base_size * 0.8, 
                                              math.sin(strut_angle_offset) * dish_base_size * 0.8])
            dish_connection_world = dish_center + self._rotate_point(dish_connection_local, angle - (math.pi/2)) # Dish points forward

            # Connection on hull (approximate attachment points)
            hull_attach_local = np.array([SPACESHIP_SIZE * 0.1 * scale, 
                                          (i-1) * SPACESHIP_SIZE * 0.2 * scale]) # Spread along y-axis of hull
            hull_connection_world = pos + self._rotate_point(hull_attach_local, angle)
            
            pygame.draw.line(surface, self.metallic_colors['hull_accent'], hull_connection_world, dish_connection_world, int(max(1, 2*scale)))

    def _draw_solar_panels(self, surface, pos, angle, energy_ratio, scale):
        """Draw extending solar panel arrays"""
        panel_length = SPACESHIP_SIZE * 1.6 * scale # Slightly shorter
        panel_width = SPACESHIP_SIZE * 0.5 * scale  # Slightly wider
        
        if energy_ratio > 0.4: # Brighter if more energy
            panel_color = self.metallic_colors['solar_active']
            grid_color = (50, 90, 150) # Brighter grid
        else:
            panel_color = self.metallic_colors['solar_inactive']
            grid_color = (50, 50, 50) # Darker grid
        
        for side in [-1, 1]: # Left and right panels
            # Panel base points (local to probe, before rotation)
            # Assuming probe's "forward" is along positive X after rotation by 'angle'
            # Panels extend perpendicular to the probe's forward direction (along local Y)
            
            # Mount point on hull (offset slightly from center along local X)
            mount_x_local = -SPACESHIP_SIZE * 0.2 * scale 
            
            # Panel corners relative to a central line extending sideways from mount_x_local
            # Start of panel (closer to hull)
            start_y_local = side * (SPACESHIP_SIZE * 0.3 * scale) # Gap from hull center
            end_y_local = side * (SPACESHIP_SIZE * 0.3 * scale + panel_length)
            
            corners_local = [
                np.array([mount_x_local - panel_width/2, start_y_local]),
                np.array([mount_x_local + panel_width/2, start_y_local]),
                np.array([mount_x_local + panel_width/2, end_y_local]),
                np.array([mount_x_local - panel_width/2, end_y_local]),
            ]
            
            world_corners = [pos + self._rotate_point(corner, angle) for corner in corners_local]
            
            pygame.draw.polygon(surface, panel_color, world_corners)
            pygame.draw.polygon(surface, grid_color, world_corners, int(max(1, 2*scale)))
            
            self._draw_solar_cell_grid(surface, world_corners, grid_color)

    def _draw_solar_cell_grid(self, surface, corners, grid_color):
        if len(corners) < 4: return
        
        num_lines_short_axis = 3 # Across panel width
        num_lines_long_axis = 6  # Along panel length

        # Lines along the shorter axis (e.g., from edge [0]-[3] to edge [1]-[2])
        for i in range(1, num_lines_short_axis):
            t = i / float(num_lines_short_axis)
            start = corners[0] * (1-t) + corners[1] * t
            end = corners[3] * (1-t) + corners[2] * t
            pygame.draw.line(surface, grid_color, start.astype(int), end.astype(int), 1)

        # Lines along the longer axis (e.g., from edge [0]-[1] to edge [3]-[2])
        for i in range(1, num_lines_long_axis):
            t = i / float(num_lines_long_axis)
            start = corners[0] * (1-t) + corners[3] * t
            end = corners[1] * (1-t) + corners[2] * t
            pygame.draw.line(surface, grid_color, start.astype(int), end.astype(int), 1)

    def _draw_sensor_arrays(self, surface, pos, angle, scale):
        sensor_color = self.metallic_colors['hull_accent']
        sensor_highlight = (210, 220, 255) # Brighter highlight

        # Forward sensor pod (more prominent)
        fwd_sensor_offset = np.array([SPACESHIP_SIZE * 0.9 * scale, 0]) # Further forward
        fwd_sensor_pos = pos + self._rotate_point(fwd_sensor_offset, angle)
        pygame.draw.circle(surface, sensor_color, fwd_sensor_pos.astype(int), int(max(2, 4*scale)))
        pygame.draw.circle(surface, sensor_highlight, fwd_sensor_pos.astype(int), int(max(1, 2*scale)))

        # Side sensor clusters
        for side in [-1, 1]:
            side_sensor_offset = np.array([SPACESHIP_SIZE * 0.2 * scale, side * SPACESHIP_SIZE * 0.4 * scale])
            side_sensor_pos = pos + self._rotate_point(side_sensor_offset, angle)
            pygame.draw.rect(surface, sensor_color, (*(side_sensor_pos - np.array([2,2])*scale).astype(int), int(4*scale), int(4*scale)),0, int(1*scale))

    def _draw_rcs_thrusters(self, surface, pos, angle, scale):
        rcs_color = self.metallic_colors['hull_secondary']
        rcs_nozzle_color = (70,80,90)

        # More distributed RCS thrusters for realism
        rcs_local_positions = [
            np.array([ SPACESHIP_SIZE*0.6*scale,  SPACESHIP_SIZE*0.3*scale]), # Front-ish right
            np.array([ SPACESHIP_SIZE*0.6*scale, -SPACESHIP_SIZE*0.3*scale]), # Front-ish left
            np.array([-SPACESHIP_SIZE*0.5*scale,  SPACESHIP_SIZE*0.4*scale]), # Rear-ish right
            np.array([-SPACESHIP_SIZE*0.5*scale, -SPACESHIP_SIZE*0.4*scale]), # Rear-ish left
        ]
        
        for rcs_offset_local in rcs_local_positions:
            rcs_pos_world = pos + self._rotate_point(rcs_offset_local, angle)
            pygame.draw.circle(surface, rcs_color, rcs_pos_world.astype(int), int(max(1, 3*scale)))
            pygame.draw.circle(surface, rcs_nozzle_color, rcs_pos_world.astype(int), int(max(1, 1.5*scale)))
            
    def _draw_status_indicators(self, surface, pos, angle, energy_ratio, is_low_power, scale):
        # Determine base color
        if is_low_power:
            base_light_color = self.metallic_colors['status_red']
        elif energy_ratio < 0.3: # Low energy warning
            base_light_color = self.metallic_colors['status_yellow']
        else: # Normal operation
            base_light_color = self.metallic_colors['status_green']

        # Pulsing effect
        pulse_factor = (math.sin(pygame.time.get_ticks() * 0.006) + 1) / 2 # Slightly faster pulse
        final_light_color = tuple(min(255, int(c * (0.5 + 0.5 * pulse_factor))) for c in base_light_color)
        light_size = int(max(1, 2 * scale))

        # Define light positions (local to probe, angle=0 means forward is +X)
        indicator_local_positions = [
            np.array([SPACESHIP_SIZE * 0.7 * scale, 0]),                         # Front center
            np.array([-SPACESHIP_SIZE * 0.4 * scale,  SPACESHIP_SIZE * 0.35 * scale]), # Rear right
            np.array([-SPACESHIP_SIZE * 0.4 * scale, -SPACESHIP_SIZE * 0.35 * scale]), # Rear left
        ]

        for light_offset_local in indicator_local_positions:
            light_pos_world = pos + self._rotate_point(light_offset_local, angle)
            pygame.draw.circle(surface, final_light_color, light_pos_world.astype(int), light_size)
            # Optional: add a dim outline
            pygame.draw.circle(surface, tuple(int(c*0.5) for c in final_light_color), light_pos_world.astype(int), light_size, 1)

    def _rotate_point(self, point, angle):
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return np.array([
            point[0] * cos_a - point[1] * sin_a,
            point[0] * sin_a + point[1] * cos_a
        ])