# visualization.py
import pygame
import numpy as np
from typing import Dict, List
import math
from config import *
from environment import Resource # Added import for Resource
from environment import Message # Added import for Message

class Visualization:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Bobiverse RL Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 16)
        self.probe_visual_cache = {}  # Store interpolation data
        self.selected_probe_id_ui = None # ID of the probe selected in the UI
        self.ui_probe_id_rects = {} # Stores pygame.Rect for probe IDs in UI for click detection
        
        # Colors
        self.colors = {
            'background': (10, 10, 30),
            'resource': (0, 255, 0),
            'probe_base': (100, 150, 255),
            'communication': (255, 255, 0),
            'trail': (80, 80, 150),
            'ui_text': (255, 255, 255),
            'ui_bg': (50, 50, 50)
        }
        
        # Scaling for world to screen
        self.scale_x = (SCREEN_WIDTH - 200) / WORLD_WIDTH
        self.scale_y = SCREEN_HEIGHT / WORLD_HEIGHT
        
        # Trail storage
        self.probe_trails = {}
        
    def world_to_screen(self, world_pos):
        """Convert world coordinates to screen coordinates"""
        return (
            int(world_pos[0] * self.scale_x),
            int(world_pos[1] * self.scale_y)
        )
    
    def render(self, environment, probe_agents: Dict = None):
        """Render the current state of the simulation"""
        self.screen.fill(self.colors['background'])
        
        # Draw resources
        self._draw_resources(environment.resources)
        
        # Draw probes
        self._draw_probes(environment.probes, environment.messages)
        
        # Draw UI
        self._draw_ui(environment, probe_agents)
        
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def _draw_resources(self, resources: List[Resource]): # Added type hint
        """Draw resource nodes"""
        for resource in resources:
            if resource.amount > 0:
                screen_pos = self.world_to_screen(resource.position)
                # Size based on remaining amount
                size = max(3, int(resource.amount / RESOURCE_MAX_AMOUNT * 15))
                pygame.draw.circle(self.screen, self.colors['resource'], 
                                 screen_pos, size)
                
                # Draw amount text
                if resource.amount > 10:
                    text = self.small_font.render(f"{int(resource.amount)}", 
                                                True, (255, 255, 255))
                    self.screen.blit(text, (screen_pos[0] + size + 2, 
                                          screen_pos[1] - 8))
    
    def _draw_probes(self, probes: Dict, messages: List[Message]): # Added type hints
        """Draw probes with trails and communication links"""
        generation_colors = [
            (100, 150, 255),  # Gen 0 - Blue
            (255, 100, 100),  # Gen 1 - Red  
            (100, 255, 100),  # Gen 2 - Green
            (255, 255, 100),  # Gen 3 - Yellow
            (255, 100, 255),  # Gen 4 - Magenta
            (100, 255, 255),  # Gen 5 - Cyan
        ]
        
        # Trails are handled later by _update_smooth_trail, called per probe.
        # Old trail logic removed from here.
        
        # Draw communication links
        # current_time = len(messages)  # Simplified timestamp # Commented out unused variable
        for message in messages[-10:]:  # Show recent messages
            sender_pos = None
            for probe_id, probe in probes.items():
                # Only probes with energy can be senders of new messages
                if probe_id == message.sender_id and probe['energy'] > 0:
                    sender_pos = self.world_to_screen(probe['position'])
                    break
            
            if sender_pos:
                resource_pos = self.world_to_screen(message.position)
                pygame.draw.line(self.screen, self.colors['communication'],
                               sender_pos, resource_pos, 1)
        
        # Draw probes
        for probe_id, probe in probes.items():
            # All probes are drawn, but their appearance changes if in low power mode
            current_world_pos = probe['position']
            screen_pos_raw = self.world_to_screen(current_world_pos) # Raw screen position from physics
            
            # ENHANCED INTERPOLATION with velocity prediction
            screen_pos_display = screen_pos_raw
            if probe_id in self.probe_visual_cache:
                cache = self.probe_visual_cache[probe_id]
                if 'prev_screen_pos' in cache and 'prev_velocity' in cache:
                    # Velocity-based prediction for smoother interpolation
                    prev_screen_pos = cache['prev_screen_pos']
                    # velocity_screen is the change in screen position from last frame to this raw frame
                    velocity_screen = np.array(screen_pos_raw) - prev_screen_pos
                    
                    # Predict where the probe should be based on velocity (extrapolate slightly)
                    # The 1.2 factor means we predict a bit further along the current velocity vector.
                    # This helps to smooth out changes if the underlying physics step is large.
                    predicted_pos = prev_screen_pos + velocity_screen * 1.2
                    
                    # Blend between raw position and predicted position
                    # blend_factor = 0.3 means 30% weight to predicted, 70% to current raw.
                    # This helps to anchor the prediction to the actual current state.
                    blend_factor = 0.3
                    interpolated_pos = (
                        blend_factor * predicted_pos +
                        (1 - blend_factor) * np.array(screen_pos_raw)
                    )
                    screen_pos_display = tuple(interpolated_pos.astype(int))

            # Enhanced cache update with velocity tracking
            if probe_id not in self.probe_visual_cache:
                self.probe_visual_cache[probe_id] = {}

            cache = self.probe_visual_cache[probe_id]
            if 'prev_screen_pos' in cache: # Check if prev_screen_pos exists before calculating velocity
                # Calculate screen velocity based on the change from the last raw screen position
                cache['prev_velocity'] = np.array(screen_pos_raw) - cache['prev_screen_pos']
            else:
                # Initialize prev_velocity if it's the first time or cache was cleared
                cache['prev_velocity'] = np.array([0, 0], dtype=float)

            cache['prev_screen_pos'] = np.array(screen_pos_raw) # Store current raw for next frame

            # Use screen_pos_display for all drawing related to this probe's center
            screen_pos = screen_pos_display

            is_low_power = probe['energy'] <= 0
            
            if is_low_power:
                color = (80, 80, 80) # Dim grey for low power probes
            else:
                generation = min(probe['generation'], len(generation_colors) - 1)
                color = generation_colors[generation]
            
            # Spaceship drawing
            # base_ship_size = max(5, int(max(0, probe['energy']) / MAX_ENERGY * 10) + 5) # General size factor
            # Use fixed SPACESHIP_SIZE from config
            
            # Define base spaceship shape (triangle pointing up)
            # Points relative to (0,0) center
            #       A (0, -SPACESHIP_SIZE*0.8)
            #      / \
            #     /   \
            # B(-SPACESHIP_SIZE*0.5, SPACESHIP_SIZE*0.4) C(SPACESHIP_SIZE*0.5, SPACESHIP_SIZE*0.4)
            base_points = [
                np.array([0, -SPACESHIP_SIZE * 0.8]),  # Nose
                np.array([-SPACESHIP_SIZE * 0.5, SPACESHIP_SIZE * 0.4]), # Left tail
                np.array([SPACESHIP_SIZE * 0.5, SPACESHIP_SIZE * 0.4])  # Right tail
            ]

            # Spaceship orientation is now directly from probe['angle']
            # probe['angle'] = 0 means pointing along world +X.
            # Spaceship base_points has nose at [0, -length], i.e., pointing along local -Y.
            # To align local -Y with world +X, rotate by +pi/2.
            # To align local -Y with world +Y (probe['angle']=pi/2), rotate by pi.
            # So, the visual rotation angle for the matrix is probe['angle'] + np.pi / 2.
            current_probe_angle = probe.get('angle', -np.pi/2) # Default to pointing up if no angle
            angle_rad = current_probe_angle + (np.pi / 2)
            
            rotation_matrix = np.array([
                [math.cos(angle_rad), -math.sin(angle_rad)],
                [math.sin(angle_rad), math.cos(angle_rad)]
            ])
            
            rotated_points = [rotation_matrix @ p for p in base_points]
            # screen_points uses the (potentially interpolated) screen_pos for its center
            screen_ship_points = [(rp[0] + screen_pos[0], rp[1] + screen_pos[1]) for rp in rotated_points]
            
            pygame.draw.polygon(self.screen, color, screen_ship_points)
            pygame.draw.polygon(self.screen, (200, 200, 200), screen_ship_points, 1) # Outline

            # ENHANCED THRUST VISUALIZATION
            smoothing_state = probe.get('action_smoothing_state', {})
            # current_thrust_ramp is target_power * ramp_progress. Max target_power is 3 (index for THRUST_FORCE).
            # So current_thrust_ramp can go from 0 up to 3 (if ramp_progress is 1 and target_power is 3).
            thrust_ramp_value = smoothing_state.get('current_thrust_ramp', 0.0)
            
            # is_thrusting_visual is set in environment if actual force is applied.
            # We can use thrust_ramp_value directly from smoothing_state if available.
            # The prompt uses `probe.get('is_thrusting_visual', False)` which is fine.
            # Let's use the ramp value for intensity.
            
            if probe.get('is_thrusting_visual', False) and not is_low_power and thrust_ramp_value > 0.01:
                # Max possible value for target_thrust_power is len(THRUST_FORCE) - 1
                # THRUST_FORCE is now [0.0, 0.08, 0.18, 0.32]
                max_ramp_val = float(len(THRUST_FORCE) - 1) if len(THRUST_FORCE) > 1 else 1.0
                flame_intensity = min(1.0, thrust_ramp_value / max_ramp_val) if max_ramp_val > 0 else 0.0
                
                # Base flame size with intensity scaling
                base_flame_length = SPACESHIP_SIZE * (0.4 + 0.9 * flame_intensity)
                
                # Multi-layered flickering for more realistic flame
                time_ticks = pygame.time.get_ticks()
                primary_flicker = 0.85 + 0.15 * math.sin(time_ticks * 0.025)
                secondary_flicker = 0.95 + 0.05 * math.sin(time_ticks * 0.04 + 1.5)
                
                flame_length = base_flame_length * primary_flicker * secondary_flicker
                flame_width = SPACESHIP_SIZE * 0.3 * (0.6 + 0.4 * flame_intensity) # Adjusted width formula
                
                thrust_power_level_visual = probe.get('thrust_power_visual', 0) # Get target power level

                # Create multiple flame layers for depth
                for layer in range(2):  # Inner and outer flame
                    layer_factor = 1.0 - layer * 0.3 # Inner layer (0) is 1.0, outer (1) is 0.7
                    layer_length = flame_length * layer_factor
                    layer_width = flame_width * layer_factor
                    
                    # Points for this flame layer, relative to ship center, pointing "down" (positive Y in local ship coords)
                    layer_flame_points_local = [
                        np.array([0, SPACESHIP_SIZE * 0.4 + layer_length]), # Tip of the flame
                        np.array([-layer_width * 0.5, SPACESHIP_SIZE * 0.4]), # Base left
                        np.array([layer_width * 0.5, SPACESHIP_SIZE * 0.4])  # Base right
                    ]
                    
                    # Rotate flame points according to ship's angle
                    rotated_layer_points = [rotation_matrix @ p for p in layer_flame_points_local]
                    # Translate to screen position
                    screen_layer_points = [(rp[0] + screen_pos[0], rp[1] + screen_pos[1])
                                          for rp in rotated_layer_points]
                    
                    # Color variations by layer and intensity
                    if layer == 0:  # Inner flame - hotter
                        # Brighter/whiter for higher intensity
                        if thrust_power_level_visual >= 2: # Corresponds to THRUST_FORCE index 2 (0.18) or 3 (0.32)
                            layer_color = (255, 200, 150) # Bright orange-yellow
                        else: # Lower thrust
                            layer_color = (255, 230, 180) # Paler yellow
                    else:  # Outer flame - cooler
                        if thrust_power_level_visual >= 2:
                            layer_color = (255, 120, 0)   # Deep orange
                        else:
                            layer_color = (255, 165, 50)  # Softer orange
                    
                    pygame.draw.polygon(self.screen, layer_color, screen_layer_points)

            # Draw mining laser if active (using screen_pos for laser origin calculation)
            if probe.get('is_mining_visual', False) and probe.get('mining_target_pos_visual') is not None and not is_low_power:
                mining_target_world_pos = probe['mining_target_pos_visual']
                mining_target_screen_pos = self.world_to_screen(mining_target_world_pos)
                
                # Laser originates from the ship's nose
                # Ship's nose in local coords: np.array([0, -SPACESHIP_SIZE * 0.8])
                # This point is already part of 'base_points[0]'
                # We need its rotated and screen-translated position
                laser_origin_local = base_points[0] # Nose of the ship
                rotated_laser_origin = rotation_matrix @ laser_origin_local
                # Laser start screen pos uses the (potentially interpolated) screen_pos
                laser_start_screen_pos = (rotated_laser_origin[0] + screen_pos[0], rotated_laser_origin[1] + screen_pos[1])
                
                laser_color = (0, 255, 150) # Bright cyan/green
                pygame.draw.line(self.screen, laser_color, laser_start_screen_pos, mining_target_screen_pos, 2)

            # Draw target indicator line
            selected_target_info = probe.get('selected_target_info')
            if selected_target_info and selected_target_info.get('world_pos') is not None and not is_low_power:
                target_world_pos = selected_target_info['world_pos']
                target_screen_pos = self.world_to_screen(target_world_pos)
                
                # Line from ship center to target
                # We can use screen_pos (ship's center) as the start
                target_line_color = (255, 0, 255, 100) # Magenta, slightly transparent if using alpha
                
                # Draw a thicker, perhaps dashed line or a line with a small circle at the target
                # For simplicity, a magenta line:
                # Target line starts from the (potentially interpolated) screen_pos
                pygame.draw.line(self.screen, target_line_color, screen_pos, target_screen_pos, 1)
                
                # Optionally, draw a small circle at the target resource to highlight it
                pygame.draw.circle(self.screen, target_line_color, target_screen_pos, 5, 1)

            # SMOOTH TRAIL RENDERING
            # Pass the raw (non-interpolated) screen position for trail physics,
            # or the displayed one for visual trail. Prompt implies visual.
            # Let's use screen_pos_display (which is screen_pos here) for the trail points.
            self._update_smooth_trail(probe_id, screen_pos, probe.get('velocity', np.array([0.0,0.0])))


            # Draw probe ID (adjust position based on spaceship size, using screen_pos)
            id_text_color = (150, 150, 150) if is_low_power else (255, 255, 255)
            id_text_content = str(probe_id) + (" (LP)" if is_low_power else "")
            text_surface = self.small_font.render(id_text_content, True, id_text_color)
            # Position text slightly above the ship's center
            text_rect = text_surface.get_rect(center=(screen_pos[0], screen_pos[1] - SPACESHIP_SIZE - 5))
            self.screen.blit(text_surface, text_rect)
            
            # Draw energy bar (only if not in low power)
            if not is_low_power:
                bar_width = 20
                bar_height = 4
                # Position bar below the ship
                bar_x = screen_pos[0] - bar_width // 2
                # Energy bar positioned relative to (potentially interpolated) screen_pos
                bar_y = screen_pos[1] + SPACESHIP_SIZE * 0.5 + 5
                
                pygame.draw.rect(self.screen, (100, 100, 100),
                               (bar_x, bar_y, bar_width, bar_height)) # Background
                
                energy_width = int((max(0, probe['energy']) / MAX_ENERGY) * bar_width)
                energy_color = (255, 255, 0) if probe['energy'] > 30 else (255, 100, 100)
                pygame.draw.rect(self.screen, energy_color,
                               (bar_x, bar_y, energy_width, bar_height))
    
    def _update_smooth_trail(self, probe_id, screen_pos, velocity):
        """Enhanced trail system with velocity-based effects"""
        if probe_id not in self.probe_trails:
            self.probe_trails[probe_id] = []
        
        # screen_pos is already the (potentially interpolated) display position
        self.probe_trails[probe_id].append(screen_pos)
        max_trail_length = 80  # Longer trails for smooth physics
        
        if len(self.probe_trails[probe_id]) > max_trail_length:
            self.probe_trails[probe_id].pop(0)
        
        # Draw trail with velocity-based coloring
        if len(self.probe_trails[probe_id]) > 1:
            # Normalize velocity magnitude. Ensure MAX_VELOCITY is not zero.
            speed = np.linalg.norm(velocity)
            speed_factor = 0.0
            if MAX_VELOCITY > 1e-6:
                 speed_factor = min(1.0, speed / MAX_VELOCITY)
            
            for i in range(1, len(self.probe_trails[probe_id])):
                alpha_base = (i / len(self.probe_trails[probe_id]))
                alpha = alpha_base ** 0.7 # Make tail fade a bit quicker
                thickness = max(1, int(alpha * 3 * (0.5 + 0.5 * speed_factor)))
                
                # Color intensity based on speed
                base_color = self.colors['trail']
                # Ensure trail_color components are valid (0-255)
                trail_color_cal = [int(c * alpha * (0.3 + 0.7 * speed_factor)) for c in base_color]
                trail_color = tuple(max(0, min(255, tc_val)) for tc_val in trail_color_cal)

                if alpha > 0.1:  # Don't draw very faint segments (was 0.2, 0.1 is fine)
                    pygame.draw.line(self.screen, trail_color,
                                   self.probe_trails[probe_id][i-1],
                                   self.probe_trails[probe_id][i], thickness)

    def _draw_ui(self, environment, probe_agents):
        """Draw UI information panel"""
        ui_x = SCREEN_WIDTH - 190
        ui_y = 10
        line_height = 16
        padding = 5
        
        # Background panel
        pygame.draw.rect(self.screen, self.colors['ui_bg'],
                        (ui_x - padding, ui_y - padding, 180 + 2 * padding, SCREEN_HEIGHT - 2 * padding))
        
        y_offset = ui_y

        # --- General Statistics ---
        active_probes_count = sum(1 for probe in environment.probes.values() if probe['energy'] > 0)
        total_energy_active = sum(probe['energy'] for probe in environment.probes.values() if probe['energy'] > 0)
        avg_energy_active = total_energy_active / max(active_probes_count, 1)
        
        generations = {}
        for probe in environment.probes.values():
            if probe['energy'] > 0:
                gen = probe['generation']
                generations[gen] = generations.get(gen, 0) + 1
        
        general_stats_text = [
            f"Step: {environment.step_count}",
            f"Active Probes: {active_probes_count}",
            f"Total Probes: {len(environment.probes)}",
            f"Avg Energy (Active): {avg_energy_active:.1f}",
            f"Resources: {len([r for r in environment.resources if r.amount > 0])}",
            f"Messages: {len(environment.messages)}",
            "", "Generations:"
        ]
        for gen, count in sorted(generations.items()):
            general_stats_text.append(f"  Gen {gen}: {count}")
        general_stats_text.append("") # Spacer

        for stat_text in general_stats_text:
            text_surface = self.small_font.render(stat_text, True, self.colors['ui_text'])
            self.screen.blit(text_surface, (ui_x, y_offset))
            y_offset += line_height

        # --- Probe List ---
        self.ui_probe_id_rects.clear() # Clear old rects
        probe_list_header = self.small_font.render("Probes (Click to select):", True, self.colors['ui_text'])
        self.screen.blit(probe_list_header, (ui_x, y_offset))
        y_offset += line_height

        sorted_probe_ids = sorted(environment.probes.keys())

        for probe_id in sorted_probe_ids:
            probe = environment.probes[probe_id]
            status_char = "LP" if probe['energy'] <= 0 else "OK"
            probe_info_str = f"  #{probe_id}: E{max(0,probe['energy']):.1f} G{probe['generation']} ({status_char})"
            
            text_color = self.colors['ui_text']
            if self.selected_probe_id_ui == probe_id:
                text_color = (255, 255, 0) # Highlight selected probe in yellow

            text_surface = self.small_font.render(probe_info_str, True, text_color)
            text_rect = text_surface.get_rect(topleft=(ui_x, y_offset))
            self.screen.blit(text_surface, text_rect)
            self.ui_probe_id_rects[probe_id] = text_rect # Store rect for click detection
            y_offset += line_height
        
        y_offset += line_height # Spacer

        # --- Selected Probe Details ---
        if self.selected_probe_id_ui is not None and self.selected_probe_id_ui in environment.probes:
            selected_probe = environment.probes[self.selected_probe_id_ui]
            details_header = self.small_font.render(f"Details for Probe #{self.selected_probe_id_ui}:", True, (200,200,255))
            self.screen.blit(details_header, (ui_x, y_offset))
            y_offset += line_height

            details = [
                f"  Energy: {selected_probe['energy']:.2f} / {MAX_ENERGY}",
                f"  Position: ({selected_probe['position'][0]:.1f}, {selected_probe['position'][1]:.1f})",
                f"  Velocity: ({selected_probe['velocity'][0]:.2f}, {selected_probe['velocity'][1]:.2f})",
                f"  Angle: {math.degrees(selected_probe['angle']):.1f}°",
                f"  Ang. Vel: {math.degrees(selected_probe['angular_velocity']):.2f}°/s",
                f"  Age: {selected_probe['age']}",
                f"  Generation: {selected_probe['generation']}",
                f"  Mass: {selected_probe['mass']}",
                f"  Target: {selected_probe.get('selected_target_info', 'None')}",
                f"  Dist to Target: {selected_probe.get('distance_to_target_last_step', 'N/A')}",
                "  Smoothing State:"
            ]
            smoothing_state = selected_probe.get('action_smoothing_state', {})
            for key, val in smoothing_state.items():
                details.append(f"    {key}: {val:.2f}" if isinstance(val, float) else f"    {key}: {val}")
            
            for detail_text in details:
                if y_offset + line_height > SCREEN_HEIGHT - padding: # Prevent drawing off-screen
                    break
                text_surface = self.small_font.render(detail_text, True, self.colors['ui_text'])
                self.screen.blit(text_surface, (ui_x, y_offset))
                y_offset += line_height
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left mouse button
                    # Check if click is on a probe ID in the UI
                    clicked_on_probe_id = False
                    for probe_id, rect in self.ui_probe_id_rects.items():
                        if rect.collidepoint(event.pos):
                            if self.selected_probe_id_ui == probe_id:
                                self.selected_probe_id_ui = None # Deselect if clicking the same one
                            else:
                                self.selected_probe_id_ui = probe_id
                            clicked_on_probe_id = True
                            break
                    # If clicked in UI panel area but not on a specific ID, deselect
                    # Assuming UI panel x starts at SCREEN_WIDTH - 190 - 5
                    ui_panel_x_start = SCREEN_WIDTH - 190 - 5
                    if not clicked_on_probe_id and event.pos[0] >= ui_panel_x_start:
                         # Check if it's not a click on the general stats area above probe list
                         # This is a rough check; a more robust way would be to define a rect for the selectable list
                         # For now, if it's in the panel and not a probe, deselect.
                         # self.selected_probe_id_ui = None # Potentially deselect if clicking empty UI space
                         pass # Let's not deselect on empty space for now, only on re-click or specific deselect button

        return True