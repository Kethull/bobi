import pygame
import numpy as np
import math
from config import * # Assuming config might have UI related constants later

class ModernUI:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        try:
            self.ui_font = pygame.font.Font(None, 18) # Default font if specific one fails
            self.title_font = pygame.font.Font(None, 22) # Slightly smaller title
            self.small_font = pygame.font.Font(None, 15) # Slightly larger small
        except pygame.error: # Fallback if font system not fully init or specific font missing
            self.ui_font = pygame.font.SysFont("arial", 16)
            self.title_font = pygame.font.SysFont("arial", 20)
            self.small_font = pygame.font.SysFont("arial", 13)

        self.ui_colors = {
            'panel_bg': (10, 30, 50, 190), # Darker, more saturated blue, slightly more alpha
            'panel_border': (30, 120, 220, 200), # Brighter border
            'text_primary': (210, 230, 255), # Lighter primary text
            'text_secondary': (150, 170, 200), # Lighter secondary
            'accent_blue': (50, 180, 255),
            'accent_green': (50, 220, 120),
            'accent_yellow': (255, 220, 80),
            'accent_red': (255, 120, 120),
            'meter_bg': (25, 45, 65), # Slightly lighter meter bg
            'meter_fill': (50, 180, 255) # Consistent with accent_blue
        }
        self.line_height = 18 # For ui_font
        self.small_line_height = 15 # For small_font
        self.title_line_height = 24 # For title_font

    def draw_holographic_panel(self, surface, rect, title="", content_lines=None, scroll_offset_y=0):
        if content_lines is None: content_lines = []
            
        panel_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(panel_surf, self.ui_colors['panel_bg'], panel_surf.get_rect(), border_radius=3) # Slight rounding
        
        # Border
        pygame.draw.rect(panel_surf, self.ui_colors['panel_border'], panel_surf.get_rect(), 2, border_radius=3)
        
        # Corner accents (simplified)
        corner_size = 8
        accent_color = self.ui_colors['accent_blue']
        pygame.draw.line(panel_surf, accent_color, (2, corner_size), (corner_size, 2), 2)
        pygame.draw.line(panel_surf, accent_color, (rect.width - 2, corner_size), (rect.width - corner_size, 2), 2)
        pygame.draw.line(panel_surf, accent_color, (2, rect.height - corner_size), (corner_size, rect.height - 2), 2)
        pygame.draw.line(panel_surf, accent_color, (rect.width - 2, rect.height - corner_size), (rect.width - corner_size, rect.height - 2), 2)
        
        content_clip_rect = pygame.Rect(5, 5, rect.width - 10, rect.height - 10)
        panel_surf.set_clip(content_clip_rect)

        y_offset = 10 - scroll_offset_y
        if title:
            title_surf = self.title_font.render(title, True, self.ui_colors['text_primary'])
            panel_surf.blit(title_surf, (10, y_offset))
            y_offset += self.title_line_height 
        
        for line_idx, line_item in enumerate(content_lines):
            if y_offset < -self.line_height : # Content scrolled completely above panel top
                y_offset += self.line_height # Still increment y_offset to calculate total content height
                continue
            if y_offset > rect.height - 10: # Content scrolled below panel bottom
                break 

            text_to_render = ""
            color_to_use = self.ui_colors['text_secondary']
            font_to_use = self.ui_font
            current_line_height = self.line_height

            if isinstance(line_item, dict):
                text_to_render = line_item.get('text', '')
                color_to_use = line_item.get('color', self.ui_colors['text_secondary'])
                font_choice = line_item.get('font', 'default')
                if font_choice == 'title': 
                    font_to_use = self.title_font
                    current_line_height = self.title_line_height
                elif font_choice == 'small':
                    font_to_use = self.small_font
                    current_line_height = self.small_line_height
            else:
                text_to_render = str(line_item)
            
            if text_to_render.strip():
                text_surf = font_to_use.render(text_to_render, True, color_to_use)
                panel_surf.blit(text_surf, (15, y_offset)) # Indent content slightly
            y_offset += current_line_height
        
        panel_surf.set_clip(None)
        surface.blit(panel_surf, rect)
        return y_offset + scroll_offset_y # Return total content height for scrollbar calculation

    def draw_circular_meter(self, surface, center, radius, value, max_value, 
                           meter_color, label="", show_text=True, width=3):
        if max_value <= 0: return
            
        pygame.draw.circle(surface, self.ui_colors['meter_bg'], center, radius, width)
        
        progress_ratio = np.clip(value / max_value, 0.0, 1.0)
        
        if progress_ratio > 0.005: # Only draw if there's some progress
            num_segments = int(50 * progress_ratio) + 2 # Ensure at least 2 points for a line
            
            arc_points = []
            for i in range(num_segments + 1):
                angle_rad = -math.pi/2 + (2 * math.pi * progress_ratio * (i / float(num_segments)))
                x = center[0] + (radius - width//2) * math.cos(angle_rad)
                y = center[1] + (radius - width//2) * math.sin(angle_rad)
                arc_points.append((x,y))

            if len(arc_points) > 1:
                 pygame.draw.lines(surface, meter_color, False, arc_points, width)
        
        if show_text:
            text_val = f"{int(value)}" if value >=1 or value == 0 else f"{value:.1f}"
            text_surf = self.small_font.render(text_val, True, self.ui_colors['text_primary'])
            text_rect = text_surf.get_rect(center=center)
            surface.blit(text_surf, text_rect)
        
        if label:
            label_surf = self.small_font.render(label, True, self.ui_colors['text_secondary'])
            label_rect = label_surf.get_rect(center=(center[0], center[1] + radius + self.small_line_height // 2 + 2))
            surface.blit(label_surf, label_rect)
    
    def draw_velocity_vector(self, surface, probe_screen_pos, world_velocity, scale=5):
        speed = np.linalg.norm(world_velocity)
        if speed < 0.01: return # Don't draw for negligible velocity
            
        vector_end_screen = probe_screen_pos + world_velocity * scale # Velocity is in world units/step
        
        pygame.draw.line(surface, self.ui_colors['accent_yellow'], 
                        probe_screen_pos.astype(int), vector_end_screen.astype(int), 2)
        
        angle_rad = math.atan2(world_velocity[1], world_velocity[0])
        arrow_size = 6
        p1 = vector_end_screen + np.array([math.cos(angle_rad + math.pi * 0.85) * arrow_size, 
                                           math.sin(angle_rad + math.pi * 0.85) * arrow_size])
        p2 = vector_end_screen + np.array([math.cos(angle_rad - math.pi * 0.85) * arrow_size, 
                                           math.sin(angle_rad - math.pi * 0.85) * arrow_size])
        pygame.draw.polygon(surface, self.ui_colors['accent_yellow'], [vector_end_screen.astype(int), p1.astype(int), p2.astype(int)])
        
        speed_text = f"{speed:.1f} u/s"
        text_surf = self.small_font.render(speed_text, True, self.ui_colors['text_primary'])
        text_pos = vector_end_screen + np.array([8, -8]) # Offset from arrow tip
        surface.blit(text_surf, text_pos.astype(int))
    
    def draw_target_lock_indicator(self, surface, probe_screen_pos, target_screen_pos, world_distance):
        bracket_size = 12
        bracket_color = self.ui_colors['accent_red']
        
        # Brackets around target
        points_list = [
            # Top-left
            [(target_screen_pos[0] - bracket_size, target_screen_pos[1] - bracket_size // 2),
             (target_screen_pos[0] - bracket_size, target_screen_pos[1] - bracket_size),
             (target_screen_pos[0] - bracket_size // 2, target_screen_pos[1] - bracket_size)],
            # Top-right
            [(target_screen_pos[0] + bracket_size // 2, target_screen_pos[1] - bracket_size),
             (target_screen_pos[0] + bracket_size, target_screen_pos[1] - bracket_size),
             (target_screen_pos[0] + bracket_size, target_screen_pos[1] - bracket_size // 2)],
            # Bottom-left
            [(target_screen_pos[0] - bracket_size // 2, target_screen_pos[1] + bracket_size),
             (target_screen_pos[0] - bracket_size, target_screen_pos[1] + bracket_size),
             (target_screen_pos[0] - bracket_size, target_screen_pos[1] + bracket_size // 2)],
            # Bottom-right
            [(target_screen_pos[0] + bracket_size, target_screen_pos[1] + bracket_size // 2),
             (target_screen_pos[0] + bracket_size, target_screen_pos[1] + bracket_size),
             (target_screen_pos[0] + bracket_size // 2, target_screen_pos[1] + bracket_size)],
        ]
        for points in points_list:
            pygame.draw.lines(surface, bracket_color, False, [(int(p[0]), int(p[1])) for p in points], 2)

        distance_text = f"{world_distance:.0f}m"
        text_surf = self.small_font.render(distance_text, True, bracket_color)
        text_rect = text_surf.get_rect(midtop=(target_screen_pos[0], target_screen_pos[1] + bracket_size + 2))
        surface.blit(text_surf, text_rect)
        
        self.draw_dashed_line(surface, probe_screen_pos.astype(int), target_screen_pos.astype(int), bracket_color, 6, 3)

    def draw_dashed_line(self, surface, start_point, end_point, color, dash_len=8, gap_len=4):
        start_np = np.array(start_point, dtype=float)
        end_np = np.array(end_point, dtype=float)
        direction = end_np - start_np
        distance = np.linalg.norm(direction)
        if distance < 1e-3: return
        
        unit_direction = direction / distance
        current_pos = start_np.copy()
        drawn_distance = 0
        is_dash = True

        while drawn_distance < distance:
            segment_len = dash_len if is_dash else gap_len
            if drawn_distance + segment_len > distance:
                segment_len = distance - drawn_distance
            
            next_pos = current_pos + unit_direction * segment_len
            if is_dash:
                pygame.draw.line(surface, color, current_pos.astype(int), next_pos.astype(int), 1)
            
            current_pos = next_pos
            drawn_distance += segment_len
            is_dash = not is_dash
    
    def draw_radar_display(self, surface, rect_area, probe_world_pos, all_objects_world, radar_world_range=300):
        # Radar background (using holographic panel for consistency)
        self.draw_holographic_panel(surface, rect_area, title="Radar")
        
        radar_center_screen = np.array(rect_area.center)
        radar_radius_screen = min(rect_area.width, rect_area.height) // 2 - 15 # Padding for title/border

        # Range rings (drawn relative to radar_center_screen, clipped by panel)
        # Create a subsurface for radar content to clip it properly
        radar_content_rect = pygame.Rect(
            rect_area.left + 10, rect_area.top + 30, # Below title
            rect_area.width - 20, rect_area.height - 40 
        )
        if radar_content_rect.width <=0 or radar_content_rect.height <=0: return
        
        radar_surface = surface.subsurface(radar_content_rect).copy() # Copy to draw on it
        radar_surface.fill(self.ui_colors['panel_bg']) # Clear subsurface with panel bg
        
        # Calculate center relative to this new radar_surface
        sub_center_x = radar_content_rect.width // 2
        sub_center_y = radar_content_rect.height // 2
        sub_radius = min(sub_center_x, sub_center_y)


        for i in range(1, 4): # 3 rings
            ring_rad_screen = sub_radius * (i / 3.0)
            pygame.draw.circle(radar_surface, self.ui_colors['text_secondary'], 
                               (sub_center_x, sub_center_y), int(ring_rad_screen), 1)
        
        # Center dot for probe's own position
        pygame.draw.circle(radar_surface, self.ui_colors['accent_yellow'], (sub_center_x, sub_center_y), 2)

        for obj_dict in all_objects_world: # Expects list of dicts with 'position' and 'type'
            obj_world_pos = np.array(obj_dict.get('position', [0,0]))
            obj_type = obj_dict.get('type', 'unknown') # e.g. 'resource', 'probe'
            
            relative_pos_world = obj_world_pos - probe_world_pos
            dist_world = np.linalg.norm(relative_pos_world)

            if 0 < dist_world <= radar_world_range:
                # Scale world relative pos to radar screen relative pos
                # Angle is preserved, distance is scaled
                angle_rad = math.atan2(relative_pos_world[1], relative_pos_world[0])
                dist_on_radar_screen = (dist_world / radar_world_range) * sub_radius
                
                radar_blip_x = sub_center_x + dist_on_radar_screen * math.cos(angle_rad)
                radar_blip_y = sub_center_y + dist_on_radar_screen * math.sin(angle_rad)
                
                blip_color = self.ui_colors['text_secondary']
                blip_size = 1
                if obj_type == 'resource':
                    blip_color = self.ui_colors['accent_green']
                    blip_size = 3
                elif obj_type == 'probe':
                    blip_color = self.ui_colors['accent_blue']
                    blip_size = 2
                
                pygame.draw.circle(radar_surface, blip_color, (int(radar_blip_x), int(radar_blip_y)), blip_size)
        
        surface.blit(radar_surface, radar_content_rect.topleft)
    def update_data(self, environment_data=None, probe_agents_data=None,
                        selected_probe_id=None, selected_probe_details=None,
                        camera_offset=None, fps=0):
            """
            Update the UI with the latest data from the simulation.
            This method should be called each frame before draw_ui.
            """
            self.environment_data = environment_data
            self.probe_agents_data = probe_agents_data
            self.selected_probe_id = selected_probe_id
            self.selected_probe_details = selected_probe_details
            self.camera_offset = camera_offset
            self.fps = fps
            # Potentially update internal states of UI elements here based on new data

    def handle_event(self, event):
        """
        Process a Pygame event and update UI state or return actions.
        Returns:
            dict: A dictionary of actions or state changes.
                  Example: {'selected_probe_id': new_id, 'event_consumed': True}
        """
        # TODO: Implement actual UI event handling logic here.
        # This could involve checking for clicks on buttons, scrollbar interactions, etc.
        # For now, it's a stub.
        
        # Example: if a click on a probe list item happens:
        # if event.type == pygame.MOUSEBUTTONDOWN:
        #     mouse_pos = pygame.mouse.get_pos()
        #     # ... logic to check if mouse_pos is over a probe list item ...
        #     if clicked_on_probe_X:
        #         return {'selected_probe_id': 'probe_X_id', 'event_consumed': True}

        return {} # Return an empty dict if no UI action taken

    def draw_ui(self, surface):
        """
        Draw all UI elements onto the provided surface.
        This is the main drawing call for the ModernUI system.
        """
        # This is where all the individual draw calls for panels, meters, text, etc.
        # would be orchestrated based on the current UI state and data.

        # --- Example Layout ---
        # 1. Main Stats Panel (Top-Right or Top-Left)
        stats_panel_rect = pygame.Rect(self.screen_width - 260, 10, 250, 150)
        stats_content = []
        if hasattr(self, 'environment_data') and self.environment_data:
            env = self.environment_data
            active_probes = sum(1 for p in env.probes.values() if p['energy'] > 0)
            stats_content.extend([
                f"Step: {env.step_count}",
                f"FPS: {self.fps:.1f}",
                f"Active Probes: {active_probes} / {len(env.probes)}",
                f"Resources: {len([r for r in env.resources if r.amount > 0])}",
                f"Messages: {len(env.messages)}"
            ])
        self.draw_holographic_panel(surface, stats_panel_rect, "System Status", stats_content)

        # 2. Selected Probe Details Panel (Below Stats or separate area)
        if self.selected_probe_id and self.selected_probe_details:
            probe_panel_rect = pygame.Rect(self.screen_width - 260, 170, 250, 250)
            probe_details_content = [
                {'text': f"Probe ID: {self.selected_probe_id}", 'font': 'title'},
                f"Energy: {self.selected_probe_details['energy']:.1f} / {MAX_ENERGY}", # type: ignore
                f"Position: ({self.selected_probe_details['position'][0]:.1f}, {self.selected_probe_details['position'][1]:.1f})",
                f"Velocity: ({self.selected_probe_details['velocity'][0]:.1f}, {self.selected_probe_details['velocity'][1]:.1f})",
                f"Angle: {math.degrees(self.selected_probe_details['angle']):.1f}°",
                f"Age: {self.selected_probe_details['age']}",
                # Add more details as needed
            ]
            self.draw_holographic_panel(surface, probe_panel_rect, "Selected Probe", probe_details_content)

            # Draw circular energy meter for selected probe
            meter_center = (self.screen_width - 260 + 50, 170 + 250 + 40)
            self.draw_circular_meter(surface, meter_center, 30,
                                     self.selected_probe_details['energy'], MAX_ENERGY, # type: ignore
                                     self.ui_colors['accent_green'], "Energy")
            
            # Draw velocity vector for selected probe (if it has one)
            if 'position' in self.selected_probe_details and 'velocity' in self.selected_probe_details:
                 # Need to convert world pos to screen pos for the probe
                 # This requires access to the main visualization's world_to_screen or similar logic
                 # For now, let's assume selected_probe_details might contain a 'screen_pos' if available
                 # Or, ModernUI needs a way to do this conversion.
                 # This part is complex as UI doesn't directly know about world_to_screen.
                 # Visualization should pass screen_pos of selected probe to update_data.
                 pass # Placeholder for velocity vector drawing

        # 3. Radar Display (Example: Bottom-Left)
        radar_rect = pygame.Rect(10, self.screen_height - 210, 200, 200)
        all_objects = []
        if hasattr(self, 'environment_data') and self.environment_data:
            probe_world_pos = np.array([0,0]) # Default if no selected probe
            if self.selected_probe_id and self.selected_probe_details:
                 probe_world_pos = np.array(self.selected_probe_details['position'])
            elif len(self.environment_data.probes) > 0: # Fallback to first probe
                 probe_world_pos = np.array(list(self.environment_data.probes.values())[0]['position'])


            for res in self.environment_data.resources:
                if res.amount > 0:
                    all_objects.append({'position': res.position, 'type': 'resource'})
            for p_id, p_data in self.environment_data.probes.items():
                if p_id != self.selected_probe_id : # Don't show self on radar as a blip
                     all_objects.append({'position': p_data['position'], 'type': 'probe'})
            
            self.draw_radar_display(surface, radar_rect, probe_world_pos, all_objects)

        # 4. Alerts/Notifications Area (e.g., bottom center)
        # ...
        
        # 5. Probe List (Scrollable panel)
        # ... This would require more complex state management for scrolling and item selection
        # ... and interaction within handle_event.