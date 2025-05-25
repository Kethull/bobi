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
            'panel_bg': (10, 30, 50, 190), 
            'panel_border': (30, 120, 220, 200),
            'text_primary': (210, 230, 255),
            'text_secondary': (150, 170, 200),
            'accent_blue': (50, 180, 255),
            'accent_green': (50, 220, 120),
            'accent_yellow': (255, 220, 80),
            'accent_red': (255, 120, 120),
            'meter_bg': (25, 45, 65),
            'meter_fill': (50, 180, 255)
        }
        self.line_height = 18 
        self.small_line_height = 15
        self.title_line_height = 24

        # UI state variables
        self.clickable_elements = {}
        self.probe_list_scroll_offset = 0
        self.probe_list_panel_height = 220  # Default height for the probe list panel
        self.probe_list_item_height = self.small_line_height + 4 # Approx height for each item (font + padding)

    def draw_holographic_panel(self, surface, rect, title="", content_lines=None, scroll_offset_y=0, item_type_for_clickables=None):
        if content_lines is None: content_lines = []
            
        panel_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(panel_surf, self.ui_colors['panel_bg'], panel_surf.get_rect(), border_radius=3)
        
        pygame.draw.rect(panel_surf, self.ui_colors['panel_border'], panel_surf.get_rect(), 2, border_radius=3)
        
        corner_size = 8
        accent_color = self.ui_colors['accent_blue']
        pygame.draw.line(panel_surf, accent_color, (2, corner_size), (corner_size, 2), 2)
        pygame.draw.line(panel_surf, accent_color, (rect.width - 2, corner_size), (rect.width - corner_size, 2), 2)
        pygame.draw.line(panel_surf, accent_color, (2, rect.height - corner_size), (corner_size, rect.height - 2), 2)
        pygame.draw.line(panel_surf, accent_color, (rect.width - 2, rect.height - corner_size), (rect.width - corner_size, rect.height - 2), 2)
        
        # This is the clipping rect for the panel's content area, relative to panel_surf
        content_clip_rect_on_panel_surf = pygame.Rect(5, 5, rect.width - 10, rect.height - 10)
        panel_surf.set_clip(content_clip_rect_on_panel_surf)

        y_offset = 10 - scroll_offset_y # Initial y position for content on panel_surf
        if title:
            title_surf = self.title_font.render(title, True, self.ui_colors['text_primary'])
            panel_surf.blit(title_surf, (10, y_offset))
            y_offset += self.title_line_height 
        
        for line_idx, line_item in enumerate(content_lines):
            # Calculate current line height based on font
            text_to_render = ""
            color_to_use = self.ui_colors['text_secondary']
            font_to_use = self.ui_font
            current_actual_line_height = self.line_height # Use actual font height for this item

            if isinstance(line_item, dict):
                text_to_render = line_item.get('text', '')
                color_to_use = line_item.get('color', self.ui_colors['text_secondary'])
                font_choice = line_item.get('font', 'default')
                if font_choice == 'title': 
                    font_to_use = self.title_font
                    current_actual_line_height = self.title_line_height
                elif font_choice == 'small':
                    font_to_use = self.small_font
                    current_actual_line_height = self.small_line_height
            else:
                text_to_render = str(line_item)

            # Check if item is outside visible area before rendering (optimization)
            if y_offset >= rect.height - 5 : # Item starts below panel bottom padding
                 # Increment y_offset anyway for total height calculation if needed, but don't break yet
                 # as total height calculation is now separate.
                 # For clickable items, we only care about those rendered.
                 pass # It will be skipped by the blit check anyway
            
            if y_offset + current_actual_line_height < 5: # Item ends above panel top padding
                y_offset += current_actual_line_height
                continue


            if text_to_render.strip():
                text_surf = font_to_use.render(text_to_render, True, color_to_use)
                # item_rect_on_panel is relative to the panel_surf's top-left (0,0)
                item_rect_on_panel = text_surf.get_rect(topleft=(15, y_offset))
                
                # Only blit if item is at least partially visible within the clip_rect
                if item_rect_on_panel.bottom > content_clip_rect_on_panel_surf.top and \
                   item_rect_on_panel.top < content_clip_rect_on_panel_surf.bottom:
                    panel_surf.blit(text_surf, item_rect_on_panel.topleft)

                    if item_type_for_clickables and isinstance(line_item, dict) and 'id' in line_item:
                        # screen_item_rect is relative to the main game screen
                        screen_item_rect = pygame.Rect(
                            rect.left + item_rect_on_panel.left, 
                            rect.top + item_rect_on_panel.top,   
                            item_rect_on_panel.width,
                            item_rect_on_panel.height 
                        )
                        element_key = f"{item_type_for_clickables}_{line_item['id']}"
                        self.clickable_elements[element_key] = {
                            'rect': screen_item_rect, 
                            'action': 'select_probe', 
                            'id': line_item['id']
                        }
            y_offset += current_actual_line_height
        
        panel_surf.set_clip(None)
        surface.blit(panel_surf, rect)
        
        # Calculate total content height (unscrolled) for scrollbar logic
        unscrolled_content_height = 10 # Initial top padding for content lines area
        if title:
            unscrolled_content_height += self.title_line_height + 5 # Title text + padding after title
        
        for item_data in content_lines:
            line_h = self.line_height # Default line height
            if isinstance(item_data, dict):
                font_choice = item_data.get('font', 'default')
                if font_choice == 'title': line_h = self.title_line_height
                elif font_choice == 'small': line_h = self.small_line_height
            unscrolled_content_height += line_h
        unscrolled_content_height += 5 # Bottom padding for content lines area
        
        return unscrolled_content_height

    def draw_circular_meter(self, surface, center, radius, value, max_value, 
                           meter_color, label="", show_text=True, width=3):
        if max_value <= 0: return
            
        pygame.draw.circle(surface, self.ui_colors['meter_bg'], center, radius, width)
        
        progress_ratio = np.clip(value / max_value, 0.0, 1.0)
        
        if progress_ratio > 0.005: 
            num_segments = int(50 * progress_ratio) + 2 
            
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
        if speed < 0.01: return
            
        vector_end_screen = probe_screen_pos + world_velocity * scale
        
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
        text_pos = vector_end_screen + np.array([8, -8]) 
        surface.blit(text_surf, text_pos.astype(int))
    
    def draw_target_lock_indicator(self, surface, probe_screen_pos, target_screen_pos, world_distance):
        bracket_size = 12
        bracket_color = self.ui_colors['accent_red']
        
        points_list = [
            [(target_screen_pos[0] - bracket_size, target_screen_pos[1] - bracket_size // 2),
             (target_screen_pos[0] - bracket_size, target_screen_pos[1] - bracket_size),
             (target_screen_pos[0] - bracket_size // 2, target_screen_pos[1] - bracket_size)],
            [(target_screen_pos[0] + bracket_size // 2, target_screen_pos[1] - bracket_size),
             (target_screen_pos[0] + bracket_size, target_screen_pos[1] - bracket_size),
             (target_screen_pos[0] + bracket_size, target_screen_pos[1] - bracket_size // 2)],
            [(target_screen_pos[0] - bracket_size // 2, target_screen_pos[1] + bracket_size),
             (target_screen_pos[0] - bracket_size, target_screen_pos[1] + bracket_size),
             (target_screen_pos[0] - bracket_size, target_screen_pos[1] + bracket_size // 2)],
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
    
    def update_data(self, environment_data=None, probe_agents_data=None,
                        selected_probe_id=None, selected_probe_details=None,
                        camera_offset=None, fps=0):
            self.environment_data = environment_data
            self.probe_agents_data = probe_agents_data
            self.selected_probe_id = selected_probe_id
            self.selected_probe_details = selected_probe_details
            self.camera_offset = camera_offset
            self.fps = fps

    def handle_event(self, event):
        # Clickable element handling
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # Left click
                mouse_pos = pygame.mouse.get_pos()
                for element_key, data in self.clickable_elements.items():
                    if data['rect'].collidepoint(mouse_pos):
                        if data['action'] == 'select_probe':
                            return {'selected_probe_id': data['id'], 'event_consumed': True}
            
            # Probe list scrolling (mouse wheel)
            probe_list_panel_rect_for_scroll = pygame.Rect(
                10, 
                self.screen_height - self.probe_list_panel_height - 10, 
                200, 
                self.probe_list_panel_height
            )
            if probe_list_panel_rect_for_scroll.collidepoint(pygame.mouse.get_pos()):
                scroll_increment = self.probe_list_item_height * 2 # Scroll two items at a time

                if event.button == 4: # Scroll up
                    self.probe_list_scroll_offset = max(0, self.probe_list_scroll_offset - scroll_increment)
                    return {'event_consumed': True}
                elif event.button == 5: # Scroll down
                    if hasattr(self, 'environment_data') and self.environment_data:
                        num_probes = len(self.environment_data.probes)
                        if num_probes > 0:
                            # Calculate total content height for the probe list items
                            # This must match the calculation in draw_holographic_panel's return value for the probe list
                            _title_h = self.title_line_height + 5 if "Probes" else 0 # Title + padding after
                            _items_total_h = num_probes * self.probe_list_item_height 
                            # Total height of content within the scrollable area of the panel
                            total_content_h_for_scroll = 10 + _title_h + _items_total_h + 5 # top_pad + title_section + items + bottom_pad
                            
                            # Visible height of the content area within the panel (panel height minus its own chrome/padding)
                            panel_internal_display_height = self.probe_list_panel_height - (5 + 5) # panel_clip_rect top/bottom padding
                                                        
                            max_scroll = max(0, total_content_h_for_scroll - panel_internal_display_height)
                            self.probe_list_scroll_offset = min(max_scroll, self.probe_list_scroll_offset + scroll_increment)
                        return {'event_consumed': True}
        return {}

    def draw_ui(self, surface):
        self.clickable_elements.clear() # Clear before drawing new frame's elements

        # 1. Main Stats Panel
        stats_panel_rect = pygame.Rect(self.screen_width - 260, 10, 250, 150)
        stats_content = []
        if hasattr(self, 'environment_data') and self.environment_data:
            env = self.environment_data
            active_probes = sum(1 for p_data in env.probes.values() if p_data['energy'] > 0)
            stats_content.extend([
                f"Step: {env.step_count}",
                f"FPS: {self.fps:.1f}",
                f"Active Probes: {active_probes} / {len(env.probes)}",
                f"Resources: {len([r for r in env.resources if r.amount > 0])}",
                f"Messages: {len(env.messages)}"
            ])
        self.draw_holographic_panel(surface, stats_panel_rect, "System Status", stats_content)

        # 2. Selected Probe Details Panel
        if self.selected_probe_id and self.selected_probe_details:
            probe_panel_rect = pygame.Rect(self.screen_width - 260, 170, 250, 220) # Adjusted height
            probe_details_content = [
                {'text': f"Probe ID: {self.selected_probe_id}", 'font': 'title'},
                f"Energy: {self.selected_probe_details['energy']:.1f} / {MAX_ENERGY}",
                f"Position: ({self.selected_probe_details['position'][0]:.1f}, {self.selected_probe_details['position'][1]:.1f})",
                f"Velocity: ({self.selected_probe_details['velocity'][0]:.1f}, {self.selected_probe_details['velocity'][1]:.1f})",
                f"Angle: {math.degrees(self.selected_probe_details['angle']):.1f}°",
                f"Age: {self.selected_probe_details['age']}",
                f"Target: {self.selected_probe_details.get('current_target_id', 'None')}",
                f"Task: {self.selected_probe_details.get('current_task', 'Idle')}",
            ]
            self.draw_holographic_panel(surface, probe_panel_rect, "Selected Probe", probe_details_content)

            meter_center = (self.screen_width - 260 + 60, probe_panel_rect.bottom + 40)
            self.draw_circular_meter(surface, meter_center, 30,
                                     self.selected_probe_details['energy'], MAX_ENERGY,
                                     self.ui_colors['accent_green'], "Energy")
        
        # 3. Probe List Panel (Bottom-Left)
        probe_list_panel_rect = pygame.Rect(
            10, 
            self.screen_height - self.probe_list_panel_height - 10, 
            200, 
            self.probe_list_panel_height
        )
        probe_list_items_for_panel = []
        
        if hasattr(self, 'environment_data') and self.environment_data:
            sorted_probe_ids = sorted(self.environment_data.probes.keys())
            for probe_id in sorted_probe_ids:
                probe_data = self.environment_data.probes[probe_id]
                text = f"{probe_id} (E: {probe_data['energy']:.0f})"
                color = self.ui_colors['text_primary'] if probe_id == self.selected_probe_id else self.ui_colors['text_secondary']
                
                probe_list_items_for_panel.append({
                    'text': text, 
                    'color': color, 
                    'id': probe_id, 
                    'font': 'small' 
                })

        self.draw_holographic_panel(surface, probe_list_panel_rect, "Probes", 
                                    probe_list_items_for_panel, 
                                    scroll_offset_y=self.probe_list_scroll_offset,
                                    item_type_for_clickables='probe_list_item')

        # 4. Alerts/Notifications Area (Placeholder)
        # ...