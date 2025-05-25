# visualization.py
import pygame
import numpy as np
from typing import Dict, List
import math
from config import *
from environment import Resource # Added import for Resource
from environment import Message # Added import for Message
from organic_ship_renderer import OrganicShipRenderer
from particle_system import AdvancedParticleSystem
from visual_effects import VisualEffects, StarField # Added imports
from advanced_ui import ModernUI # Added import

class Visualization:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Bobiverse RL Simulation - Enhanced") # Updated caption
        self.clock = pygame.time.Clock()
        
        # Enhanced rendering systems
        self.ship_renderer = OrganicShipRenderer()  # Replace DetailedProbeRenderer
        self.particle_system = AdvancedParticleSystem()
        self.visual_effects = VisualEffects()
        self.starfield = StarField(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.modern_ui = ModernUI(SCREEN_WIDTH, SCREEN_HEIGHT)
        
        # Existing attributes...
        self.font = pygame.font.Font(None, 24) # Keep for now, ModernUI has its own
        self.small_font = pygame.font.Font(None, 16) # Keep for now
        self.probe_visual_cache = {}
        self.selected_probe_id_ui = None
        self.ui_probe_id_rects = {} # Will be managed by ModernUI or adapted
        self.probe_trails = {} # Keep for now, might be replaced or enhanced
        self.camera_offset = np.array([0.0, 0.0]) # For starfield parallax
        self.zoom_level = 16.0 # Initial zoom level to show about 5 AU diameter
        
        # Colors (old color dict, ModernUI has its own, this might be phased out)
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
        # WORLD_WIDTH and WORLD_HEIGHT are now WORLD_SIZE from config.py
        # The UI panel width (200) might need to be re-evaluated if ModernUI takes full screen or specific part.
        # For now, assuming the main viewport is still SCREEN_WIDTH - 200 for game world.
        # If ModernUI is an overlay, then full SCREEN_WIDTH might be used for world.
        # Let's assume the main simulation view area is the full screen for now,
        # as the solar system is vast. ModernUI can be an overlay.
        self.scale_x = SCREEN_WIDTH / WORLD_SIZE
        self.scale_y = SCREEN_HEIGHT / WORLD_SIZE
        
        # Trail storage
        self.probe_trails = {}
        
    def world_to_screen(self, world_pos):
        """Convert world coordinates to screen coordinates, centering the world"""
        # Calculate the offset from the world center
        offset_x = world_pos[0] - (WORLD_SIZE / 2)
        offset_y = world_pos[1] - (WORLD_SIZE / 2)

        # Scale the offset by zoom level and add the screen center
        screen_x = int(offset_x * self.scale_x * self.zoom_level + SCREEN_WIDTH / 2)
        screen_y = int(offset_y * self.scale_y * self.zoom_level + SCREEN_HEIGHT / 2)

        return (screen_x, screen_y)
    
    def render(self, environment, probe_agents: Dict = None):
        """Enhanced rendering with all new systems"""
        if ENABLE_PARTICLE_EFFECTS:
            self.particle_system.update() # Update all particles first

        # Clear with space background
        self.screen.fill((5, 5, 15)) # Darker space blue/black
        
        # Draw animated starfield (parallax needs camera_offset)
        if self.selected_probe_id_ui is not None and self.selected_probe_id_ui in environment.probes:
            selected_probe_world_pos = environment.probes[self.selected_probe_id_ui]['position']
            target_camera_world_center = selected_probe_world_pos
            screen_center_world_x = (SCREEN_WIDTH / 2) / self.scale_x if self.scale_x != 0 else 0
            screen_center_world_y = (SCREEN_HEIGHT / 2) / self.scale_y if self.scale_y != 0 else 0
            
            target_offset_x = target_camera_world_center[0] - screen_center_world_x
            target_offset_y = target_camera_world_center[1] - screen_center_world_y
            
            self.camera_offset[0] += (target_offset_x - self.camera_offset[0]) * 0.05
            self.camera_offset[1] += (target_offset_y - self.camera_offset[1]) * 0.05 # Moved here
        # else for camera logic if needed, or just default camera_offset if no probe selected
        
        self.starfield.draw(self.screen, self.camera_offset)
        
        self._draw_celestial_bodies(environment) # Call to draw celestial bodies
        self._draw_enhanced_resources(environment.resources)
        self._draw_enhanced_probes(environment.probes, environment.messages)
        if ENABLE_PARTICLE_EFFECTS:
            self.particle_system.render(self.screen)
        self._draw_enhanced_ui(environment, probe_agents)
        
        pygame.display.flip()
        self.clock.tick(FPS)
    
    # Definition of _draw_celestial_bodies should be at class level, not interrupting render method
    def _draw_celestial_bodies(self, environment):
        """Draws the Sun and planets."""
        if hasattr(environment, 'get_celestial_bodies_data_for_render'):
            celestial_data = environment.get_celestial_bodies_data_for_render()
            for body_data in celestial_data:
                screen_pos = self.world_to_screen(body_data['position'])
                # Scale radius by zoom_level, ensuring a minimum pixel size
                screen_radius = int(body_data['radius_sim'] * self.zoom_level)
                screen_radius = max(1, screen_radius) # Ensure at least 1 pixel radius
                pygame.draw.circle(self.screen, body_data['color'], screen_pos, screen_radius)
                # Optional: Draw name labels
                # text_surface = self.small_font.render(body_data['name'], True, (200, 200, 200))
                # text_rect = text_surface.get_rect(center=(screen_pos[0], screen_pos[1] - screen_radius - 10))
                # self.screen.blit(text_surface, text_rect)
        # Removed 'else: pass' as it's not strictly needed if the if block is self-contained

    def _draw_enhanced_resources(self, resources: List[Resource]): # Renamed
        """Draw resource nodes with enhanced visual effects"""
        for resource in resources:
            if resource.amount > 0:
                screen_pos = self.world_to_screen(resource.position)
                
                # Size based on remaining amount
                base_size = max(3, int(resource.amount / RESOURCE_MAX_AMOUNT * 15)) # type: ignore
                
                # Add resource glow effect
                # Intensity can be linked to resource amount or be a fixed value
                glow_intensity = 0.5 + (resource.amount / RESOURCE_MAX_AMOUNT) * 0.5 # type: ignore
                glow_radius = base_size * 2.5
                self.visual_effects.draw_resource_glow(self.screen, screen_pos, glow_radius, glow_intensity)

                # Draw the main resource circle on top of the glow
                pygame.draw.circle(self.screen, self.colors['resource'],
                                 screen_pos, base_size)
                
                # Draw amount text (optional, can be part of ModernUI later)
                if resource.amount > 10 and SHOW_SIMPLE_RESOURCE_INFO: # Add SHOW_SIMPLE_RESOURCE_INFO to config
                    text_content = f"{int(resource.amount)}"
                    text_surface = self.small_font.render(text_content, True, (220, 220, 220))
                    text_rect = text_surface.get_rect(center=(screen_pos[0], screen_pos[1] - base_size - 8))
                    self.screen.blit(text_surface, text_rect)
    
    def _draw_enhanced_probes(self, probes: Dict, messages: List[Message]): # Renamed
        """Draw probes using detailed renderer and advanced effects"""
        
        # Draw communication links first (as they are 'beams' between probes/targets)
        for message in messages[-10:]:  # Show recent messages
            sender_probe_data = probes.get(message.sender_id)
            if sender_probe_data and sender_probe_data['energy'] > 0:
                if message.position is None or sender_probe_data.get('position') is None:
                    continue 
                sender_pos_screen = self.world_to_screen(sender_probe_data['position'])
                target_pos_screen = self.world_to_screen(message.position) 
                
                self.visual_effects.draw_communication_beam(
                     self.screen, sender_pos_screen, target_pos_screen, intensity=0.8
                )
        
        # Draw probes
        for probe_id, probe_data in probes.items():
            current_world_pos = probe_data['position']
            screen_pos_raw = self.world_to_screen(current_world_pos)
            
            screen_pos_display = screen_pos_raw
            if probe_id in self.probe_visual_cache:
                cache = self.probe_visual_cache[probe_id]
                if 'prev_screen_pos' in cache and 'prev_velocity' in cache:
                    prev_screen_pos = cache['prev_screen_pos']
                    velocity_screen = np.array(screen_pos_raw) - prev_screen_pos
                    predicted_pos = prev_screen_pos + velocity_screen * 1.2 
                    blend_factor = 0.3 
                    interpolated_pos = (blend_factor * predicted_pos + (1 - blend_factor) * np.array(screen_pos_raw))
                    screen_pos_display = tuple(interpolated_pos.astype(int))

            if probe_id not in self.probe_visual_cache: self.probe_visual_cache[probe_id] = {}
            cache = self.probe_visual_cache[probe_id]
            if 'prev_screen_pos' in cache:
                cache['prev_velocity'] = np.array(screen_pos_raw) - cache['prev_screen_pos']
            else:
                cache['prev_velocity'] = np.array([0,0], dtype=float)
            cache['prev_screen_pos'] = np.array(screen_pos_raw)
            
            screen_pos = screen_pos_display
            is_low_power = probe_data['energy'] <= 0
            current_probe_angle_rad = probe_data.get('angle', 0.0)

            # Draw organic ship
            self.ship_renderer.draw_organic_ship(
                self.screen, probe_data, screen_pos, scale=1.0
            )
            
            # Enhanced thruster effects
            smoothing_state = probe_data.get('action_smoothing_state', {})
            thrust_ramp_value = smoothing_state.get('current_thrust_ramp', 0.0)
            if probe_data.get('is_thrusting_visual', False) and not is_low_power and thrust_ramp_value > 0.01:
                target_power_idx = probe_data.get('thrust_power_visual', 0)
                # Use organic thruster rendering
                self.ship_renderer.draw_enhanced_thruster_effects(
                    self.screen, screen_pos, current_probe_angle_rad,
                    target_power_idx, thrust_ramp_value
                )
                
                # Organic particle effects
                if ENABLE_PARTICLE_EFFECTS:
                    self.particle_system.emit_organic_thruster_exhaust(
                        screen_pos, current_probe_angle_rad,
                        target_power_idx, thrust_ramp_value
                    )

            if probe_data.get('is_mining_visual', False) and probe_data.get('mining_target_pos_visual') is not None and not is_low_power:
                mining_target_world_pos = probe_data['mining_target_pos_visual']
                mining_target_screen_pos = self.world_to_screen(mining_target_world_pos)
                
                nose_offset_dist = SPACESHIP_SIZE * 0.7 
                laser_start_x = screen_pos[0] + nose_offset_dist * math.cos(current_probe_angle_rad)
                laser_start_y = screen_pos[1] + nose_offset_dist * math.sin(current_probe_angle_rad)
                laser_start_screen_pos = (int(laser_start_x), int(laser_start_y))

                self.visual_effects.draw_mining_beam_advanced(
                    self.screen, laser_start_screen_pos, mining_target_screen_pos,
                    intensity=1.0, pulse_phase=pygame.time.get_ticks()
                )
                if ENABLE_PARTICLE_EFFECTS:
                    self.particle_system.emit_mining_sparks(
                        impact_pos=mining_target_screen_pos, intensity=1.0
                    )
            
            if probe_data['energy'] > MAX_ENERGY * 0.8 and not is_low_power:
                self.visual_effects.draw_energy_field_distortion(
                    self.screen, screen_pos, 
                    intensity_factor = (probe_data['energy'] / MAX_ENERGY if MAX_ENERGY > 0 else 0),
                    radius = SPACESHIP_SIZE * 2.0 
                )

            selected_target_info = probe_data.get('selected_target_info')
            if selected_target_info and selected_target_info.get('world_pos') is not None and not is_low_power:
                target_screen_pos = self.world_to_screen(selected_target_info['world_pos'])
                pygame.draw.line(self.screen, (200, 0, 200, 150), screen_pos, target_screen_pos, 1)

            self._update_smooth_trail(probe_id, screen_pos, probe_data.get('velocity', np.array([0.0,0.0])))

            if SHOW_SIMPLE_PROBE_INFO: 
                id_text_color = (180, 180, 180) if is_low_power else (230, 230, 230)
                id_text_content = str(probe_id)
                text_surface = self.small_font.render(id_text_content, True, id_text_color)
                text_rect = text_surface.get_rect(center=(screen_pos[0], screen_pos[1] - SPACESHIP_SIZE * 1.5)) 
                self.screen.blit(text_surface, text_rect)
            
                if not is_low_power and probe_data['energy'] < MAX_ENERGY * 0.98 : 
                    bar_width = SPACESHIP_SIZE * 1.2
                    bar_height = 3
                    bar_x = screen_pos[0] - bar_width / 2
                    bar_y = screen_pos[1] + SPACESHIP_SIZE * 1.2 
                    
                    energy_ratio = np.clip(probe_data['energy'] / MAX_ENERGY if MAX_ENERGY > 0 else 0, 0, 1)
                    energy_width = int(energy_ratio * bar_width)
                    
                    meter_bg_color = self.modern_ui.ui_colors.get('meter_bg', (40,40,60,180))
                    meter_fill_color = self.modern_ui.ui_colors.get('meter_fill', (0,150,255,220))
                    if energy_ratio < 0.3:
                        meter_fill_color = self.modern_ui.ui_colors.get('accent_red', (200,50,50,220))
                    
                    s = pygame.Surface((bar_width, bar_height), pygame.SRCALPHA)
                    s.fill((*meter_bg_color[:3], meter_bg_color[3] if len(meter_bg_color)>3 else 180))
                    if energy_width > 0:
                        pygame.draw.rect(s, (*meter_fill_color[:3], meter_fill_color[3] if len(meter_fill_color)>3 else 220), (0,0, energy_width, bar_height))
                    self.screen.blit(s, (bar_x, bar_y))

    def _update_smooth_trail(self, probe_id, screen_pos, velocity):
        """Enhanced trail system with velocity-based effects"""
        if probe_id not in self.probe_trails:
            self.probe_trails[probe_id] = []
        
        self.probe_trails[probe_id].append(screen_pos)
        max_trail_length = 80 
        
        if len(self.probe_trails[probe_id]) > max_trail_length:
            self.probe_trails[probe_id].pop(0)
        
        if len(self.probe_trails[probe_id]) > 1:
            speed = np.linalg.norm(velocity)
            speed_factor = 0.0
            if MAX_VELOCITY > 1e-6: # type: ignore
                 speed_factor = min(1.0, speed / MAX_VELOCITY) # type: ignore
            
            for i in range(1, len(self.probe_trails[probe_id])):
                alpha_base = (i / len(self.probe_trails[probe_id]))
                alpha = alpha_base ** 0.7 
                thickness = max(1, int(alpha * 3 * (0.5 + 0.5 * speed_factor)))
                
                base_color = self.colors['trail']
                trail_color_cal = [int(c * alpha * (0.3 + 0.7 * speed_factor)) for c in base_color]
                trail_color = tuple(max(0, min(255, tc_val)) for tc_val in trail_color_cal)

                if alpha > 0.1: 
                    pygame.draw.line(self.screen, trail_color,
                                   self.probe_trails[probe_id][i-1],
                                   self.probe_trails[probe_id][i], thickness)

    def _draw_enhanced_ui(self, environment, probe_agents): # Renamed
        """Draw the enhanced UI using the ModernUI system."""
        # Pass necessary data to the ModernUI system
        # The ModernUI's draw_ui method will handle all UI elements,
        # including stats, probe lists, selected probe details, alerts, etc.
        
        # Data to pass to ModernUI:
        # - environment: for general stats, probe data, resource counts, messages
        # - probe_agents: for agent-specific info if any (e.g., RL model status)
        # - selected_probe_id: self.selected_probe_id_ui
        # - camera_offset: self.camera_offset (for minimap or other spatial UI elements)
        # - clock: self.clock (for FPS display or time-based UI elements)

        # The ModernUI instance will update its internal state based on this data
        # and then draw itself to the screen.
        
        # Example of data that ModernUI might need:
        selected_probe_data = None
        if self.selected_probe_id_ui and self.selected_probe_id_ui in environment.probes:
            selected_probe_data = environment.probes[self.selected_probe_id_ui]

        self.modern_ui.update_data(
            environment_data=environment, # Contains probes, resources, messages, step_count
            probe_agents_data=probe_agents, # Optional, for agent-specific UI
            selected_probe_id=self.selected_probe_id_ui,
            selected_probe_details=selected_probe_data,
            camera_offset=self.camera_offset, # For potential minimap or spatial UI
            fps=self.clock.get_fps() # Pass current FPS
        )
        
        self.modern_ui.draw_ui(self.screen)
        
        # Old UI drawing logic is now entirely handled by ModernUI.
        # self.ui_probe_id_rects is also managed internally by ModernUI if it
        # needs to provide clickable elements.
    
    def handle_events(self):
        """Handle user input events, delegating UI events to ModernUI."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Signal to quit the main loop

            # Pass event to ModernUI for processing
            # ModernUI.handle_event is expected to return a dictionary of actions/results.
            ui_event_result = self.modern_ui.handle_event(event)

            # Check if the UI handled the event and if there's a probe selection change
            if ui_event_result:
                if 'selected_probe_id' in ui_event_result:
                    self.selected_probe_id_ui = ui_event_result['selected_probe_id']
                
                if ui_event_result.get('event_consumed', False):
                    continue

            # Handle zoom keys
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.zoom_level *= 1.2 # Zoom in
                elif event.key == pygame.K_MINUS:
                    self.zoom_level /= 1.2 # Zoom out
                self.zoom_level = max(0.0001, self.zoom_level) # Prevent zoom from becoming zero or negative

            # Mouse wheel for zoom
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4: # Scroll up, zoom in
                    self.zoom_level *= 1.1
                elif event.button == 5: # Scroll down, zoom out
                    self.zoom_level /= 1.1
                self.zoom_level = max(0.0001, self.zoom_level)


        return True # Signal to continue the main loop