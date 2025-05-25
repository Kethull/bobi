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
        
        # Update and draw trails
        for probe_id, probe in probes.items():
            # Trails are drawn for all probes, regardless of energy level, as they still exist
            screen_pos = self.world_to_screen(probe['position'])
            
            # Update trail
            if probe_id not in self.probe_trails:
                self.probe_trails[probe_id] = []
            
            self.probe_trails[probe_id].append(screen_pos)
            if len(self.probe_trails[probe_id]) > 50:
                self.probe_trails[probe_id].pop(0)
            
            # Draw trail
            if len(self.probe_trails[probe_id]) > 1:
                for i in range(1, len(self.probe_trails[probe_id])):
                    alpha = i / len(self.probe_trails[probe_id])
                    color = tuple(int(c * alpha) for c in self.colors['trail'])
                    if i > 0:
                        pygame.draw.line(self.screen, color,
                                       self.probe_trails[probe_id][i-1],
                                       self.probe_trails[probe_id][i], 1)
        
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
            screen_pos = self.world_to_screen(probe['position'])
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

            velocity_vector = probe.get('velocity', np.array([0,1])) # Default to pointing up if no velocity
            if np.linalg.norm(velocity_vector) > 0.1: # Only rotate if moving significantly
                angle_rad = math.atan2(velocity_vector[0], -velocity_vector[1]) # Angle with positive Y axis (pointing up)
            else:
                angle_rad = probe.get('last_angle_rad', -math.pi/2) # Default or last angle (pointing up)
            probe['last_angle_rad'] = angle_rad # Store for next frame if stationary
            
            rotation_matrix = np.array([
                [math.cos(angle_rad), -math.sin(angle_rad)],
                [math.sin(angle_rad), math.cos(angle_rad)]
            ])
            
            rotated_points = [rotation_matrix @ p for p in base_points]
            screen_points = [(rp[0] + screen_pos[0], rp[1] + screen_pos[1]) for rp in rotated_points]
            
            pygame.draw.polygon(self.screen, color, screen_points)
            pygame.draw.polygon(self.screen, (200, 200, 200), screen_points, 1) # Outline

            # Draw thruster flame if active
            if probe.get('is_thrusting_visual', False) and not is_low_power:
                thrust_power_level = probe.get('thrust_power_visual', 0) # 0, 1, or 2
                flame_length_factor = 0.5 + (thrust_power_level / 2.0) * 0.8 # Scale from 0.5 to 1.3 of SPACESHIP_SIZE
                flame_length = SPACESHIP_SIZE * flame_length_factor
                flame_width = SPACESHIP_SIZE * 0.4

                # Flame points relative to ship's rear center, extending "down" in ship's local coords
                # Ship's rear center is roughly (0, SPACESHIP_SIZE * 0.4)
                # Tip of flame: (0, SPACESHIP_SIZE * 0.4 + flame_length)
                # Base corners: (-flame_width/2, SPACESHIP_SIZE * 0.4), (flame_width/2, SPACESHIP_SIZE * 0.4)
                base_flame_points = [
                    np.array([0, SPACESHIP_SIZE * 0.4 + flame_length]), # Tip
                    np.array([-flame_width * 0.5, SPACESHIP_SIZE * 0.4]),  # Left base
                    np.array([flame_width * 0.5, SPACESHIP_SIZE * 0.4])   # Right base
                ]
                
                rotated_flame_points = [rotation_matrix @ p for p in base_flame_points]
                screen_flame_points = [(rp[0] + screen_pos[0], rp[1] + screen_pos[1]) for rp in rotated_flame_points]
                
                flame_color = (255, 255, 100) # Yellowish
                if thrust_power_level == 2 : # Max thrust
                    flame_color = (255,165,0) # Orange
                elif thrust_power_level == 1:
                    flame_color = (255,255,0) # Yellow
                
                pygame.draw.polygon(self.screen, flame_color, screen_flame_points)

            # Draw mining laser if active
            if probe.get('is_mining_visual', False) and probe.get('mining_target_pos_visual') is not None and not is_low_power:
                mining_target_world_pos = probe['mining_target_pos_visual']
                mining_target_screen_pos = self.world_to_screen(mining_target_world_pos)
                
                # Laser originates from the ship's nose
                # Ship's nose in local coords: np.array([0, -SPACESHIP_SIZE * 0.8])
                # This point is already part of 'base_points[0]'
                # We need its rotated and screen-translated position
                laser_origin_local = base_points[0] # Nose of the ship
                rotated_laser_origin = rotation_matrix @ laser_origin_local
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
                pygame.draw.line(self.screen, target_line_color, screen_pos, target_screen_pos, 1)
                
                # Optionally, draw a small circle at the target resource to highlight it
                pygame.draw.circle(self.screen, target_line_color, target_screen_pos, 5, 1)


            # Draw probe ID (adjust position based on spaceship size)
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
                bar_y = screen_pos[1] + SPACESHIP_SIZE * 0.5 + 5 # Below the ship
                
                pygame.draw.rect(self.screen, (100, 100, 100),
                               (bar_x, bar_y, bar_width, bar_height)) # Background
                
                energy_width = int((max(0, probe['energy']) / MAX_ENERGY) * bar_width)
                energy_color = (255, 255, 0) if probe['energy'] > 30 else (255, 100, 100)
                pygame.draw.rect(self.screen, energy_color,
                               (bar_x, bar_y, energy_width, bar_height))
    
    def _draw_ui(self, environment, probe_agents):
        """Draw UI information panel"""
        ui_x = SCREEN_WIDTH - 190
        ui_y = 10
        
        # Background panel
        pygame.draw.rect(self.screen, self.colors['ui_bg'],
                        (ui_x - 5, ui_y - 5, 185, SCREEN_HEIGHT - 10))
        
        # Statistics
        # "Alive" now means energy > 0 for UI purposes
        active_probes_count = sum(1 for probe in environment.probes.values() if probe['energy'] > 0)
        total_energy_active = sum(probe['energy'] for probe in environment.probes.values() if probe['energy'] > 0)
        avg_energy_active = total_energy_active / max(active_probes_count, 1)
        
        # Generation statistics for active probes
        generations = {}
        for probe in environment.probes.values():
            if probe['energy'] > 0: # Count only active probes for generation stats
                gen = probe['generation']
                generations[gen] = generations.get(gen, 0) + 1
        
        stats = [
            f"Step: {environment.step_count}",
            f"Active Probes: {active_probes_count}", # Changed "Alive" to "Active"
            f"Total Probes: {len(environment.probes)}",
            f"Avg Energy (Active): {avg_energy_active:.1f}", # Clarified Avg Energy
            f"Resources: {len([r for r in environment.resources if r.amount > 0])}",
            f"Messages: {len(environment.messages)}",
            "",
            "Generations:"
        ]
        
        for gen, count in sorted(generations.items()):
            stats.append(f"  Gen {gen}: {count}")
        
        # Individual probe info
        stats.append("")
        stats.append("Active Probes:")
        for probe_id, probe in environment.probes.items():
            # Display all probes, indicate low power status
            status_char = "LP" if probe['energy'] <= 0 else "OK"
            stats.append(f"  #{probe_id}: E{max(0,probe['energy']):.0f} G{probe['generation']} ({status_char})")
        
        # Render text
        y_offset = ui_y
        for stat in stats:
            text = self.small_font.render(stat, True, self.colors['ui_text'])
            self.screen.blit(text, (ui_x, y_offset))
            y_offset += 16
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True