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
            
            # Probe size based on energy (energy can be 0)
            size = max(3, int(max(0, probe['energy']) / MAX_ENERGY * 8) + 3)
            pygame.draw.circle(self.screen, color, screen_pos, size)
            pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, size, 1) # Outline
            
            # Draw probe ID
            id_text_color = (150, 150, 150) if is_low_power else (255, 255, 255)
            id_text_content = str(probe_id) + (" (LP)" if is_low_power else "")
            text = self.small_font.render(id_text_content, True, id_text_color)
            self.screen.blit(text, (screen_pos[0] + size + 2, screen_pos[1] - 8))
            
            # Draw energy bar (only if not in low power, or always draw background)
            if not is_low_power: # Or always draw and it will just be empty
                bar_width = 20
                bar_height = 4
                bar_x = screen_pos[0] - bar_width // 2
                bar_y = screen_pos[1] - size - 8
                
                # Background
                pygame.draw.rect(self.screen, (100, 100, 100),
                               (bar_x, bar_y, bar_width, bar_height))
                
                # Energy level (ensure energy_width is not negative)
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