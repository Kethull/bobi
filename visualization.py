# visualization.py
import pygame
import numpy as np
from typing import Dict, List
import math
import logging
from config import config, ConfigurationError
from environment import Resource # Added import for Resource
from environment import Message # Added import for Message
from solarsystem import CelestialBody # For type hinting and direct access
from organic_ship_renderer import OrganicShipRenderer
from particle_system import AdvancedParticleSystem
from visual_effects import VisualEffects, StarField # Added imports
from advanced_ui import ModernUI # Added import

class Visualization:
    """Handles all aspects of rendering the simulation state using Pygame.

    This class is responsible for:
    - Initializing Pygame and the display window.
    - Managing rendering subsystems: `OrganicShipRenderer` for probes,
      `AdvancedParticleSystem` for effects, `VisualEffects` for glows/beams,
      `StarField` for background, and `ModernUI` for user interface.
    - Rendering all simulation entities: celestial bodies (Sun, planets, moons),
      probes (with trails), resources, particles, and UI elements.
    - Handling user input events: window close, camera zoom (keyboard/mouse),
      and interactions with the `ModernUI` (e.g., probe selection).
    - Managing camera state: offset (pan) and zoom level.
    - Converting world coordinates to screen coordinates.

    The rendering process includes extensive error handling to log issues and
    attempt to continue or degrade gracefully if parts of the visualization fail.
    If critical initialization errors occur (e.g., Pygame display fails to set up),
    `visualization_enabled` is set to `False`, and rendering calls are skipped.

    Attributes:
        screen (pygame.Surface | None): The main Pygame display surface. `None` if
            Pygame display initialization fails.
        visualization_enabled (bool): `True` if rendering is active, `False` if
            critical initialization errors occurred or visualization is disabled.
        clock (pygame.time.Clock | None): Pygame clock for controlling FPS. `None`
            if `visualization_enabled` is `False`.
        font (pygame.font.Font | None): Default Pygame font for UI text. `None` if
            font initialization fails or `visualization_enabled` is `False`.
        small_font (pygame.font.Font | None): Smaller Pygame font. `None` if
            font initialization fails or `visualization_enabled` is `False`.
        ship_renderer (OrganicShipRenderer | None): Renderer for organic-style probes.
        particle_system (AdvancedParticleSystem | None): System for managing particle effects.
        visual_effects (VisualEffects | None): Handles various visual effects.
        starfield (StarField | None): Manages the parallax star background.
        modern_ui (ModernUI | None): Handles the advanced user interface elements.
        probe_visual_cache (Dict): Cache for probe-specific visual elements (currently unused).
        selected_probe_id_ui (int | None): ID of the probe currently selected via the UI.
        probe_trails (Dict[int, List[Tuple[int, int]]]): Stores screen positions for
            drawing probe movement trails, keyed by probe ID.
        camera_offset (np.ndarray): A 2D NumPy array `[x, y]` representing the world
            coordinates at the center of the camera's view.
        zoom_level (float): Current zoom level for the camera (1.0 is default).
        colors (Dict[str, Tuple[int, int, int]]): A dictionary of predefined RGB colors
            used for various UI elements and default entity rendering.
        scale_x (float): Calculated scale factor for converting world X coordinates to screen X.
        scale_y (float): Calculated scale factor for converting world Y coordinates to screen Y.

    Raises:
        ConfigurationError: If initialization fails due to missing or invalid values
                            in `config.Visualization` or `config.World` that are
                            critical for setting up the display or core parameters.
        pygame.error: If Pygame itself encounters critical errors during its
                      initialization (e.g., `pygame.init()`, `pygame.display.set_mode()`).
        Exception: For other unexpected errors during the initialization of
                   visualization components.
    """
    def __init__(self):
        """Initializes the Visualization system.

        Performs the following steps:
        1.  Calls `pygame.init()`.
        2.  Attempts to set up the Pygame display mode using dimensions from
            `config.Visualization.SCREEN_WIDTH_PX` and `SCREEN_HEIGHT_PX`.
            If this fails, `visualization_enabled` is set to `False`.
        3.  If the screen is successfully created:
            -   Sets the window caption.
            -   Initializes `pygame.time.Clock`.
            -   Attempts to load default and small fonts.
            -   Initializes rendering subsystems: `OrganicShipRenderer`,
                `AdvancedParticleSystem`, `VisualEffects`, `StarField`, and `ModernUI`.
                Configuration errors or other exceptions during subsystem init
                can also lead to `visualization_enabled` being set to `False`.
            -   Initializes `probe_visual_cache`, `selected_probe_id_ui`, `probe_trails`.
            -   Sets initial `camera_offset` to the center of the world
                (from `config.World.WIDTH_SIM`, `HEIGHT_SIM`).
            -   Sets initial `zoom_level` to 1.0.
            -   Defines a `colors` dictionary.
            -   Calculates `scale_x` and `scale_y` based on screen and world dimensions.
        4.  Includes extensive error handling:
            -   `ConfigurationError` is raised if critical config values are missing.
            -   `pygame.error` is caught for Pygame-specific initialization issues.
            -   General `Exception` is caught for other unexpected problems.
            In case of critical failures, `visualization_enabled` is set to `False`,
            and relevant attributes might be `None`.
        """
        try:
            pygame.init()
            self.visualization_enabled = True # Assume enabled unless an error occurs
            
            # Screen Initialization
            try:
                screen_w = config.Visualization.SCREEN_WIDTH_PX
                screen_h = config.Visualization.SCREEN_HEIGHT_PX
                if not (isinstance(screen_w, int) and screen_w > 0 and
                        isinstance(screen_h, int) and screen_h > 0):
                    raise ConfigurationError("SCREEN_WIDTH_PX and SCREEN_HEIGHT_PX must be positive integers.")
                self.screen = pygame.display.set_mode((screen_w, screen_h))
            except AttributeError as e_attr: # Config values missing
                logging.critical(f"Configuration error for screen dimensions (missing attribute): {e_attr}. Visualization disabled.", exc_info=True)
                self.screen = None
                self.visualization_enabled = False
                raise ConfigurationError(f"Missing screen dimension config: {e_attr}")
            except (pygame.error, ConfigurationError) as e_disp: # Pygame error or explicit ConfigurationError
                logging.critical(f"Error setting display mode: {e_disp}. Visualization disabled.", exc_info=True)
                self.screen = None
                self.visualization_enabled = False
                if isinstance(e_disp, ConfigurationError): raise # Re-raise if it was our specific error

            # Proceed only if screen was successfully initialized
            if self.screen and self.visualization_enabled:
                pygame.display.set_caption("Bobiverse RL Simulation - Advanced Visualization")
                self.clock = pygame.time.Clock()
                
                # Font Initialization
                try:
                    self.font = pygame.font.Font(None, 24) # Default font
                    self.small_font = pygame.font.Font(None, 16) # Smaller font
                except pygame.error as e_font:
                    logging.error(f"Pygame error initializing fonts: {e_font}. Text rendering may be impaired.", exc_info=True)
                    self.font = self.small_font = None # Allow graceful degradation
                except Exception as e_font_other: # Catch other potential font errors
                    logging.error(f"Unexpected error initializing fonts: {e_font_other}. Text rendering may be impaired.", exc_info=True)
                    self.font = self.small_font = None

                # Rendering Subsystems Initialization
                try:
                    self.ship_renderer = OrganicShipRenderer()
                    self.particle_system = AdvancedParticleSystem() # Max particles from config
                    self.visual_effects = VisualEffects()
                    self.starfield = StarField(config.Visualization.SCREEN_WIDTH_PX, config.Visualization.SCREEN_HEIGHT_PX)
                    self.modern_ui = ModernUI(config.Visualization.SCREEN_WIDTH_PX, config.Visualization.SCREEN_HEIGHT_PX)
                except ConfigurationError as e_subsystem_config: # If subsystems raise config error
                    logging.error(f"Configuration error initializing rendering subsystems: {e_subsystem_config}", exc_info=True)
                    self.visualization_enabled = False # Critical failure
                    raise
                except Exception as e_subsystem_other: # Other errors from subsystems
                    logging.error(f"Unexpected error initializing rendering subsystems: {e_subsystem_other}", exc_info=True)
                    self.visualization_enabled = False # Critical failure
                    raise # Re-raise to signal major init problem

                # State Variables
                self.probe_visual_cache: Dict = {} # Currently unused, but keep for future
                self.selected_probe_id_ui: int | None = None
                self.probe_trails: Dict[int, List[Tuple[int, int]]] = {}
                
                world_w_sim = config.World.WIDTH_SIM
                world_h_sim = config.World.HEIGHT_SIM
                if not (isinstance(world_w_sim, (int,float)) and world_w_sim > 0 and
                        isinstance(world_h_sim, (int,float)) and world_h_sim > 0) :
                    raise ConfigurationError("config.World.WIDTH_SIM and HEIGHT_SIM must be positive numbers.")
                self.camera_offset = np.array([world_w_sim / 2, world_h_sim / 2], dtype=np.float64)
                self.zoom_level: float = 1.0
                
                self.colors: Dict[str, Tuple[int, int, int]] = {
                    'background': (10, 10, 30),
                    'resource': (0, 200, 50), # Slightly adjusted resource color
                    'probe_base': (120, 180, 255),
                    'communication': (255, 255, 0),
                    'trail': (100, 100, 180),
                    'ui_text': (220, 220, 220),
                    'ui_bg': (40, 40, 60)
                }
                
                # Scaling factors (world to screen)
                self.scale_x: float = config.Visualization.SCREEN_WIDTH_PX / world_w_sim
                self.scale_y: float = config.Visualization.SCREEN_HEIGHT_PX / world_h_sim
            
            else: # Screen not initialized or visualization_enabled became false earlier
                 if self.visualization_enabled: # If it was true but screen is None
                     logging.warning("Visualization screen is None despite visualization_enabled being True. Disabling visualization.")
                 self.visualization_enabled = False
                 # Ensure attributes that might be accessed are None or default to prevent errors later
                 self.clock = self.font = self.small_font = None
                 self.ship_renderer = self.particle_system = self.visual_effects = self.starfield = self.modern_ui = None
                 self.scale_x = self.scale_y = 1.0 # Default scale to avoid division by zero if world dims were bad

        except ConfigurationError as e_config_outer: # Catch config errors raised directly in __init__
            logging.critical(f"Visualization initialization failed due to ConfigurationError: {e_config_outer}", exc_info=True)
            self.visualization_enabled = False
            raise # Re-raise to signal failure to caller
        except pygame.error as e_pygame_outer: # Catch general Pygame errors not caught by specific blocks
            logging.critical(f"A general Pygame error occurred during Visualization init: {e_pygame_outer}", exc_info=True)
            self.visualization_enabled = False
            # Optionally, re-raise or handle to prevent app from continuing if Pygame is unusable
        except Exception as e_outer: # Catch any other unexpected errors
            logging.critical(f"Unexpected critical error during Visualization initialization: {e_outer}", exc_info=True)
            self.visualization_enabled = False
            raise # Re-raise to signal critical failure

    def world_to_screen(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Converts world coordinates to screen coordinates.

        Applies camera offset (pan) and zoom level to transform a point from
        the simulation's world space to the Pygame screen's pixel space.

        Args:
            world_pos (np.ndarray): A 2D NumPy array `[x, y]` representing the
                point's coordinates in the world.

        Returns:
            Tuple[int, int]: The corresponding `(x, y)` screen coordinates as integers.
                             If visualization is disabled or an error occurs, returns
                             `(0,0)` or the screen center as a fallback.

        Error Handling:
            - Returns `(0,0)` if `self.visualization_enabled` is `False` or `self.screen` is `None`.
            - Catches `TypeError` (e.g., if `world_pos` is not valid), `AttributeError`
              (e.g., if `config` values are missing or attributes like `camera_offset`
              are not initialized), or `IndexError` (if `world_pos` is not 2D).
            - Logs errors and returns a default screen position (center or `(0,0)`)
              to prevent crashes in rendering loops.
        """
        if not self.visualization_enabled or self.screen is None:
            return (0, 0) # Cannot perform conversion
        try:
            # Validate inputs and attributes
            if not isinstance(world_pos, np.ndarray) or world_pos.shape != (2,):
                raise TypeError(f"world_pos must be a 2D NumPy array. Got: {world_pos}")
            if not (isinstance(self.camera_offset, np.ndarray) and self.camera_offset.shape == (2,)):
                 raise AttributeError("camera_offset is not a valid 2D NumPy array.")
            if not isinstance(self.zoom_level, (int, float)) or self.zoom_level <= 0:
                # Zoom can be very small, but not zero or negative.
                # Depending on desired effect, a very small positive zoom might be okay.
                # For now, let's assume > 0.
                raise ValueError(f"zoom_level must be a positive number. Got: {self.zoom_level}")

            # Perform transformation
            # 1. Translate world position relative to camera's world position
            relative_world_coords = world_pos - self.camera_offset
            
            # 2. Scale by zoom level and then by world-to-screen scale
            # Note: scale_x and scale_y are pixels per world unit.
            scaled_coords_x = relative_world_coords[0] * self.zoom_level * self.scale_x
            scaled_coords_y = relative_world_coords[1] * self.zoom_level * self.scale_y
            
            # 3. Translate to screen origin (center of screen is 0,0 in this relative coord system)
            #    then add screen center to get final screen coordinates.
            screen_center_x = config.Visualization.SCREEN_WIDTH_PX / 2
            screen_center_y = config.Visualization.SCREEN_HEIGHT_PX / 2
            
            screen_x = int(screen_center_x + scaled_coords_x)
            screen_y = int(screen_center_y + scaled_coords_y) # Pygame Y is often top-down, ensure consistency if needed
            
            return (screen_x, screen_y)

        except (TypeError, ValueError, AttributeError, IndexError) as e_transform:
            logging.error(f"Error in world_to_screen (world_pos: {world_pos}, zoom: {self.zoom_level}): {e_transform}", exc_info=True)
            # Fallback to screen center if possible, else (0,0)
            try:
                return (config.Visualization.SCREEN_WIDTH_PX // 2, config.Visualization.SCREEN_HEIGHT_PX // 2)
            except AttributeError: # If config itself is broken
                return (0,0)
        except Exception as e_unexpected: # Catch any other unexpected errors
            logging.error(f"Unexpected error in world_to_screen: {e_unexpected}", exc_info=True)
            return (0,0)


    def render(self, environment: 'SolarSystemEnvironment', probe_agents: Dict[int, 'ProbeAgent'] = None):
        """Renders the current state of the simulation.

        This is the main rendering method called in each simulation step by
        `BobiverseSimulation.run_episode()`. It orchestrates the drawing of all
        visual elements if `self.visualization_enabled` is `True`.

        The rendering sequence is typically:
        1.  Update particle systems (if enabled).
        2.  Fill the screen with the background color.
        3.  Adjust camera position to follow a selected probe (if any).
        4.  Draw the starfield (parallax background).
        5.  Draw celestial bodies (Sun, planets, moons) and their orbit paths.
        6.  Draw resource nodes with visual enhancements.
        7.  Draw probes (using `OrganicShipRenderer`), their trails, and communication effects.
        8.  Render active particle effects.
        9.  Draw the `ModernUI` (info panels, buttons, etc.).
        10. Flip the Pygame display to show the rendered frame.
        11. Tick the Pygame clock to maintain the target FPS.

        Args:
            environment (SolarSystemEnvironment): The instance of the simulation
                environment, providing data for all entities to be rendered.
            probe_agents (Dict[int, ProbeAgent], optional): A dictionary mapping
                probe IDs to their `ProbeAgent` instances. Used by the UI to display
                agent-specific information. Defaults to `None` if not provided.

        Error Handling:
            - Skips rendering entirely if `visualization_enabled` is `False` or `screen` is `None`.
            - Catches `pygame.error`, `AttributeError` (e.g., from misconfigured or
              uninitialized components like `particle_system` or `modern_ui`), and
              other general `Exception` types that might occur during the render loop.
            - Logs errors and attempts to continue the simulation, though visualization
              might become unstable or incomplete if errors persist.
        """
        if not self.visualization_enabled or self.screen is None:
            return # Skip rendering if disabled or screen not available

        try:
            # 1. Update particle systems (if enabled and system exists)
            if config.Visualization.ENABLE_PARTICLE_EFFECTS and self.particle_system:
                try:
                    self.particle_system.update()
                except Exception as e_particle_update:
                    logging.error(f"Error updating particle system: {e_particle_update}", exc_info=True)
                    # Potentially disable particle effects or this specific system if errors persist

            # 2. Fill screen
            self.screen.fill(self.colors.get('background', (10, 10, 30))) # Fallback color
            
            # 3. Camera follow logic
            if self.selected_probe_id_ui is not None and \
               hasattr(environment, 'probes') and self.selected_probe_id_ui in environment.probes:
                selected_probe_data = environment.probes[self.selected_probe_id_ui]
                if 'position' in selected_probe_data and isinstance(selected_probe_data['position'], np.ndarray):
                    target_camera_pos = selected_probe_data['position']
                    # Smooth camera movement (Lerp)
                    lerp_factor = 0.05
                    self.camera_offset += (target_camera_pos - self.camera_offset) * lerp_factor
            
            # 4. Draw Starfield (if system exists)
            if self.starfield:
                try:
                    self.starfield.draw(self.screen, self.camera_offset)
                except Exception as e_starfield:
                    logging.error(f"Error drawing starfield: {e_starfield}", exc_info=True)
            
            # 5. Draw Celestial Bodies
            self._draw_celestial_bodies(environment)
            
            # 6. Draw Resources (if they exist in environment)
            if hasattr(environment, 'resources'):
                self._draw_enhanced_resources(environment.resources)
            
            # 7. Draw Probes and Messages (if they exist in environment)
            if hasattr(environment, 'probes') and hasattr(environment, 'messages'):
                 self._draw_enhanced_probes(environment.probes, environment.messages)

            # 8. Render active particle effects (if enabled and system exists)
            if config.Visualization.ENABLE_PARTICLE_EFFECTS and self.particle_system:
                try:
                    self.particle_system.render(self.screen)
                except Exception as e_particle_render:
                    logging.error(f"Error rendering particle system: {e_particle_render}", exc_info=True)
            
            # 9. Draw ModernUI (if system exists)
            if self.modern_ui:
                self._draw_enhanced_ui(environment, probe_agents if probe_agents else {})
            
            # 10. Flip display
            pygame.display.flip()
            
            # 11. Tick clock (if clock exists)
            if self.clock:
                self.clock.tick(config.Visualization.FPS if config.Visualization.FPS > 0 else 60) # Ensure FPS > 0

        except pygame.error as e_pygame_render:
            logging.error(f"Pygame error during main render loop: {e_pygame_render}. Attempting to continue.", exc_info=True)
        except AttributeError as e_attr_render:
            # This could happen if a component (e.g., particle_system, modern_ui) was None due to init failure
            logging.error(f"AttributeError during render: {e_attr_render}. A rendering component might be uninitialized or misconfigured.", exc_info=True)
        except Exception as e_render_critical:
            logging.critical(f"Unexpected critical error in main render loop: {e_render_critical}. Visualization may be unstable.", exc_info=True)
            # Consider setting self.visualization_enabled = False if errors are too severe or frequent.

    def _draw_celestial_bodies(self, environment: 'SolarSystemEnvironment'):
        """Draws all celestial bodies (Sun, planets, moons) and their orbit paths.

        Iterates through the `environment.sun` and `environment.planets` (which
        includes moons). For each body:
        - Converts its world position to screen coordinates using `world_to_screen()`.
        - Draws its orbit path if available (list of screen points).
        - Draws the body as a circle with its configured color and `display_radius_sim`,
          scaled by the current zoom level. Radius is clamped to a minimum of 1 pixel.
        - If zoom level and screen radius are sufficient, renders the body's name
          as text near it.

        Args:
            environment (SolarSystemEnvironment): The simulation environment instance,
                providing access to `sun` and `planets` attributes.

        Error Handling:
            - Skips rendering if `visualization_enabled` is `False` or `screen` is `None`.
            - Catches `pygame.error` for Pygame-specific drawing issues.
            - Catches `AttributeError` if `environment` or celestial bodies are missing
              expected attributes (e.g., `position_sim`, `color`, `orbit_path`).
            - Catches other `Exception` types for unexpected issues.
            - Logs errors and attempts to continue rendering other bodies.
            - Skips individual bodies if they lack essential attributes for rendering.
        """
        if not self.visualization_enabled or self.screen is None: return
        try:
            bodies_to_render: List[CelestialBody] = []
            if hasattr(environment, 'sun') and environment.sun:
                bodies_to_render.append(environment.sun)
            if hasattr(environment, 'planets') and isinstance(environment.planets, list):
                 bodies_to_render.extend(environment.planets)

            for body in bodies_to_render:
                # Validate body object and its essential attributes
                if not isinstance(body, CelestialBody) or \
                   not all(hasattr(body, attr) for attr in ['position_sim', 'color', 'display_radius_sim', 'name']):
                    body_name = getattr(body, 'name', 'UnknownBody')
                    logging.warning(f"Celestial body '{body_name}' is malformed or missing essential attributes for rendering. Skipping.")
                    continue
                
                # Draw orbit path
                if hasattr(body, 'orbit_path') and isinstance(body.orbit_path, list) and body.orbit_path:
                    try:
                        # Filter out None points just in case, though orbit_path should be clean
                        screen_orbit_points = [self.world_to_screen(point) for point in body.orbit_path if point is not None]
                        if len(screen_orbit_points) > 1:
                            pygame.draw.lines(self.screen, (80, 80, 80), False, screen_orbit_points, 1) # Dimmer orbit path
                    except Exception as e_orbit_draw: # Catch errors during path drawing for one body
                        logging.error(f"Error drawing orbit path for {body.name}: {e_orbit_draw}", exc_info=True)
                
                # Draw body itself
                screen_pos = self.world_to_screen(body.position_sim)
                
                # Calculate screen radius, ensuring positive scale factors
                effective_scale_x = self.scale_x if self.scale_x > 0 else 1.0
                effective_scale_y = self.scale_y if self.scale_y > 0 else 1.0
                # Use an average or min scale if they differ significantly, or ensure aspect ratio is maintained
                avg_scale = (effective_scale_x + effective_scale_y) / 2.0
                
                screen_radius_float = body.display_radius_sim * avg_scale * self.zoom_level
                screen_radius = max(1, int(screen_radius_float)) # Ensure radius is at least 1 pixel
                
                pygame.draw.circle(self.screen, body.color, screen_pos, screen_radius)

                # Draw name label if conditions met
                if self.zoom_level > 0.005 and screen_radius > 3 and self.small_font: # Check font exists
                    try:
                        text_surface = self.small_font.render(body.name, True, self.colors.get('ui_text', (220,220,220)))
                        text_rect = text_surface.get_rect(center=(screen_pos[0], screen_pos[1] - screen_radius - 10)) # Position above body
                        self.screen.blit(text_surface, text_rect)
                    except pygame.error as e_font_render:
                        logging.error(f"Pygame font error rendering label for {body.name}: {e_font_render}", exc_info=True)
                        # Don't let font error stop other rendering; small_font might have become None

        except pygame.error as e_pygame_bodies:
            logging.error(f"Pygame error during _draw_celestial_bodies: {e_pygame_bodies}", exc_info=True)
        except AttributeError as e_attr_bodies:
            logging.error(f"AttributeError in _draw_celestial_bodies (environment or body missing attributes?): {e_attr_bodies}", exc_info=True)
        except Exception as e_bodies_unexpected:
            logging.error(f"Unexpected error in _draw_celestial_bodies: {e_bodies_unexpected}", exc_info=True)


    def _draw_enhanced_resources(self, resources: List[Resource]):
        """Draws resource nodes with visual enhancements like glows.

        Iterates through the provided list of `Resource` objects. For each resource:
        - If its `amount` is positive, calculates its screen position and display size.
          Size is proportional to `resource.amount` relative to `config.Resource.MAX_AMOUNT`,
          scaled by `self.zoom_level`, and clamped to a minimum pixel size.
        - Uses `self.visual_effects.draw_resource_glow()` to render a glow effect.
        - Draws the resource as a circle.
        - If `config.Visualization.SHOW_SIMPLE_RESOURCE_INFO` is `True` and the
          resource amount is significant, displays the amount as text.

        Args:
            resources (List[Resource]): A list of `Resource` objects from the
                simulation environment.

        Error Handling:
            - Skips rendering if `visualization_enabled` is `False` or `screen` is `None`.
            - Catches `pygame.error` for Pygame-specific drawing issues.
            - Catches `AttributeError` if `resources` or individual resource objects
              are missing expected attributes (e.g., `amount`, `position`), or if
              `config.Resource` or `config.Visualization` values are missing.
            - Catches other `Exception` types for unexpected issues.
            - Logs errors and attempts to continue rendering other resources.
            - Skips individual resources if they are malformed.
        """
        if not self.visualization_enabled or self.screen is None: return
        try:
            resource_color = self.colors.get('resource', (0, 200, 50))
            text_color = self.colors.get('ui_text', (220, 220, 220))

            for resource_node in resources:
                if not isinstance(resource_node, Resource) or \
                   not all(hasattr(resource_node, attr) for attr in ['amount', 'position', 'max_amount']):
                    logging.warning(f"Resource object is malformed or missing attributes. Skipping: {resource_node}")
                    continue
                
                if resource_node.amount > 0:
                    screen_pos = self.world_to_screen(resource_node.position)
                    
                    min_display_px = 2
                    # Use resource_node.max_amount if available and valid, else config fallback
                    effective_max_amount = resource_node.max_amount if resource_node.max_amount > 0 else config.Resource.MAX_AMOUNT
                    if effective_max_amount <= 0: effective_max_amount = 1.0 # Avoid division by zero

                    # Size based on current amount relative to its own max or config max
                    size_ratio = np.clip(resource_node.amount / effective_max_amount, 0.0, 1.0)
                    base_render_size_sim_units = 15 # Arbitrary base size in sim units for max amount
                    
                    # Calculate screen size based on sim units, then scale by zoom
                    # This requires knowing how many pixels 'base_render_size_sim_units' corresponds to
                    # Assuming scale_x/y are pixels per sim_unit
                    avg_pixel_per_sim_unit_scale = (self.scale_x + self.scale_y) / 2.0
                    base_screen_size_no_zoom = base_render_size_sim_units * avg_pixel_per_sim_unit_scale * size_ratio
                    
                    current_screen_size = max(min_display_px, int(base_screen_size_no_zoom * self.zoom_level))
                    
                    # Draw glow (if visual_effects system exists)
                    if self.visual_effects:
                        try:
                            glow_intensity_val = 0.4 + (size_ratio * 0.6) # More intense for fuller resources
                            glow_radius_val = current_screen_size * 2.0 # Glow larger than the object
                            self.visual_effects.draw_resource_glow(self.screen, screen_pos, glow_radius_val, glow_intensity_val)
                        except Exception as e_glow:
                             logging.error(f"Error drawing resource glow for {resource_node.position}: {e_glow}", exc_info=True)
                    
                    # Draw resource circle
                    pygame.draw.circle(self.screen, resource_color, screen_pos, current_screen_size)
                    
                    # Draw amount text (if font exists and conditions met)
                    if resource_node.amount > 10 and config.Visualization.SHOW_SIMPLE_RESOURCE_INFO and self.small_font:
                        try:
                            text_content = f"{int(resource_node.amount)}"
                            text_surface = self.small_font.render(text_content, True, text_color)
                            text_rect = text_surface.get_rect(center=(screen_pos[0], screen_pos[1] - current_screen_size - 8))
                            self.screen.blit(text_surface, text_rect)
                        except pygame.error as e_font_res:
                             logging.error(f"Pygame font error rendering resource amount for {resource_node.position}: {e_font_res}", exc_info=True)
        
        except pygame.error as e_pygame_res:
            logging.error(f"Pygame error during _draw_enhanced_resources: {e_pygame_res}", exc_info=True)
        except AttributeError as e_attr_res:
             logging.error(f"AttributeError in _draw_enhanced_resources (check config or resource attributes): {e_attr_res}", exc_info=True)
        except Exception as e_res_unexpected:
            logging.error(f"Unexpected error in _draw_enhanced_resources: {e_res_unexpected}", exc_info=True)

    
    def _draw_enhanced_probes(self, probes: Dict[int, Dict], messages: List[Message]):
        """Draws probes with enhanced visuals, trails, and communication effects.

        Iterates through the `probes` dictionary. For each probe:
        - If `config.Visualization.ORGANIC_SHIP_ENABLED` is `True` and `self.ship_renderer`
          exists, renders the probe using `OrganicShipRenderer`.
        - Otherwise (or as a fallback), could draw a simple shape (currently relies on organic).
        - Updates and draws the probe's movement trail via `_update_smooth_trail()`.
        - If `config.Visualization.SHOW_SIMPLE_PROBE_INFO` is `True`, displays basic
          info like probe ID.
        - Renders particle effects for thrusters, mining, etc., if enabled and
          `self.particle_system` exists (particle emission logic would be in `ProbeAgent`
          or `Environment`, visualization just renders them).

        Also iterates through recent `messages` to draw communication beams between
        sender probes and message target positions using `self.visual_effects`.

        Args:
            probes (Dict[int, Dict]): A dictionary where keys are probe IDs and values
                are dictionaries containing probe data (e.g., 'position', 'energy',
                'velocity', 'angle_rad', 'generation', 'is_thrusting', etc.).
            messages (List[Message]): A list of `Message` objects currently active
                in the environment.

        Error Handling:
            - Skips rendering if `visualization_enabled` is `False` or `screen` is `None`.
            - Cleans up trails for probes that no longer exist in the `probes` dict.
            - Catches `pygame.error`, `KeyError` (if probe data is malformed),
              `AttributeError` (e.g., missing config values or uninitialized renderers),
              and other `Exception` types.
            - Logs errors and attempts to continue rendering other probes/effects.
            - Skips individual probes or messages if their data is malformed or
              essential rendering components are missing/fail.
        """
        if not self.visualization_enabled or self.screen is None: return
        try:
            # 1. Clean up trails for probes that no longer exist
            active_probe_ids = set(probes.keys())
            stale_trail_ids = [pid for pid in self.probe_trails if pid not in active_probe_ids]
            for pid_stale in stale_trail_ids:
                if pid_stale in self.probe_trails:
                    del self.probe_trails[pid_stale]
            
            # 2. Draw communication beams from recent messages (if visual_effects exists)
            if self.visual_effects:
                try:
                    # Iterate over a slice of recent messages to avoid too many beams
                    for msg in messages[-config.Visualization.MAX_MESSAGES_TO_DISPLAY_BEAMS if hasattr(config.Visualization, 'MAX_MESSAGES_TO_DISPLAY_BEAMS') else 5:]:
                        if not isinstance(msg, Message) or not all(hasattr(msg, attr) for attr in ['sender_id', 'position']):
                            continue # Skip malformed messages
                        
                        sender_data = probes.get(msg.sender_id)
                        if sender_data and sender_data.get('alive', False) and 'position' in sender_data:
                            sender_screen_pos = self.world_to_screen(sender_data['position'])
                            target_screen_pos = self.world_to_screen(msg.position)
                            self.visual_effects.draw_communication_beam(self.screen, sender_screen_pos, target_screen_pos, intensity=0.7)
                except Exception as e_comm_beam:
                    logging.error(f"Error drawing communication beams: {e_comm_beam}", exc_info=True)

            # 3. Draw probes
            for probe_id, probe_data_map in probes.items():
                if not isinstance(probe_data_map, dict) or \
                   not all(k in probe_data_map for k in ['position', 'energy', 'alive']): # Check essential keys
                    logging.warning(f"Probe {probe_id} data is malformed or missing essential keys. Skipping rendering.")
                    continue
                
                if not probe_data_map.get('alive', False): # Don't render dead probes
                    if probe_id in self.probe_trails: del self.probe_trails[probe_id] # Clean trail if dead
                    continue

                world_pos_probe = probe_data_map['position']
                screen_pos_probe = self.world_to_screen(world_pos_probe)
                
                # Render probe body (e.g., organic ship)
                if config.Visualization.ORGANIC_SHIP_ENABLED and self.ship_renderer:
                    try:
                        # Scale can be dynamic based on zoom or probe state
                        effective_render_scale = np.clip(self.zoom_level * config.Visualization.SHIP_SCALE_FACTOR, 0.3, 2.5)
                        self.ship_renderer.draw_organic_ship(self.screen, probe_data_map, screen_pos_probe, scale=effective_render_scale)
                    except Exception as e_ship_render:
                        logging.error(f"Error rendering organic ship for probe {probe_id}: {e_ship_render}", exc_info=True)
                else: # Fallback simple probe drawing if organic is off or renderer fails
                    pygame.draw.circle(self.screen, self.colors.get('probe_base', (100,150,255)), screen_pos_probe, int(config.Visualization.PROBE_SIZE_PX / 2 * self.zoom_level))

                # Update and draw trail
                self._update_smooth_trail(probe_id, screen_pos_probe, probe_data_map.get('velocity', np.array([0.0,0.0])))

                # Draw simple info text (if font exists and enabled)
                if config.Visualization.SHOW_SIMPLE_PROBE_INFO and self.small_font:
                    try:
                        info_text = f"ID:{probe_id} E:{probe_data_map.get('energy', 0.0):.0f}"
                        text_surf = self.small_font.render(info_text, True, self.colors.get('ui_text', (220,220,220)))
                        self.screen.blit(text_surf, (screen_pos_probe[0] + 10, screen_pos_probe[1] - 20))
                    except pygame.error as e_font_probe:
                        logging.error(f"Pygame font error rendering info for probe {probe_id}: {e_font_probe}", exc_info=True)
                
                # Particle effects for thrusters, mining, etc.
                # (Actual emission logic should be tied to probe actions in Environment/ProbeAgent,
                # here we just tell particle_system to emit if flags are set in probe_data)
                if config.Visualization.ENABLE_PARTICLE_EFFECTS and self.particle_system:
                    try:
                        if probe_data_map.get('is_thrusting', False):
                             # Example: self.particle_system.emit_thrust_particles(screen_pos_probe, probe_data_map.get('angle_rad',0))
                            pass # Actual call would depend on particle system API
                        if probe_data_map.get('is_mining', False):
                            # Example: self.particle_system.emit_mining_particles(screen_pos_probe, target_resource_pos_screen)
                            pass
                    except Exception as e_probe_particles:
                        logging.error(f"Error emitting particles for probe {probe_id}: {e_probe_particles}", exc_info=True)


        except pygame.error as e_pygame_probes:
            logging.error(f"Pygame error during _draw_enhanced_probes: {e_pygame_probes}", exc_info=True)
        except KeyError as e_key_probes: # If probe_data_map is missing expected keys
            logging.error(f"KeyError in _draw_enhanced_probes (probe data malformed?): {e_key_probes}", exc_info=True)
        except AttributeError as e_attr_probes: # e.g., self.ship_renderer is None
             logging.error(f"AttributeError in _draw_enhanced_probes (check config or uninitialized components): {e_attr_probes}", exc_info=True)
        except Exception as e_probes_unexpected:
            logging.error(f"Unexpected error in _draw_enhanced_probes: {e_probes_unexpected}", exc_info=True)


    def _update_smooth_trail(self, probe_id: int, screen_pos: Tuple[int, int], velocity: np.ndarray):
        """Updates and draws a smooth visual trail for a given probe.

        Appends the probe's current `screen_pos` to its trail list in
        `self.probe_trails`. The trail's length is limited by
        `config.Visualization.MAX_PROBE_TRAIL_POINTS`.
        The trail is rendered as a series of connected line segments. Older segments
        can be made fainter or thinner to create a fading effect.

        Args:
            probe_id (int): The unique ID of the probe.
            screen_pos (Tuple[int, int]): The current screen (pixel) coordinates
                `(x, y)` of the probe.
            velocity (np.ndarray): The current velocity vector `[vx, vy]` of the
                probe in world units. (Currently unused in trail drawing logic but
                available for future enhancements like velocity-dependent trail style).

        Error Handling:
            - Skips if `visualization_enabled` is `False` or `screen` is `None`.
            - Catches `pygame.error` for Pygame-specific drawing issues.
            - Catches `AttributeError` if `config.Visualization.MAX_PROBE_TRAIL_POINTS`
              or `self.colors['trail']` are missing.
            - Catches other `Exception` types for unexpected issues.
            - Logs errors and attempts to continue. If a line segment fails to draw,
              it breaks drawing for that specific trail to prevent repeated errors.
        """
        if not self.visualization_enabled or self.screen is None: return
        try:
            # Initialize trail list for the probe if it doesn't exist
            if probe_id not in self.probe_trails:
                self.probe_trails[probe_id] = []
            
            self.probe_trails[probe_id].append(screen_pos) # Add current position
            
            # Limit trail length
            max_points = config.Visualization.MAX_PROBE_TRAIL_POINTS
            if not isinstance(max_points, int) or max_points <= 0:
                logging.warning(f"MAX_PROBE_TRAIL_POINTS ({max_points}) invalid. Defaulting to 50.")
                max_points = 50 # Fallback
            
            while len(self.probe_trails[probe_id]) > max_points:
                self.probe_trails[probe_id].pop(0) # Remove oldest point
            
            # Draw the trail if it has at least two points
            current_trail_points = self.probe_trails[probe_id]
            if len(current_trail_points) > 1:
                trail_base_color = self.colors.get('trail', (80, 80, 150)) # Fallback color
                num_segments = len(current_trail_points) - 1

                for i in range(num_segments):
                    p_start = current_trail_points[i]
                    p_end = current_trail_points[i+1]
                    
                    # Calculate alpha and thickness based on segment age (i)
                    # Older segments (smaller i) are more transparent and thinner
                    age_ratio = (i + 1) / (num_segments + 1) # Ratio from ~0 (oldest visible) to ~1 (newest)
                    
                    alpha_val = int(50 + age_ratio * 150) # Alpha from 50 to 200
                    alpha_val = np.clip(alpha_val, 0, 255)
                    
                    # Thickness can also vary, e.g., thicker for newer segments
                    thickness_val = max(1, int(1 + age_ratio * 2)) # Thickness from 1 to 3
                                        
                    # Pygame's basic draw.line doesn't directly support alpha.
                    # For true alpha, one would draw to a separate transparent surface.
                    # Here, we can simulate fading by adjusting color towards background,
                    # or just use a solid color. For simplicity, using solid color.
                    # If a more advanced effect is needed, this part needs rework.
                    # Example: color = (trail_base_color[0], trail_base_color[1], trail_base_color[2], alpha_val)
                    # then blit with special_flags=pygame.BLEND_RGBA_MULT or similar.
                    
                    # Using a slightly modified color for older segments if no true alpha
                    # color_intensity_factor = 0.5 + age_ratio * 0.5
                    # segment_color = (
                    #     int(trail_base_color[0] * color_intensity_factor),
                    #     int(trail_base_color[1] * color_intensity_factor),
                    #     int(trail_base_color[2] * color_intensity_factor)
                    # )
                    segment_color = trail_base_color # Keep it simple for now

                    try:
                        pygame.draw.line(self.screen, segment_color, p_start, p_end, thickness_val)
                    except pygame.error as e_draw_line:
                        logging.error(f"Pygame error drawing trail segment for probe {probe_id} between {p_start} and {p_end}: {e_draw_line}", exc_info=True)
                        break # Stop drawing this specific trail if a segment fails

        except pygame.error as e_pygame_trail:
            logging.error(f"Pygame error in _update_smooth_trail for probe {probe_id}: {e_pygame_trail}", exc_info=True)
        except AttributeError as e_attr_trail: # e.g., config.Visualization or self.colors missing
             logging.error(f"AttributeError in _update_smooth_trail for probe {probe_id} (check config or colors dict): {e_attr_trail}", exc_info=True)
        except Exception as e_trail_unexpected:
            logging.error(f"Unexpected error in _update_smooth_trail for probe {probe_id}: {e_trail_unexpected}", exc_info=True)


    def _draw_enhanced_ui(self, environment: 'SolarSystemEnvironment', probe_agents: Dict[int, 'ProbeAgent']):
        """Draws the enhanced user interface using the `ModernUI` component.

        This method is responsible for feeding current simulation data to the
        `ModernUI` instance and then telling it to render itself.

        The data passed to `ModernUI.update_data()` includes:
        -   `environment_data`: The entire `SolarSystemEnvironment` instance.
        -   `probe_agents_data`: The dictionary of `ProbeAgent` instances.
        -   `selected_probe_id`: The ID of the probe currently selected in the UI.
        -   `selected_probe_details`: Detailed data for the selected probe.
        -   `camera_offset`: Current camera world coordinates.
        -   `fps`: Current frames per second.

        Args:
            environment (SolarSystemEnvironment): The current simulation environment.
            probe_agents (Dict[int, ProbeAgent]): Dictionary of active `ProbeAgent`
                instances, keyed by probe ID.

        Error Handling:
            - Skips if `visualization_enabled` is `False`, `screen` is `None`,
              `self.modern_ui` is `None` (e.g., due to init failure), or `self.clock`
              is `None`.
            - Catches `pygame.error` for Pygame-specific drawing issues within the UI.
            - Catches `AttributeError` if `ModernUI` or its dependencies are
              misconfigured or if expected data (e.g., from `environment` or
              `probe_agents`) is missing.
            - Catches other `Exception` types for unexpected issues during UI rendering.
            - Logs errors and attempts to continue the simulation.
        """
        if not self.visualization_enabled or self.screen is None or \
           not self.modern_ui or not self.clock: # Check all necessary components
            return
        
        try:
            selected_probe_details_data = None
            if self.selected_probe_id_ui is not None and \
               hasattr(environment, 'probes') and isinstance(environment.probes, dict) and \
               self.selected_probe_id_ui in environment.probes:
                selected_probe_details_data = environment.probes.get(self.selected_probe_id_ui)

            # Ensure probe_agents is a dict, even if empty
            current_probe_agents = probe_agents if isinstance(probe_agents, dict) else {}

            self.modern_ui.update_data(
                environment_data=environment, # Pass the whole environment
                probe_agents_data=current_probe_agents,
                selected_probe_id=self.selected_probe_id_ui,
                selected_probe_details=selected_probe_details_data, # Can be None
                camera_offset=self.camera_offset,
                zoom_level=self.zoom_level, # Pass zoom level to UI
                fps=self.clock.get_fps() # get_fps() is safe even if clock just started
            )
            self.modern_ui.draw_ui(self.screen) # ModernUI handles its own drawing logic

        except pygame.error as e_pygame_ui:
            logging.error(f"Pygame error during ModernUI drawing: {e_pygame_ui}", exc_info=True)
        except AttributeError as e_attr_ui: # e.g., environment missing 'probes', or modern_ui methods fail
            logging.error(f"AttributeError in _draw_enhanced_ui (check ModernUI, environment, or config): {e_attr_ui}", exc_info=True)
        except Exception as e_ui_unexpected:
            logging.error(f"Unexpected error in _draw_enhanced_ui: {e_ui_unexpected}", exc_info=True)

    
    def handle_events(self) -> bool:
        """Handles Pygame events like window close, keyboard input, and mouse actions.

        This method is called once per simulation step (if rendering is active)
        to process the Pygame event queue. It manages:
        -   **QUIT Event**: If the user closes the window, returns `False` to signal
            the main simulation loop to terminate.
        -   **UI Interaction**: Passes events to `self.modern_ui.handle_event()`.
            If the UI consumes an event (e.g., a button click), further processing
            of that event by this method might be skipped. UI interactions can
            update `self.selected_probe_id_ui`.
        -   **Camera Zoom**:
            -   Keyboard: `+`/`=` keys zoom in, `-` key zooms out.
            -   Mouse Wheel: Scroll up zooms in, scroll down zooms out.
            Zoom level is clamped to a sensible range defined by
            `config.Visualization.MIN_ZOOM` and `MAX_ZOOM` (or internal defaults).

        Returns:
            bool: `False` if a `pygame.QUIT` event is received, indicating the
                  simulation should stop. `True` otherwise, allowing the simulation
                  to continue.

        Error Handling:
            - If `visualization_enabled` is `False`:
                - It still attempts minimal event polling for `pygame.QUIT` if Pygame
                  was initialized enough to have an event queue. This allows headless
                  runs to potentially be quit via signals if Pygame captures them.
                - If `pygame.event.get()` fails (e.g., display never fully initialized),
                  it ignores the error and returns `True` to let a headless simulation continue.
            - Catches `pygame.error` for Pygame-specific issues during event processing.
            - Catches other `Exception` types for unexpected problems.
            - Logs errors and returns `True` to attempt to continue the simulation,
              as event handling errors are usually not critical enough to stop everything.
        """
        # If visualization is off, only poll for QUIT if Pygame was somewhat initialized.
        if not self.visualization_enabled:
            try:
                for event in pygame.event.get(): # Minimal polling
                    if event.type == pygame.QUIT:
                        logging.info("QUIT event received (visualization was disabled). Signaling shutdown.")
                        return False # Signal main loop to stop
            except pygame.error: # If pygame.display never initialized, event.get() can fail
                pass # Ignore, assume truly headless and no interactive quit possible this way
            return True # Allow simulation to continue (e.g., headless training run)

        # Full event handling if visualization is enabled
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logging.info("QUIT event received via Pygame window. Signaling shutdown.")
                    return False # Signal main loop to stop

                # Pass event to ModernUI first. It might consume it.
                ui_consumed_event = False
                if self.modern_ui: # Check if modern_ui was initialized
                    try:
                        ui_event_response = self.modern_ui.handle_event(event)
                        if ui_event_response: # If UI provides a response
                            if 'selected_probe_id' in ui_event_response:
                                self.selected_probe_id_ui = ui_event_response['selected_probe_id']
                            if ui_event_response.get('event_consumed', False):
                                ui_consumed_event = True
                    except Exception as e_ui_event:
                        logging.error(f"Error during ModernUI event handling: {e_ui_event}", exc_info=True)
                
                if ui_consumed_event:
                    continue # UI handled it, no further processing for this event

                # Camera zoom controls (if not consumed by UI)
                min_zoom = getattr(config.Visualization, 'MIN_ZOOM', 0.0001)
                max_zoom = getattr(config.Visualization, 'MAX_ZOOM', 100.0)

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.zoom_level *= 1.2
                    elif event.key == pygame.K_MINUS:
                        self.zoom_level /= 1.2
                    self.zoom_level = np.clip(self.zoom_level, min_zoom, max_zoom)

                if event.type == pygame.MOUSEBUTTONDOWN: # Mouse wheel for zoom
                    if event.button == 4: # Scroll up
                        self.zoom_level *= 1.1
                    elif event.button == 5: # Scroll down
                        self.zoom_level /= 1.1
                    self.zoom_level = np.clip(self.zoom_level, min_zoom, max_zoom)
            
            return True # Continue simulation

        except pygame.error as e_pygame_event:
            logging.error(f"Pygame error during event handling: {e_pygame_event}. Attempting to continue.", exc_info=True)
            return True # Try to continue
        except Exception as e_event_unexpected:
            logging.error(f"Unexpected error during event handling: {e_event_unexpected}", exc_info=True)
            return True # Try to continue