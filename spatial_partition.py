import numpy as np
from config import config
from particle_system import Particle # Assuming Particle class is in particle_system

class Rectangle:
    def __init__(self, x, y, w, h):
        self.x = x  # Center x
        self.y = y  # Center y
        self.w = w  # Half-width
        self.h = h  # Half-height

    def contains(self, particle: Particle) -> bool:
        """Check if a particle (based on its position) is within this rectangle."""
        px, py = particle.pos
        return (self.x - self.w <= px < self.x + self.w and
                self.y - self.h <= py < self.y + self.h)

    def intersects(self, other_rect: 'Rectangle') -> bool:
        """Check if this rectangle intersects with another rectangle."""
        # No intersection if one rectangle is to the left of the other
        if self.x + self.w < other_rect.x - other_rect.w or \
           other_rect.x + other_rect.w < self.x - self.w:
            return False
        # No intersection if one rectangle is above the other
        if self.y + self.h < other_rect.y - other_rect.h or \
           other_rect.y + other_rect.h < self.y - self.h:
            return False
        return True

class Quadtree:
    def __init__(self, boundary: Rectangle, capacity: int, max_depth: int, depth: int = 0):
        self.boundary = boundary
        self.capacity = capacity  # Max objects before subdividing
        self.max_depth = max_depth
        self.depth = depth
        self.particles: list[Particle] = []
        self.divided = False
        self.northwest: 'Quadtree' = None
        self.northeast: 'Quadtree' = None
        self.southwest: 'Quadtree' = None
        self.southeast: 'Quadtree' = None

    def _subdivide(self):
        x = self.boundary.x
        y = self.boundary.y
        hw = self.boundary.w / 2 # half-width of children
        hh = self.boundary.h / 2 # half-height of children

        # Define boundaries for the four new quadrants
        nw_boundary = Rectangle(x - hw, y - hh, hw, hh)
        ne_boundary = Rectangle(x + hw, y - hh, hw, hh)
        sw_boundary = Rectangle(x - hw, y + hh, hw, hh)
        se_boundary = Rectangle(x + hw, y + hh, hw, hh)

        self.northwest = Quadtree(nw_boundary, self.capacity, self.max_depth, self.depth + 1)
        self.northeast = Quadtree(ne_boundary, self.capacity, self.max_depth, self.depth + 1)
        self.southwest = Quadtree(sw_boundary, self.capacity, self.max_depth, self.depth + 1)
        self.southeast = Quadtree(se_boundary, self.capacity, self.max_depth, self.depth + 1)
        self.divided = True

        # Move particles from this node to children
        for particle in self.particles:
            if self.northwest.insert(particle):
                continue
            if self.northeast.insert(particle):
                continue
            if self.southwest.insert(particle):
                continue
            if self.southeast.insert(particle):
                continue
            # If a particle somehow doesn't fit, it might be an issue with boundary checks or particle position
            # For now, we'll assume it should fit in one.
        self.particles = [] # Clear particles from this node as they are now in children

    def insert(self, particle: Particle) -> bool:
        if not self.boundary.contains(particle):
            return False  # Particle is not within this quadtree's boundary

        if not self.divided:
            if len(self.particles) < self.capacity or self.depth == self.max_depth:
                self.particles.append(particle)
                return True
            else:
                self._subdivide()
        
        # If divided (either previously or just now), try inserting into children
        if self.northwest.insert(particle): return True
        if self.northeast.insert(particle): return True
        if self.southwest.insert(particle): return True
        if self.southeast.insert(particle): return True
        
        # Should not happen if boundary.contains was true and subdivision logic is correct
        # print(f"Warning: Particle at {particle.pos} could not be inserted into children of quadtree at depth {self.depth} with boundary {self.boundary.x, self.boundary.y, self.boundary.w, self.boundary.h}")
        return False


    def query(self, range_rect: Rectangle, found_particles: list[Particle]) -> list[Particle]:
        if not self.boundary.intersects(range_rect):
            return found_particles # No intersection, no need to check further

        if self.divided:
            self.northwest.query(range_rect, found_particles)
            self.northeast.query(range_rect, found_particles)
            self.southwest.query(range_rect, found_particles)
            self.southeast.query(range_rect, found_particles)
        else:
            for particle in self.particles:
                if range_rect.contains(particle): # Check if particle is within the query range
                    found_particles.append(particle)
        return found_particles

    def clear(self):
        self.particles = []
        self.divided = False
        self.northwest = None
        self.northeast = None
        self.southwest = None
        self.southeast = None

    def count(self) -> int:
        """Returns the total number of particles in this quadtree and its children."""
        total = len(self.particles)
        if self.divided:
            total += self.northwest.count()
            total += self.northeast.count()
            total += self.southwest.count()
            total += self.southeast.count()
        return total