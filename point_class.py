"this file defines a basic Point class to help represent coordinates in a 2D space."

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Point:
    """A class representing a point in 2D space with integers x and y coordinates."""
    x: int
    y: int

    def __eq__(self, other: Point) -> bool:
        """Checks if two points are equal based on their coordinates."""
        return (self.x == other.x) and (self.y == other.y)

    def norm(self, other: "Point") -> float:
        """Computes the Euclidean distance between this point and another point."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def __lt__(self, other: Point) -> bool: 
        # This function compare artificially two points. As we are using a heap implementation, we need to have a comparison 
        # between points when the priority is the same. When the priority is the same we don't care which point we use first
        # for dijkstra, returning true is enough for our implementation
        return True