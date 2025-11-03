from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def __eq__(self, other: Point) -> bool:
        return (self.x == other.x) and (self.y == other.y)

    def norm(self, other: "Point") -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def __lt__(self, other: Point) -> bool:
        return True