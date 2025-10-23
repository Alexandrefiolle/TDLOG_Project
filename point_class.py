from __future__ import annotations
import numpy as np

class Point():
    def __init__(self, x: int, y: int) -> None:
        self._x = x
        self._y = y

    @property
    def x(self) -> int:
        return self._x
    @x.setter
    def x(self, new_x: int) -> None:
        self._x = new_x
    
    @property
    def y(self) -> int:
        return self._y
    @y.setter
    def y(self, new_y) -> None:
        self._y = new_y

    def __eq__(self, other: Point) -> bool:
        return (self._x == other._x) and (self._y == other._y)

    def norm(self, other: Point) -> float:
        return np.sqrt((self._x-other._x)**2 + (self._y-other._y)**2)