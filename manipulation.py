"this file contains the GreyImage class which handles the loading and manipulation of grey-scale images."

from PIL import Image
import numpy as np
import point_class as pc

class GreyImage:
    """A class representing a grey-scale image as a graph of grey levels and stocking dimensions."""
    def __init__(self, file_path : str) -> None:
        """Loads the image from the given file path and initializes the graph representation."""
        im = Image.open(file_path)
        image = np.array(im.convert('L'), dtype=np.int16)
        self._height = image.shape[0]
        self._width = image.shape[1]
        self._graph: dict[pc.Point, int] = {}
        for x in range(self._width):
            for y in range(self._height):
                p = pc.Point(x,y)
                self._graph[p] = image[y][x]

    @property
    def height(self) -> int:
        """Returns the height of the image."""
        return self._height
    @property
    def width(self) -> int:
        """Returns the width of the image."""
        return self._width
    @property
    def graph(self) -> dict[pc.Point, int]:
        """Returns the graph representation of the image."""
        return self._graph
    
    def __getitem__(self, key: pc.Point) -> int:
        """Allows accessing the grey level of a point using indexing."""
        return self._graph[key]
    
    def neighbors(self, m: pc.Point) -> list[pc.Point]:
        """Returns the list of neighbors of a given point m"""
        neigh = []
        x, y = m.x, m.y
        if x > 0:
            neigh.append(pc.Point(x-1, y))
        if x < self._width - 1:
            neigh.append(pc.Point(x+1, y))
        if y > 0:
            neigh.append(pc.Point(x, y-1))
        if y < self._height - 1:
            neigh.append(pc.Point(x, y+1))
        return neigh
    
    def cost(self, m0: pc.Point, m: pc.Point, epsilon: float=1) -> float:
        """Computes the cost induced two points of the image"""
        return epsilon + np.abs(self[m0] - self[m])