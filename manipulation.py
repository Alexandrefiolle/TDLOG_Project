"this file contains the GreyImage class which handles the loading and manipulation of grey-scale images."

from PIL import Image
import numpy as np
import point_class as pc

class GreyImage:
    """A class representing a grey-scale image as a graph of grey levels and stocking dimensions."""
    def __init__(self, file_path : str) -> None:
        """Loads the image from the given file path and initializes the graph representation."""
        im = Image.open(file_path)
        self.image = np.array(im.convert('L'), dtype=np.int16)
        self._height = self.image.shape[0]
        self._width = self.image.shape[1]

    @property
    def height(self) -> int:
        """Returns the height of the image."""
        return self._height
    @property
    def width(self) -> int:
        """Returns the width of the image."""
        return self._width
    
    def __getitem__(self, key: pc.Point) -> int:
        """Allows accessing the grey level of a point using indexing."""
        return self.image[key.y][key.x]
    
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
    
class NumpyDict:
    """A class representing a distance map."""
    def __init__(self, im : GreyImage) -> None:
        """initializes the distance map."""
        self.map = np.full_like(im.image, np.inf, dtype=np.float64)
        self._height = self.map.shape[0]
        self._width = self.map.shape[1]

    @property
    def height(self) -> int:
        """Returns the height of the image."""
        return self._height
    @property
    def width(self) -> int:
        """Returns the width of the image."""
        return self._width
    
    def __getitem__(self, key: pc.Point) -> int:
        """Allows accessing the grey level of a point using indexing."""
        return self.map[key.y][key.x]
    
    def __setitem__(self, key: pc.Point, value) -> int:
        """Allows accessing the grey level of a point using indexing."""
        self.map[key.y][key.x] = value

    def __iter__(self):
        return (pc.Point(x, y) for x in range(self.width) for y in range(self.height))