from PIL import Image
import numpy as np
import point_class as pc

im = Image.open('Carte.png')
im.show()
im_array = np.array(im.convert('L'), dtype=np.int16)
print(im_array.shape)
print(im_array[0][0])

class GreyImage:
    def __init__(self, image: np.ndarray) -> None:
        self._height = image.shape[0]
        self._width = image.shape[1]
        self._graph: dict[pc.Point, int] = {}
        for x in range(self._height):
            for y in range(self._width):
                p = pc.Point(x,y)
                self._graph[p] = image[x][y]

    @property
    def height(self) -> int:
        return self._height
    @property
    def width(self) -> int:
        return self._width
    @property
    def graph(self) -> dict[pc.Point, int]:
        return self._graph
    
    def __getitem__(self, key: pc.Point) -> int:
        return self._graph[key]
    
    def neighbors(self, m: pc.Point) -> list[pc.Point]:
        """Returns the list of neighbors of a given point m"""
        neigh = []
        x, y = m.x, m.y
        if x > 0:
            neigh.append(pc.Point(x-1, y))
        if x < self._height - 1:
            neigh.append(pc.Point(x+1, y))
        if y > 0:
            neigh.append(pc.Point(x, y-1))
        if y < self._width - 1:
            neigh.append(pc.Point(x, y+1))
        return neigh
    
    def cost(self, m0: pc.Point, m: pc.Point, epsilon: float=1) -> float:
        """Computes the cost induced two points of the image"""
        return epsilon + np.abs(self[m0] - self[m])