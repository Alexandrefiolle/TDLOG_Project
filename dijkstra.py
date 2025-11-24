import point_class as pc
import interface as ui
from collections import deque
import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as col
from math import*
epsilon = 2.0

class PriorityQueue:
    def __init__(self, dic: dict[pc.Point, float]) -> None:
        self._dic = dic
    
    def _find_higher_priority_point(self) -> pc.Point: # ensuite, utiliser un tas de priorité
        dist_inf = np.inf
        best_point = None
        for (point, distance) in self._dic.items():
            if dist_inf > distance:
                dist_inf = distance
                best_point = point
        return best_point

    def append(self, point: pc.Point, priority: float) -> None:
        self._dic[point] = priority

    def remove(self) -> pc.Point:
        best_point = self._find_higher_priority_point()
        del self._dic[best_point] 
        return best_point
    
    def size(self) -> int:
        return len(self._dic)
        

def distances_costs(start: pc.Point, end: pc.Point, grey_levels: ui.GreyImage) -> dict[pc.Point, float]:
    """Computes the list of shortest path costs from start until we reach the end point"""
    dist = {}
    for point in grey_levels.graph.keys():
        dist[point] = np.inf
    dist[start] = 0
    to_visit = PriorityQueue({})
    to_visit.append(start, 0)
    while to_visit.size() > 0:
        candidate = to_visit.remove()
        if candidate == end: break
        for neighbor in grey_levels.neighbors(candidate):
            cost = grey_levels.cost(start, neighbor, epsilon)
            if dist[neighbor] > dist[candidate] + cost:
                dist[neighbor] = dist[candidate] + cost
                to_visit.append(neighbor, dist[neighbor])
    return dist

def coloration_map(distances: dict[pc.Point, float], grey_levels: ui.GreyImage) -> np.ndarray:
    """Colors the map according to the distances computed"""
    max_dist = 0
    for distance in distances.values():
        if distance < np.inf and distance>max_dist:
            max_dist = distance
    min_dist = min(distances.values())
    print(max_dist)
    colored_map = np.zeros((grey_levels.height, grey_levels.width, 3), dtype=np.uint8)
    myMap = plt.get_cmap('Spectral')
    for point, distance in distances.items():
        if distance < np.inf:
            intensity = (distance - min_dist) / (max_dist - min_dist)
            color = col.to_rgb(myMap(intensity))
            color_list = [color[0], color[1], color[2]]
            for i in range (3):
                color_list[i] = int(255*color_list[i])
            colored_map[point.x, point.y] = color_list
    return colored_map

def gradient_point_x(point: pc.Point, dist: dict[pc.Point, float], grey_levels: ui.GreyImage) -> float:
    """Compute the gradient of a point of the distance_map"""
    p_north = pc.Point(point.x-1,point.y)
    if point.x == grey_levels.height - 1:
        p_south = point
    else : 
        p_south = pc.Point(point.x+1, point.y)
        if not(dist[p_south] < np.inf):
            p_south = point
    if point.x==0:
        p_north = point
    return (dist[p_south] - dist[p_north])/2

def gradient_point_y(point: pc.Point, dist: dict[pc.Point, float], grey_levels: ui.GreyImage) -> float:
    """Compute the gradient of a point of the distance_map"""
    p_west = pc.Point(point.x, point.y-1)
    if point.y == grey_levels.width - 1:
        p_east = point
    else : 
        p_east = pc.Point(point.x, point.y+1)
        if not(dist[p_east] < np.inf):
            p_east = point
    if point.y==0 :
        p_west = point
    return (dist[p_west] - dist[p_east])/2

def gradient_x(dist: dict[pc.Point, float], grey_levels: ui.GreyImage) -> dict[pc.Point, float]:
    """compute the gradient on the distance map"""
    image_gradient = {}
    for point in dist:
        if dist[point] < np.inf:
            image_gradient[point] = gradient_point_x(point, dist, grey_levels)
    return image_gradient

def gradient_y(dist: dict[pc.Point, float], grey_levels: ui.GreyImage) -> dict[pc.Point, float]:
    """compute the gradient on the distance map"""
    image_gradient = {}
    for point in dist:
        if dist[point] < np.inf:
           image_gradient[point] = gradient_point_y(point, dist, grey_levels)
    return image_gradient

def gradient_on_image(dist: dict[pc.Point, float], grey_levels: ui.GreyImage) -> np.ndarray:
    """Display the gradient on an image"""
    grad_x = gradient_x(dist, grey_levels)
    grad_y = gradient_y(dist, grey_levels)
    colored_map = np.zeros((grey_levels.height, grey_levels.width, 3), dtype=np.uint8)
    myMap = plt.get_cmap('GnBu')
    intensity = {}
    max_intensity = 0
    for point in grad_x:
        intensity[point] = sqrt(abs(grad_x[point])+abs(grad_y[point]))
        if intensity[point] > max_intensity and intensity[point] < np.inf:
            max_intensity = intensity[point]
    print("max intensity: ", max_intensity)
    for point in grad_x:
        if intensity[point]<np.inf:
            r = intensity[point]/max_intensity
            theta = (atan2(grad_y[point],grad_x[point])*180/np.pi+180)/360
            color = col.to_rgb(myMap(theta))
            color_list = [color[0], color[1], color[2]]
            for i in range (3):
                color_list[i] = int(255*color_list[i]*r)
            colored_map[point.x, point.y] = color_list
    return colored_map

if __name__ == "__main__":
    im = ui.GreyImage(ui.im_array)
    start = pc.Point(10,10)
    end = pc.Point(700,1300)
    distances = distances_costs(start, end, im)
    colored_map = coloration_map(distances, im)
    img = ui.Image.fromarray(colored_map, 'RGB')
    img.show()
    grad_image = gradient_on_image(distances, im)
    img = ui.Image.fromarray(grad_image, 'RGB')
    img.show()