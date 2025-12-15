"This file dwells on the implementation of Dijkstra's algorithm "
"to compute shortest paths on a graph represented by image grey levels."

import point_class as pc
import manipulation as ui
from collections import deque
import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as col
from math import*
import time
import interface as vis
from PIL import Image
epsilon = 2.0

class PriorityQueue_heap:
    def __init__(self, heap: list[pc.Point, float]) -> None:
        self._heap = heap
    
    def _find_higher_priority_point(self) -> pc.Point: 
        best_point = self._heap[0][1]
        return best_point

    def append(self, point: pc.Point, priority: float) -> None:
        heapq.heappush(self._heap, (priority, point)) # this function adds the new value (priority, point) by preseving the heap structure

    def remove(self) -> pc.Point:
        """Removes and returns the point with the highest priority (lowest cost)."""
        best_point = self._find_higher_priority_point()
        heapq.heappop(self._heap) 
        return best_point
    
    def size(self) -> int:
        return len(self._heap) 
        
def distances_costs(start: pc.Point, end: pc.Point, grey_levels: ui.GreyImage, obs = None) -> tuple[dict[pc.Point, float], list[pc.Point]]:
    """Computes the list of shortest path costs from start until we reach the end point"""
    dist = ui.Distances(grey_levels)
    dist[start] = 0
    to_visit = PriorityQueue_heap([])
    to_visit.append(start, 0)
    visited = []
    while to_visit.size() > 0:
        candidate = to_visit.remove()
        if obs is not None:
            obs.notify_observer(candidate.norm(end))
        visited.append(candidate)
        if candidate == end: 
            break
        for neighbor in grey_levels.neighbors(candidate):
            assert neighbor.x < grey_levels.width and neighbor.y < grey_levels.height
            cost = grey_levels.cost(start, neighbor, epsilon)
            if dist[neighbor] > dist[candidate] + cost:
                dist[neighbor] = dist[candidate] + cost
                to_visit.append(neighbor, dist[neighbor])
    return dist

def coloration_map(distances: ui.Distances, grey_levels: ui.GreyImage) -> np.ndarray:
    """Colors the map according to the distances computed"""
    max_dist = np.max(distances.map, where=np.isfinite(distances.map), initial=0)
    min_dist = np.min(distances.map)
    print(max_dist)
    intensity = (distances.map - min_dist)/(max_dist - min_dist)
    myMap = plt.get_cmap('Spectral')
    myMap.set_over(color='black')
    colored_map = (myMap(intensity)[:, :, :3] * 255).astype(np.uint8)
    return colored_map

def gradient_point_x(point: pc.Point, dist: dict[pc.Point, float], grey_levels: ui.GreyImage) -> float:
    """Compute the gradient of a point of the distance_map"""
    p_north = pc.Point(point.x-1,point.y)
    if point.x == grey_levels.width - 1:
        p_south = point
    else : 
        p_south = pc.Point(point.x+1, point.y)
        if not(dist[p_south] < np.inf):
            p_south = point
    if point.x==0:
        p_north = point
    if p_south == point or p_north == point:
        return dist[p_south] - dist[p_north]
    return (dist[p_south] - dist[p_north])/2

def gradient_point_y(point: pc.Point, dist: dict[pc.Point, float], grey_levels: ui.GreyImage) -> float:
    """Compute the gradient of a point of the distance_map"""
    p_west = pc.Point(point.x, point.y-1)
    if point.y == grey_levels.height - 1:
        p_east = point
    else : 
        p_east = pc.Point(point.x, point.y+1)
        if not(dist[p_east] < np.inf):
            p_east = point
    if point.y==0 :
        p_west = point
    return (dist[p_west] - dist[p_east])/2

def gradient_y(dist: dict[pc.Point, float], grey_levels: ui.GreyImage) -> dict[pc.Point, float]:
    """compute the gradient on the distance map"""
    image_gradient = ui.Distances(grey_levels)
    for point in dist:
        if dist[point] < np.inf:
            image_gradient[point] = gradient_point_y(point, dist, grey_levels)
    return image_gradient

def gradient_x(dist: dict[pc.Point, float], grey_levels: ui.GreyImage) -> dict[pc.Point, float]:
    """compute the gradient on the distance map"""
    image_gradient = ui.Distances(grey_levels)
    for point in dist:
        if dist[point] < np.inf:
           image_gradient[point] = gradient_point_x(point, dist, grey_levels)
    return image_gradient

def gradient_on_image(dist: dict[pc.Point, float], grey_levels: ui.GreyImage, obs = None) -> np.ndarray:
    """Display the gradient on an image"""
    grad_x = gradient_x(dist, grey_levels)
    
    grad_y = gradient_y(dist, grey_levels)
    
    myMap = plt.get_cmap('GnBu')
    intensity = np.sqrt(np.abs(grad_x.map) + np.abs(grad_y.map))
    np.putmask(intensity, np.isinf(intensity), 0)
    intensity = intensity/np.max(intensity)
    theta = np.arctan2(grad_x.map,grad_y.map)/(2*np.pi) + 0.5
    colored_map = np.einsum("ij, ijk -> ijk", intensity, myMap(theta)[:, :, :3])

    cpt = grey_levels.width*grey_levels.height
    
    
    return (colored_map * 255).astype(np.uint8)

def valid_neighbours(grey_levels: ui.GreyImage, point:pc.Point, visited: dict[pc.Point, bool],
                    dist: dict[pc.Point, float], list_visited: list[pc.Point]) -> list[pc.Point]:
    neighbours = [pc.Point(point.x-1,point.y), pc.Point(point.x,point.y+1), pc.Point(point.x+1,point.y), pc.Point(point.x,point.y-1)]
    for i in range(len(neighbours)):
        if neighbours[i] in list_visited:
            pass
        else:
            neighbours[i] = point
    valid_neighbours = []
    for p in neighbours:
        if p != point and visited[p] == False:
            valid_neighbours.append(p)
    #print(neighbours, valid_neighbours)
    return valid_neighbours
    
def test_minimum_neighbours(point: pc.Point, grad_x: dict[pc.Point, float], grad_y: dict[pc.Point, float], 
                            grey_levels: ui.GreyImage, dist: dict[pc.Point, float], visited: dict[pc.Point, bool],
                            list_visited: list[pc.Point]) -> pc.Point:
    neighbours = valid_neighbours(grey_levels, point, visited, dist, list_visited)
    mini = 0
    mini_point = None
    print(grad_x[point], grad_y[point])
    if pc.Point(point.x-copysign(1,grad_y[point]), point.y) in neighbours:
        if (abs(grad_y[point]) < abs(grad_x[point]) or grad_x[point] == 0) :
            mini_point = pc.Point(point.x-int(copysign(1,grad_y[point])), point.y)
    elif pc.Point(point.x, point.y-copysign(1,grad_x[point])) in neighbours:
        mini_point = pc.Point(point.x, int(point.y-copysign(1,grad_x[point])))

    if pc.Point(point.x, point.y-copysign(1,grad_x[point])) in neighbours:
        if (abs(grad_x[point]) < abs(grad_y[point]) or grad_x[point] == 0) :
            mini_point = pc.Point(point.x, int(point.y-copysign(1,grad_x[point])))
    elif pc.Point(point.x-copysign(1,grad_y[point]), point.y) in neighbours:
        mini_point = pc.Point(point.x-int(copysign(1,grad_y[point])), point.y)
    return mini_point

def gradient_descent(dist: dict[pc.Point, float], grey_levels: ui.GreyImage, start_point: pc.Point, end_point: pc.Point, list_visited: list[pc.Point]) -> list[pc.Point]:
    grad_x = gradient_x(dist, grey_levels)
    grad_y = gradient_y(dist, grey_levels)
    point = end_point
    descent = [point]
    i,k=0,0
    visited = {}
    for p in list_visited:
        visited[p] = False
    visited[point] = True
    while point != start_point:
            next_point = test_minimum_neighbours(point, grad_x, grad_y, grey_levels, dist, visited, list_visited)
            if next_point is None:
                next_point = descent[-k-1]
                k += 1
                descent.pop()
            else:
                k=0
                visited[next_point] = True
                descent.append(next_point)
            print(i, point, next_point)
            #assert(point != next_point)
            point = next_point
            i+=1
            assert(i<10000)
    return descent

def affiche_descent(descent: list[pc.Point], img: ui.GreyImage) -> np.ndarray:
    colored_map = np.zeros((img.height, img.width, 3), dtype=np.uint8)
    for i in range(img.height):
        for j in range(img.width):
            p = pc.Point(i,j)
            colored_map[i][j] = [img[p], img[p], img[p]]
    sum = 0
    for point in descent:
        sum += img.cost(pc.Point(289,136), point)
        colored_map[point.x][point.y] = [255, 0, 0]
    print(sum)
    return colored_map

def distances_map(start: pc.Point, end: pc.Point, grey_levels: ui.GreyImage, obs = None) -> np.ndarray:
    """Generates a colored distances map from start to end points based on grey levels."""
    return coloration_map(distances_costs(start, end, grey_levels, obs), grey_levels)

if __name__ == "__main__":
    im = ui.GreyImage('Carte.png')
    start = pc.Point(10,10)
    end = pc.Point(120,10)
    distances = distances_costs(start, end, im)
    colored_map = coloration_map(distances, im)
    img = Image.fromarray(colored_map)
    img.save("color.png")
    colored_map[start.x, start.y] = [255,0,0]
    for k in range(10):
        colored_map[min(start.x+k,700), start.y] = [0,0,0]
        colored_map[start.x-k, start.y] = [0,0,0]
        colored_map[start.x, min(start.y+k,1324)] = [0,0,0]
        colored_map[start.x, start.y-k] = [0,0,0]
        colored_map[min(end.x+k,700), end.y] = [0,255,0]
        colored_map[end.x-k, end.y] = [0,255,0]
        colored_map[end.x, min(end.y+k,1324)] = [0,255,0]
        colored_map[end.x, end.y-k] = [0,255,0]
    img = ui.Image.fromarray(colored_map, 'RGB')
    img.show()
    grad_image = gradient_on_image(distances, im)
    grad_image = ui.Image.fromarray(grad_image, 'RGB')
    grad_image.show()
    descent = gradient_descent(distances, im, start, end, visited)
    final_img = affiche_descent(descent,im)
    print("a", im.cost(start,pc.Point(288,236))+im.cost(pc.Point(288,236),end))
    final_img = ui.Image.fromarray(final_img, 'RGB')
    final_img.show()