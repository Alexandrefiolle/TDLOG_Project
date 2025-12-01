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
        
def distances_costs(start: pc.Point, end: pc.Point, grey_levels: ui.GreyImage) -> tuple[dict[pc.Point, float], list[pc.Point]]:
    """Computes the list of shortest path costs from start until we reach the end point"""
    dist = {}
    for point in grey_levels.graph.keys():
        dist[point] = np.inf
    dist[start] = 0
    to_visit = PriorityQueue_heap([])
    to_visit.append(start, 0)
    visited = []
    while to_visit.size() > 0:
        candidate = to_visit.remove()
        visited.append(candidate)
        if candidate == end: break
        for neighbor in grey_levels.neighbors(candidate):
            cost = grey_levels.cost(start, neighbor, epsilon)
            if dist[neighbor] > dist[candidate] + cost:
                dist[neighbor] = dist[candidate] + cost
                to_visit.append(neighbor, dist[neighbor])
    return dist,visited

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
            colored_map[point.y, point.x] = color_list
    return colored_map

def gradient_point_y(point: pc.Point, dist: dict[pc.Point, float], grey_levels: ui.GreyImage) -> float:
    """Compute the gradient of a point of the distance_map"""
    p_north = pc.Point(point.y-1,point.x)
    if point.x == grey_levels.height - 1:
        p_south = point
    else : 
        p_south = pc.Point(point.y+1, point.x)
        if not(dist[p_south] < np.inf):
            p_south = point
    if point.x==0:
        p_north = point
    if p_south == point or p_north == point:
        return dist[p_south] - dist[p_north]
    return (dist[p_south] - dist[p_north])/2

def gradient_point_x(point: pc.Point, dist: dict[pc.Point, float], grey_levels: ui.GreyImage) -> float:
    """Compute the gradient of a point of the distance_map"""
    p_west = pc.Point(point.y, point.x-1)
    if point.y == grey_levels.width - 1:
        p_east = point
    else : 
        p_east = pc.Point(point.y, point.x+1)
        if not(dist[p_east] < np.inf):
            p_east = point
    if point.y==0 :
        p_west = point
    return (dist[p_west] - dist[p_east])/2

def gradient_y(dist: dict[pc.Point, float], grey_levels: ui.GreyImage) -> dict[pc.Point, float]:
    """compute the gradient on the distance map"""
    image_gradient = {}
    for point in dist:
        if dist[point] < np.inf:
            image_gradient[point] = gradient_point_x(point, dist, grey_levels)
    return image_gradient

def gradient_x(dist: dict[pc.Point, float], grey_levels: ui.GreyImage) -> dict[pc.Point, float]:
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
            colored_map[point.y, point.x] = color_list
    return colored_map

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
            colored_map[i][j] = [img.graph[p], img.graph[p], img.graph[p]]
    sum = 0
    for point in descent:
        sum += img.cost(pc.Point(289,136), point)
        colored_map[point.y][point.x] = [255, 0, 0]
    print(sum)
    return colored_map

def distances_map(start: pc.Point, end: pc.Point, grey_levels: ui.GreyImage) -> np.ndarray:
    """Generates a colored distances map from start to end points based on grey levels."""
    return coloration_map(distances_costs(start, end, grey_levels), grey_levels)

if __name__ == "__main__":
    im = ui.GreyImage('Carte.png')
    start = pc.Point(123,77)
    end = pc.Point(341,81)
    distances,visited = distances_costs(start, end, im)
    colored_map = coloration_map(distances, im)
    colored_map[start.y, start.x] = [255,0,0]
    for k in range(10):
        colored_map[min(start.y+k,700), start.x] = [0,0,0]
        colored_map[start.y-k, start.x] = [0,0,0]
        colored_map[start.y, min(start.x+k,1324)] = [0,0,0]
        colored_map[start.y, start.x-k] = [0,0,0]
        colored_map[min(end.y+k,700), end.x] = [0,255,0]
        colored_map[end.y-k, end.x] = [0,255,0]
        colored_map[end.y, min(end.x+k,1324)] = [0,255,0]
        colored_map[end.y, end.x-k] = [0,255,0]
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