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
import random
import time
import interface as vis
epsilon = 2.0

# Implementation d'une file de priorié
class PriorityQueue_heap:
    def __init__(self, heap: list[pc.Point, float]) -> None:
        self._heap = heap
    
    def _find_higher_priority_point(self) -> pc.Point: 
        best_point = self._heap[0][1]
        return best_point

    def append(self, point: pc.Point, priority: float) -> None:
        """this function adds the new value (priority, point) by preseving the heap structure"""
        heapq.heappush(self._heap, (priority, point)) 

    def remove(self) -> pc.Point:
        """Removes and returns the point with the highest priority (lowest cost)."""
        best_point = self._find_higher_priority_point()
        heapq.heappop(self._heap) 
        return best_point
    
    def size(self) -> int:
        return len(self._heap) 
        
def distances_costs(start: pc.Point, end: pc.Point, grey_levels: ui.GreyImage, list_visited: list[pc.Point]) -> dict[pc.Point, float]:
    """Computes the list of shortest path costs from start until we reach the end point"""
    dist = {}
    for point in grey_levels.graph.keys():
        dist[point] = np.inf
    dist[start] = 0
    to_visit = PriorityQueue_heap([])
    to_visit.append(start, 0)
    while to_visit.size() > 0:
        candidate = to_visit.remove()
        list_visited.append(candidate)
        if candidate == end: # On arrête dès qu'on a trouvé le point final
            print(candidate)
            print(candidate.norm(end))
            break
        for neighbor in grey_levels.neighbors(candidate):
            assert neighbor.x < grey_levels.width and neighbor.y < grey_levels.height
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
        assert(point.y < grey_levels.height and point.x < grey_levels.width)
        if distance < np.inf:
            intensity = (distance - min_dist) / (max_dist - min_dist)
            color = col.to_rgb(myMap(intensity))
            color_list = [color[0], color[1], color[2]]
            for i in range (3):
                color_list[i] = int(255*color_list[i])
            colored_map[point.y, point.x] = color_list
    return colored_map

def gradient_point_x(point: pc.Point, distances: dict[pc.Point, float], grey_levels: ui.GreyImage) -> float:
    """Compute the gradient over y of a point of the distance_map"""
    p_north = pc.Point(point.x-1,point.y)
    if point.x == grey_levels.width - 1:
        p_south = point
    else : 
        p_south = pc.Point(point.x+1, point.y)
        if not(distances[p_south] < np.inf):
            p_south = point
    if point.x==0:
        p_north = point
    if p_south == point or p_north == point:
        return distances[p_south] - distances[p_north]
    return (distances[p_south] - distances[p_north])/2

def gradient_point_y(point: pc.Point, distances: dict[pc.Point, float], grey_levels: ui.GreyImage) -> float:
    """Compute the gradient over x of a point of the distance_map"""
    p_west = pc.Point(point.x, point.y-1)
    if point.y == grey_levels.height - 1:
        p_east = point
    else : 
        p_east = pc.Point(point.x, point.y+1)
        if not(distances[p_east] < np.inf):
            p_east = point
    if point.y==0 :
        p_west = point
    return (distances[p_west] - distances[p_east])/2

def gradient_y(dist: dict[pc.Point, float], grey_levels: ui.GreyImage) -> dict[pc.Point, float]:
    """compute the gradient on the distance map"""
    image_gradient = {}
    for point in dist:
        if dist[point] < np.inf:
            image_gradient[point] = gradient_point_y(point, dist, grey_levels)
    return image_gradient

def gradient_x(dist: dict[pc.Point, float], grey_levels: ui.GreyImage) -> dict[pc.Point, float]:
    """compute the gradient on the distance map"""
    image_gradient = {}
    for point in dist:
        if dist[point] < np.inf:
           image_gradient[point] = gradient_point_x(point, dist, grey_levels)
    return image_gradient

def gradient_on_image(dist: dict[pc.Point, float], grey_levels: ui.GreyImage, obs: vis.Observer|None = None) -> np.ndarray:
    """Display the gradient on an image"""
    debut = time.time()
    grad_x = gradient_x(dist, grey_levels)
    print("grad_x", time.time()-debut)
    debut = time.time()
    grad_y = gradient_y(dist, grey_levels)
    print("grad_y", time.time()-debut)
    debut = time.time()
    colored_map = np.zeros((grey_levels.height, grey_levels.width, 3), dtype=np.uint8)
    myMap = plt.get_cmap('GnBu')
    intensity = {}
    max_intensity = 0
    for point in grad_x:
        intensity[point] = sqrt(abs(grad_x[point])+abs(grad_y[point]))
        if intensity[point] > max_intensity and intensity[point] < np.inf:
            max_intensity = intensity[point]
    print("max_intensity", time.time()-debut)
    debut = time.time()
    print("max intensity: ", max_intensity)
    cpt = len(grad_x)
    for point in grad_x:
        if obs is not None:
            obs.notify_observer(cpt)
        cpt -= 1
        if intensity[point]<np.inf:
            r = intensity[point]/max_intensity
            theta = (atan2(grad_y[point],grad_x[point])*180/np.pi+180)/360
            color = col.to_rgb(myMap(theta))
            color_list = [color[0], color[1], color[2]]
            for i in range (3):
                color_list[i] = int(255*color_list[i]*r)
            colored_map[point.y, point.x] = color_list
    print("colored_map", time.time()-debut)
    return colored_map

def valid_neighbours(grey_levels: ui.GreyImage, point: pc.Point, visited: dict[pc.Point, bool],
                    dist: dict[pc.Point, float], list_visited: list[pc.Point]) -> list[pc.Point]:
    neighbours = [pc.Point(point.x-1,point.y), pc.Point(point.x,point.y+1), pc.Point(point.x+1,point.y), pc.Point(point.x,point.y-1)]
    valid_neighbours = []
    for p in neighbours:
        if p in list_visited: #if neighbours[i] in list_visited: #
            if visited[p] == False:
                valid_neighbours.append(p)
    #print(neighbours, valid_neighbours)
    return valid_neighbours
    
def test_minimum_neighbours(point: pc.Point, grad_x: dict[pc.Point, float], grad_y: dict[pc.Point, float], 
                            grey_levels: ui.GreyImage, dist: dict[pc.Point, float], visited: dict[pc.Point, bool],
                            list_visited: list[pc.Point], start_point: pc.Point) -> pc.Point:
    neighbours = valid_neighbours(grey_levels, point, visited, dist, list_visited)
    mini_point = None
    if len(neighbours)==1:
        mini_point = neighbours[0]
    else: 
        if pc.Point(point.x-copysign(1,grad_x[point]), point.y) in neighbours:
            if (abs(grad_x[point]) < abs(grad_y[point]) or grad_y[point] == 0) :
                mini_point = pc.Point(point.x-int(copysign(1,grad_x[point])), point.y)

        if pc.Point(point.x, point.y-copysign(1,grad_y[point])) in neighbours:
            if (abs(grad_y[point]) < abs(grad_x[point]) or grad_x[point] == 0) :
                mini_point = pc.Point(point.x, int(point.y-copysign(1,grad_y[point])))
        
        if mini_point is None:
            diff_x = point.x - start_point.x
            diff_y = point.y - start_point.y 
            r = random.randint(0, 1)
            if pc.Point(point.x-int(copysign(1,diff_x)), point.y) in neighbours:
                if abs(diff_x) > abs(diff_y):
                    mini_point = pc.Point(point.x-int(copysign(1,diff_x)), point.y)
                elif abs(diff_x) == abs(diff_y):
                    mini_point = pc.Point(point.x-int(copysign(1,diff_x)), point.y)
                    
            elif pc.Point(point.x, point.y-int(copysign(1,diff_y))) in neighbours:
                if (abs(diff_y) > abs(diff_x)):
                    mini_point = pc.Point(point.x, point.y- int(copysign(1,diff_y)))
                elif abs(diff_x) == abs(diff_y):
                    mini_point = pc.Point(point.x, point.y- int(copysign(1,diff_y)))
    return mini_point

def gradient_descent(distances: dict[pc.Point, float], grey_levels: ui.GreyImage, start_point: pc.Point, end_point: pc.Point, list_visited: list[pc.Point]) -> list[pc.Point]:
    grad_x = gradient_x(distances, grey_levels)
    grad_y = gradient_y(distances, grey_levels)
    point = end_point
    descent = [point]
    i=0
    visited = {}
    for p in grey_levels.graph:
        if distances[p] < np.inf:
            visited[p] = False
    visited[point] = True
    list_cost = []
    cost_ = 0
    while point != start_point:
            next_point = test_minimum_neighbours(point, grad_x, grad_y, grey_levels, dist, visited, list_visited, start_point)
            if next_point is None:
                descent.pop()
                next_point = descent[-1]
            else:
                visited[next_point] = True
                descent.append(next_point)
            #print(i, point, next_point)  
            cost_ += grey_levels.cost(point, next_point)
            list_cost.append(cost_)
            point = next_point
            i+=1
            assert(i<20000) # to be sure that the while loop is not infinite
    print("cout du chemin: ", list_cost[-1])
    print(len(descent))
    return descent

def affiche_descent(descent: list[pc.Point], img: ui.GreyImage) -> np.ndarray:
    #colored_map = np.zeros((img.height, img.width, 3), dtype=np.uint8)
    #print(img.width, img.height)
    #for i in range(img.width):
    #    for j in range(img.height):
    #        p = pc.Point(i,j)
    #        colored_map[j][i] = [img.graph[p], img.graph[p], img.graph[p]]
    #sum = 0
    for point in descent:
    #    sum += img.cost(pc.Point(289,136), point)
        img[point.y][point.x] = [255, 0, 0]
    #print(sum)
    return img

def distances_map(start: pc.Point, end: pc.Point, grey_levels: ui.GreyImage) -> np.ndarray:
    """Generates a colored distances map from start to end points based on grey levels."""
    return coloration_map(distances_costs(start, end, grey_levels), grey_levels)

if __name__ == "__main__":
    #im = ui.GreyImage('EZEZEZEZ.png')
    im = ui.GreyImage('Carte.png')
    print(im.width, im.height)
    start = pc.Point(446,332)
    end = pc.Point(716,272)
    list_visited = []
    distances = distances_costs(start, end, im, list_visited)
    print("distances okay")
    colored_map = coloration_map(distances, im)
    print("coloration map okay")
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
    print("grad image okay")
    """for k in range(10):
        grad_image[min(start.y+k,700), start.x] = [0,0,0]
        grad_image[start.y-k, start.x] = [0,0,0]
        grad_image[start.y, min(start.x+k,1324)] = [0,0,0]
        grad_image[start.y, start.x-k] = [0,0,0]
        grad_image[min(end.y+k,700), end.x] = [0,255,0]
        grad_image[end.y-k, end.x] = [0,255,0]
        grad_image[end.y, min(end.x+k,1324)] = [0,255,0]
        grad_image[end.y, end.x-k] = [0,255,0]
    """
    grad_image_ = ui.Image.fromarray(grad_image, 'RGB')
    grad_image_.show()
    descent = gradient_descent(distances, im, start, end, list_visited)
    final_img = affiche_descent(descent, grad_image)
    print("a", im.cost(start,pc.Point(288,236))+im.cost(pc.Point(288,236),end))
    final_img = ui.Image.fromarray(final_img, 'RGB')
    final_img.show()