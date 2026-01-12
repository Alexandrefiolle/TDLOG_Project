"This file dwells on the implementation of Dijkstra's algorithm "
"to compute shortest paths on a graph represented by image grey levels."

import point_class as pc
import manipulation as ui
import edge_detection as edge
from collections import deque
import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as col
from math import*
import random
import time
import observer as obs
epsilon = 2.0

# Implementation d'une file de priorié
class PriorityQueue_heap:
    """A priority queue implementation using a heap data structure."""
    def __init__(self, heap: list[pc.Point, float]) -> None:
        """Initializes the priority queue with an heap."""
        self._heap = heap
    
    def _find_higher_priority_point(self) -> pc.Point:
        """Finds and returns the point with the highest priority (lowest cost) without removing it.""" 
        best_point = self._heap[0][1]
        return best_point

    def append(self, point: pc.Point, priority: float) -> None:
        """Adds a new point with the given priority to the queue."""
        heapq.heappush(self._heap, (priority, point)) # this function adds the new value (priority, point) by preseving the heap structure

    def remove(self) -> pc.Point:
        """Removes and returns the point with the highest priority (lowest cost)."""
        best_point = self._find_higher_priority_point()
        heapq.heappop(self._heap) 
        return best_point
    
    def size(self) -> int:
        """Returns the number of points in the priority queue."""
        return len(self._heap) 
        
def distances_costs(start: pc.Point, end: pc.Point|None, grey_levels: ui.GreyImage, 
                    list_visited: list[pc.Point], edge_detection: bool = False,
                    weight_map: np.ndarray|None = None, 
                    obs: obs.Observer|None = None) -> tuple[dict[pc.Point, float], list[pc.Point]]:
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
        if end is not None and candidate == end: # On arrête dès qu'on a trouvé le point final
            print(candidate)
            print(candidate.norm(end))
            break
        for neighbor in grey_levels.neighbors(candidate):
            assert neighbor.x < grey_levels.width and neighbor.y < grey_levels.height
            if edge_detection:
                cost = weight_map[neighbor.y, neighbor.x]
            else:
                cost = grey_levels.cost(start, neighbor, epsilon)
            if dist[neighbor] > dist[candidate] + cost:
                dist[neighbor] = dist[candidate] + cost
                to_visit.append(neighbor, dist[neighbor])
    print(dist[end])
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
    return -(distances[p_west] - distances[p_east])/2

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

def gradient_on_image(dist: dict[pc.Point, float], grey_levels: ui.GreyImage, obs: obs.Observer|None = None) -> np.ndarray:
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


def mini(neighbours, grad_x, grad_y, point: pc.Point, start_point: pc.Point, visited: list[pc.Point], list_visited) -> pc.Point:
    mini_point = None
    neighbours_new = []
    for i in range(len(neighbours)):
        if neighbours[i] in list_visited:
            if visited[neighbours[i]] == False:
                neighbours_new.append(neighbours[i])
    if len(neighbours_new) == 1:
        mini_point = neighbours_new[0]
    if pc.Point(point.x-copysign(1,grad_x[point]), point.y) in neighbours_new :
            if (abs(grad_x[point]) < abs(grad_y[point]) or grad_y[point] == 0) :
                    mini_point = pc.Point(point.x-int(copysign(1,grad_x[point])), point.y)
            
    if pc.Point(point.x, point.y-copysign(1,grad_y[point])) in neighbours_new :
            if (abs(grad_y[point]) < abs(grad_x[point]) or grad_x[point] == 0) :
                mini_point = pc.Point(point.x, int(point.y-copysign(1,grad_y[point])))
            
    if mini_point is None:
        diff_x = point.x - start_point.x
        diff_y = point.y - start_point.y 
        if pc.Point(point.x-int(copysign(1,diff_x)), point.y) in neighbours_new :
                if abs(diff_x) > abs(diff_y):
                    mini_point = pc.Point(point.x-int(copysign(1,diff_x)), point.y)
                elif abs(diff_x) == abs(diff_y):
                    mini_point = pc.Point(point.x-int(copysign(1,diff_x)), point.y)
        
        if pc.Point(point.x, point.y-int(copysign(1,diff_y))) in neighbours_new :
                if abs(diff_y) > abs(diff_x):
                    mini_point = pc.Point(point.x, point.y- int(copysign(1,diff_y)))
                elif abs(diff_x) == abs(diff_y):
                    mini_point = pc.Point(point.x, point.y- int(copysign(1,diff_y)))
    #print(mini_point, point)
    return mini_point

def gradient_descent(distances: dict[pc.Point, float], grey_levels: ui.GreyImage, start_point: pc.Point, end_point: pc.Point, list_visited: list[pc.Point]) -> list[pc.Point]:
    start = time.time()
    current = end_point
    path = [current]
    grad_x = gradient_x(distances, grey_levels)
    grad_y = gradient_y(distances, grey_levels)
    visited = {}
    for p in grey_levels.graph:
        if distances[p] < np.inf:
            visited[p] = False
    visited[current] = True
    while current != start_point and distances[current] > 0:
        #print(grad_x[current], grad_y[current])
        neighbors = grey_levels.neighbors(current)
        best = mini(neighbors, grad_x, grad_y, current, start_point, visited, list_visited)
        #print(neighbors, best, current)
        if best is None:
            path.pop()
            current = path[-1]
        else:
            visited[best] = True
            current = best
            path.append(current)
    path.reverse()
    end = time.time()
    print("temps d'execution : ", end-start)
    print("longueur du chemin initial", len(path))
    return path

def affiche_descent(descent: list[pc.Point], img: ui.GreyImage, Sobel: int = 0) -> np.ndarray:
    """Displays the descent path on the image"""
    #colored_map = np.zeros((img.height, img.width, 3), dtype=np.uint8)
    #print(img.width, img.height)
    #for i in range(img.width):
    #    for j in range(img.height):
    #        p = pc.Point(i,j)
    #        colored_map[j][i] = [img.graph[p], img.graph[p], img.graph[p]]
    #sum = 0
    for point in descent:
        if Sobel == 0:
            img[point.y, point.x] = [255, 0, 0]
        else:
            img[point.y, point.x] = [255, 255, 255]
    #print(sum)
    return img

def distances_map(start: pc.Point, end: pc.Point, grey_levels: ui.GreyImage) -> np.ndarray:
    """Generates a colored distances map from start to end points based on grey levels."""
    return coloration_map(distances_costs(start, end, grey_levels), grey_levels)

def amelioration_descent(distances: dict[pc.Point, float], grey_levels: ui.GreyImage, 
                         start_point: pc.Point, end_point: pc.Point, 
                         list_visited: list[pc.Point]) -> list[pc.Point]:
    print("start gradient descent")
    initial_descent = gradient_descent(distances, grey_levels, start_point, end_point, list_visited)
    print("initial gradient done")
    final_descent = [initial_descent[0]]
    list_cost = [0]
    cost_ = 0
    for i in range(1, len(initial_descent)):
        point = initial_descent[i]
        cost = grey_levels.cost(point, final_descent[-1])
        neighbours = grey_levels.neighbors(point)
        for p in neighbours:
            construct_descent = final_descent[:-1]
            if p in construct_descent:
                cost = grey_levels.cost(point, p)
                cost_descent = 0
                i_p = initial_descent.index(p)
                i_ = i_p
                while i_p < i:
                    i_p += 1
                    p_ = initial_descent[i_p]
                    cost_descent += grey_levels.cost(p_, p)
                if cost <= cost_descent:
                    for k in range(i-1, i_, -1):
                        if initial_descent[k] in final_descent:
                            final_descent.remove(initial_descent[k])
                            list_cost.pop(-1)
                    break
                else:
                    cost = cost_descent
        final_descent.append(point)
        cost_ = list_cost[-1] + cost 
        list_cost.append(cost_)
    print("coût du nouveau chemin : ", list_cost[-1])
    print("longueur du chemin final", len(final_descent))
    return final_descent

def compute_gradient_magnitude(grey_img: ui.GreyImage) -> np.ndarray:
    """
    Point 1 : Compute the gradient magnitude image |∇f| (image IG in the subject)
    Uses a simple 3x3 Sobel gradient. See https://fr.wikipedia.org/wiki/Filtre_de_Sobel
    Returns a 2D ndarray of the same size as the image, with floats.
    """
    arr = grey_img.to_numpy_array()  # shape (height, width), values 0..255
    arr = arr.astype(float)

    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=float)

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=float)

    grad_x = np.zeros_like(arr)
    grad_y = np.zeros_like(arr)

    h, w = arr.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            grad_x[y, x] = np.sum(arr[y-1:y+2, x-1:x+2] * sobel_x)
            grad_y[y, x] = np.sum(arr[y-1:y+2, x-1:x+2] * sobel_y)

    # TO REMOVE:
    # Normalisation optionnelle pour visualisation (0..255)
    # magnitude_vis = np.clip(magnitude / magnitude.max() * 255, 0, 255).astype(np.uint8) if magnitude.max() > 0 else magnitude.astype(np.uint8)
 
    return grad_x,grad_y 

def gradient_descent_Sobel(grey_levels: ui.GreyImage, start_point: pc.Point, end_point: pc.Point) -> list[pc.Point]:
    start = time.time()
    def mini(neighbors, grad_x, grad_y, point: pc.Point, start_point: pc.Point, visited: list[pc.Point]) -> pc.Point:
        diff_x = start_point.x - point.x
        diff_y = start_point.y - point.y
        mini_point = None
        #p = [pc.Point(point.x+int(copysign(1,diff_x)), point.y),pc.Point(point.x, point.y+int(copysign(1,diff_y)))]
        #if (p[0] in neighbors and visited[p[0]] == False) and (p[1] in neighbors and visited[p[1]] == False):
        #    if abs(grad_y[p[1].y, p[1].x]) > abs(grad_x[p[0].y, p[0].x]):
        #        mini_point = p[0]
        #    elif abs(grad_y[p[1].y, p[1].x]) < abs(grad_x[p[0].y, p[0].x]):
        #        mini_point = p[1]
        #    else:
        #        if diff_x < diff_y:
        #            mini_point = p[1]
        #        else : 
        #            mini_point = p[0]
        #elif p[0] in neighbors and visited[p[0]] == False and (p[1] not in neighbors or visited[p[1]] == True):
        #    mini_point = p[0]
        #elif (p[0] not in grey_levels.graph or visited[p[0]] == True) and p[1] in grey_levels.graph and visited[p[1]] == False:
        #    mini_point = p[1]
        mini_point = None
        neighbours_new = []
        for i in range(len(neighbors)):
            if neighbors[i] in list_visited:
                if visited[neighbors[i]] == False:
                    neighbours_new.append(neighbors[i])
        if len(neighbours_new) == 1:
            mini_point = neighbours_new[0]
        if pc.Point(point.x-copysign(1,grad_x[point.y, point.x]), point.y) in neighbours_new :
                if (abs(grad_x[point.y, point.x]) < abs(grad_y[point.y, point.x]) or grad_y[point.y, point.x] == 0) :
                        mini_point = pc.Point(point.x-int(copysign(1,grad_x[point.y, point.x])), point.y)
                
        if pc.Point(point.x, point.y-copysign(1,grad_y[point.y, point.x])) in neighbours_new :
                if (abs(grad_y[point.y, point.x]) < abs(grad_x[point.y, point.x]) or grad_x[point.y, point.x] == 0) :
                    mini_point = pc.Point(point.x, int(point.y-copysign(1,grad_y[point.y, point.x])))
                
        if mini_point is None:
            diff_x = point.x - start_point.x
            diff_y = point.y - start_point.y 
            if pc.Point(point.x-int(copysign(1,diff_x)), point.y) in neighbours_new :
                    if abs(diff_x) > abs(diff_y):
                        mini_point = pc.Point(point.x-int(copysign(1,diff_x)), point.y)
                    elif abs(diff_x) == abs(diff_y):
                        mini_point = pc.Point(point.x-int(copysign(1,diff_x)), point.y)
            
            if pc.Point(point.x, point.y-int(copysign(1,diff_y))) in neighbours_new :
                    if abs(diff_y) > abs(diff_x):
                        mini_point = pc.Point(point.x, point.y- int(copysign(1,diff_y)))
                    elif abs(diff_x) == abs(diff_y):
                        mini_point = pc.Point(point.x, point.y- int(copysign(1,diff_y)))
        return mini_point
    
    start = time.time()
    current = end_point
    path = [current]
    grad_x,grad_y = compute_gradient_magnitude(grey_levels)
    visited = {}
    for p in grey_levels.graph:
        visited[p] = False
    visited[current] = True
    while current != start_point:
        #print(grad_x[current], grad_y[current])
        neighbors = grey_levels.neighbors(current)
        best = mini(neighbors, grad_x, grad_y, current, start_point, visited)
        #print(neighbors, best, current)
        if best is None:
            path.pop()
            current = path[-1]
        else:
            visited[best] = True
            current = best
            path.append(current)
    path.reverse()
    final_descent = [path[0]]
    list_cost = [0]
    cost_ = 0
    for i in range(1, len(path)):
        point = path[i]
        cost = grey_levels.cost(point, final_descent[-1])
        neighbours = grey_levels.neighbors(point)
        for p in neighbours:
            construct_descent = final_descent[:-1]
            if p in construct_descent:
                cost = grey_levels.cost(point, p)
                cost_descent = 0
                i_p = path.index(p)
                i_ = i_p
                while i_p < i:
                    i_p += 1
                    p_ = path[i_p]
                    cost_descent += grey_levels.cost(p_, p)
                if cost <= cost_descent:
                    for k in range(i-1, i_, -1):
                        if path[k] in final_descent:
                            final_descent.remove(path[k])
                            list_cost.pop(-1)
                    break
                else:
                    cost = cost_descent
        final_descent.append(point)
        cost_ = list_cost[-1] + cost 
        list_cost.append(cost_)
    print("coût du nouveau chemin : ", list_cost[-1])
    print("longueur du chemin final : ", len(final_descent))
    end = time.time()
    print("temps d'execution : ", end-start)
    print("longueur du chemin initial", len(path))
    return final_descent


if __name__ == "__main__":
    im = ui.GreyImage('EZEZEZEZ.png')
    #im = ui.GreyImage('Carte.png')
    print(im.width, im.height)
    start = pc.Point(56,42)
    end = pc.Point(1178,419)
    #start = pc.Point(170,296)
    #end = pc.Point(53,51)
    list_visited = []
    distances = distances_costs(start, end, im, list_visited)
    distances = distances_costs(start, end, im, list_visited)
    print("distances okay", distances[end])
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
    #img.show()
    grad_image = gradient_on_image(distances, im)
    print("grad image okay")
    for k in range(10):
        grad_image[min(start.y+k,700), start.x] = [0,0,0]
        grad_image[start.y-k, start.x] = [0,0,0]
        grad_image[start.y, min(start.x+k,1324)] = [0,0,0]
        grad_image[start.y, start.x-k] = [0,0,0]
        grad_image[min(end.y+k,700), end.x] = [0,255,0]
        grad_image[end.y-k, end.x] = [0,255,0]
        grad_image[end.y, min(end.x+k,1324)] = [0,255,0]
        grad_image[end.y, end.x-k] = [0,255,0]
    
    grad_image_ = ui.Image.fromarray(grad_image, 'RGB')
    #grad_image_.show()
    descent_amelioration = amelioration_descent(distances, im, start, end, list_visited)
    final_img_a = affiche_descent(descent_amelioration, grad_image)
    #final_img_a = ui.Image.fromarray(final_img_a, 'RGB')
    #final_img_a.show()
    print("Sobel")
    descent_sobel = gradient_descent_Sobel(im, start, end)
    print("end Sobel")
    final_img_s = affiche_descent(descent_sobel, final_img_a, 1)
    final_img_s = ui.Image.fromarray(final_img_s, 'RGB')
    final_img_s.show()