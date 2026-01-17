import point_class as pc
import manipulation as ui
import numpy as np
from math import*
import random
import dijkstra as d
from multiprocessing import Pool
import random
import time


def points(nb_points: int, height: int, width: int) -> list[pc.Point]:
    list_points = []
    for _ in range(nb_points):
        x = random.randint(0, width)
        y = random.randint(0, height)
        p = pc.Point(x, y)
        list_points.append(p)
    return list_points

def distances_costs(start: pc.Point, grey_levels: ui.GreyImage, list_visited: list[pc.Point]) -> dict[pc.Point, float]:
    """Computes the list of shortest path costs from start until we reach the end point"""
    dist = {}
    for point in grey_levels.graph.keys():
        dist[point] = np.inf
    dist[start] = 0
    to_visit = d.PriorityQueue_heap([])
    to_visit.append(start, 0)
    while to_visit.size() > 0:
        candidate = to_visit.remove()
        list_visited.append(candidate)
        for neighbor in grey_levels.neighbors(candidate):
            assert neighbor.x < grey_levels.width and neighbor.y < grey_levels.height
            cost = grey_levels.cost(start, neighbor, d.epsilon)
            if dist[neighbor] > dist[candidate] + cost:
                dist[neighbor] = dist[candidate] + cost
                to_visit.append(neighbor, dist[neighbor])
    return dist

def compute_distance_for_point(args):
    p, im = args
    dist = distances_costs(p, im, [])
    return dist

def distances_map(list_point: list[pc.Point], im: ui.GreyImage) -> list[np.ndarray]:
    args = [(p, im) for p in list_point]
    
    with Pool() as pool:
        list_distance_map = pool.map(compute_distance_for_point, args)
    
    list_colored_map = []
    return list_distance_map, list_colored_map

def choice_segmentation_v1(list_point: list[pc.Point], list_distance_map: dict[pc.Point, float], 
                        grey_levels: ui.GreyImage) -> np.ndarray:
    colored_map = np.zeros((grey_levels.height, grey_levels.width, 3), dtype=np.uint8)
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0]]
    for y in range(grey_levels.height):
        for x in range(grey_levels.width):
            p = pc.Point(x,y)
            mini_dist = np.inf
            k_inf = 0
            for k in range(len(list_point)):
                if list_distance_map[k][p] < mini_dist:
                    k_inf = k
                    mini_dist = list_distance_map[k][p]
                elif list_distance_map[k][p] == mini_dist:
                    k_inf = len(colors) - 1
                    mini_dist = list_distance_map[k][p]
            colored_map[y, x] = colors[k_inf]
    return colored_map
            
if __name__ == "__main__":
    start = time.time()
    #im = ui.GreyImage('EZEZEZEZ.png')
    im = ui.GreyImage('Carte.png')
    print("height: ", im.height, ", width : ", im.width)
    r = 3 #random.randint(3,10)
    print("nb of points : ", r)
    list_point = points(r, im.height, im.width)
    list_distance_map, list_colored_map = distances_map(list_point, im)
    segmentation = choice_segmentation_v1(list_point, list_distance_map, im)
    im_s = ui.Image.fromarray(segmentation, 'RGB')
    im_s.show()
    print(time.time() - start)

