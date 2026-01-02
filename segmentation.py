import point_class as pc
import manipulation as ui
import numpy as np
from math import*
import random
import dijkstra as d

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

def distances_map(list_point: list[pc.Point], im: ui.GreyImage) -> list[np.ndarray]:
    list_distance_map = []
    for k in range(len(list_point)):
        dist = distances_costs(list_point[k], im, [])
        dist_maps = d.coloration_map(dist, im)
        list_distance_map.append(dist_maps)
    return list_distance_map

if __name__ == "__main__":
    im = ui.GreyImage('EZEZEZEZ.png')
    #im = ui.GreyImage('Carte.png')
    print(im.height, im.width)
    r = 1 #random.randint(1,10)
    print("nb of points : ", r)
    list_point = points(r, im.height, im.width)
    list_point = [pc.Point(170,296)]
    list_colored_map = distances_map(list_point, im)
    for k in range(r):
        print("point ", k, ": ", list_point[k])
        print()
        for i in range(10):
            list_colored_map[k][min(list_point[k].y+i,593), list_point[k].x] = [0,0,0]
            list_colored_map[k][list_point[k].y-i, list_point[k].x] = [0,0,0]
            list_colored_map[k][list_point[k].y, min(list_point[k].x+i,1244)] = [0,0,0]
            list_colored_map[k][list_point[k].y, list_point[k].x-i] = [0,0,0]
        img = ui.Image.fromarray(list_colored_map[k], 'RGB')
        img.show()

        