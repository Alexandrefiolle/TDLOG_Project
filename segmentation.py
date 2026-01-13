import point_class as pc
import manipulation as ui
import numpy as np
from math import*
import random
import dijkstra as d
import observer as obs

def points(nb_points: int, height: int, width: int) -> list[pc.Point]:
    list_points = []
    for _ in range(nb_points):
        x = random.randint(0, width)
        y = random.randint(0, height)
        p = pc.Point(x, y)
        list_points.append(p)
    return list_points

def distances_costs(start: pc.Point, grey_levels: ui.GreyImage, obs : obs.Observer|None = None) -> dict[pc.Point, float]:
    """Computes the list of shortest path costs from start until we reach the end point"""
    dist = ui.NumpyDict(grey_levels)
    dist[start] = 0
    to_visit = d.PriorityQueue_heap([])
    to_visit.append(start, 0)
    set_visited = set()
    while to_visit.size() > 0:
        candidate = to_visit.remove()
        if obs is not None:
            obs.notify_observer(grey_levels.width*grey_levels.height - len(set_visited))
        set_visited.add(candidate)
        for neighbor in grey_levels.neighbors(candidate):
            assert neighbor.x < grey_levels.width and neighbor.y < grey_levels.height
            cost = grey_levels.cost(start, neighbor, d.epsilon)
            if dist[neighbor] > dist[candidate] + cost:
                dist[neighbor] = dist[candidate] + cost
                to_visit.append(neighbor, dist[neighbor])
    return dist

def distances_map(list_point: list[pc.Point], im: ui.GreyImage, obs : obs.Observer|None = None) -> list[np.ndarray]:
    list_distance_map = []
    list_colored_map = []
    for p in list_point:
        dist = distances_costs(p, im, obs)
        dist_maps = d.coloration_map(dist, im)
        list_distance_map.append(dist)
        list_colored_map.append(dist_maps)
        if obs is not None:
            obs.notify_observer(-im.width*im.height)
    return list_distance_map, list_colored_map

def choice_segmentation_v1(list_point: list[pc.Point], list_distance_map: list[dict[pc.Point, float]], 
                        grey_levels: ui.GreyImage, obs:obs.Observer|None = None) -> np.ndarray:
    colored_map = np.zeros((grey_levels.height, grey_levels.width, 3), dtype=np.uint8)
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0], [255, 255, 0], [0, 255, 255], [255, 0, 255], [192, 192, 192]]
    cpt = grey_levels.width*grey_levels.height
    for y in range(grey_levels.height):
        for x in range(grey_levels.width):
            if obs is not None:
                cpt -= 1
                obs.notify_observer(cpt)
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
    #im = ui.GreyImage('EZEZEZEZ.png')
    im = ui.GreyImage('images/Carte.png')
    print("height: ", im.height, ", width : ", im.width)
    r = 3 #random.randint(3,10)
    print("nb of points : ", r)
    list_point = points(r, im.height, im.width)
    list_distance_map, list_colored_map = distances_map(list_point, im)
    segmentation = choice_segmentation_v1(list_point, list_distance_map, im)
    for k in range(r):
        print("point ", k+1, ": ", list_point[k])
        for i in range(10):
            list_colored_map[k][min(list_point[k].y+i,592), list_point[k].x] = [0,0,0]
            list_colored_map[k][list_point[k].y-i, list_point[k].x] = [0,0,0]
            list_colored_map[k][list_point[k].y, min(list_point[k].x+i,1243)] = [0,0,0]
            list_colored_map[k][list_point[k].y, list_point[k].x-i] = [0,0,0]
            segmentation[min(list_point[k].y+i,592), list_point[k].x] = [0,0,0]
            segmentation[list_point[k].y-i, list_point[k].x] = [0,0,0]
            segmentation[list_point[k].y, min(list_point[k].x+i,1243)] = [0,0,0]
            segmentation[list_point[k].y, list_point[k].x-i] = [0,0,0]
        img = ui.Image.fromarray(list_colored_map[k], 'RGB')
        img.show()
    im_s = ui.Image.fromarray(segmentation, 'RGB')
    im_s.show()