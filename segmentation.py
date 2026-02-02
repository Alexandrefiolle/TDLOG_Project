import point_class as pc
import manipulation as ui
import numpy as np
from math import*
import random
import dijkstra as d
import observer as obs
import time
from multiprocessing import Pool

def points(nb_points: int, height: int, width: int) -> list[pc.Point]:
    """Generates a list of random points within the given height and width."""
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

def compute_distance_for_point(args: tuple[pc.Point, ui.GreyImage]) -> dict[pc.Point, float]:
    """Helper function to compute distances for a given point and image."""
    p, im = args
    dist = distances_costs(p, im, None)
    return dist


def distances_map(list_point: list[pc.Point], im: ui.GreyImage, obs : obs.Observer|None = None) -> list[np.ndarray]:
    """Computes the distance maps for a list of points in the given image."""
    args = [(p, im) for p in list_point]
    list_distance_map = []
    list_colored_map = []
    if obs is not None:
        cpt = len(list_point)
        obs.notify_observer(cpt)
    with Pool() as pool: # Create a pool of worker processes
        for dist in pool.imap_unordered(compute_distance_for_point, args):
            list_distance_map.append(dist)
            if obs is not None:
                cpt -= 1
                obs.notify_observer(cpt)
    return list_distance_map, list_colored_map

def choice_segmentation_v1(list_point: list[pc.Point], list_distance_map: list[dict[pc.Point, float]], 
                        grey_levels: ui.GreyImage, obs:obs.Observer|None = None) -> np.ndarray:
    """Segments the image based on the closest point from the list of points using the distance maps."""
    colored_map = np.zeros((grey_levels.height, grey_levels.width, 3), dtype=np.uint8)
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0], [255, 255, 0], [0, 255, 255], [255, 0, 255], [192, 192, 192], [128, 0, 0], [128, 128, 0], [0, 128, 0], [128, 0, 128], [0, 128, 128], [0, 0, 128]]
    cpt = grey_levels.width*grey_levels.height
    obs.notify_observer(-cpt)
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
    start = time.time()
    im = ui.GreyImage('images/Carte.png')
    print("height: ", im.height, ", width : ", im.width)
    r = random.randint(3,8)
    print("nb of points : ", r)
    list_point = points(r, im.height, im.width)
    list_distance_map, list_colored_map = distances_map(list_point, im)
    segmentation = choice_segmentation_v1(list_point, list_distance_map, im)
    im_s = ui.Image.fromarray(segmentation, 'RGB')
    im_s.show()
    print(time.time() - start)