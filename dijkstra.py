import point_class as pc
import interface as ui
from collections import deque
import numpy as np

epsilon = 2.0

class PriorityQueue:
    def __init__(self, dic: dict[pc.Point, float]) -> None:
        self._dic = dic
    
    def _find_higher_priority_point(self) -> pc.Point:
        dist_inf = np.inf
        best_point = None
        for (point, distance) in self._dic:
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
        for neighbor in candidate.neighbors():
            cost = grey_levels.cost(candidate, neighbor, epsilon)
            if dist[neighbor] > dist[candidate] + cost:
                dist[neighbor] = dist[candidate] + cost
                to_visit.append(neighbor, dist[neighbor])
    return dist