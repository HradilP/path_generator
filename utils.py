import math
import numpy as np
import random as rnd
import yaml
import shapely as sp
import copy
from itertools import product
from shapely.ops import nearest_points, unary_union

from typing import Any
from pathlib import Path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as a YAML file."""
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file)


def gen_pt_radius(x: float, y: float, max_rad: float, min_rad: float = 0) -> tuple[tuple[float, float], float]:
    """Generate a new point inside a circle of set radius around a given center point."""
    angle = rnd.uniform(0, 2 * math.pi)
    radius = math.sqrt(rnd.uniform(min_rad**2, max_rad**2))

    x += radius * math.cos(angle)
    y += radius * math.sin(angle)

    return (x, y), radius


def get_pt_pt_dist(p1: tuple, p2: tuple) -> float:
    """Get point to point distance"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def get_pt_seg_dist(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Get the distance between a 2D point and the segment defined by points A and B."""
    px, py = p
    ax, ay = a
    bx, by = b

    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay

    ab_len_sq = (abx * abx) + (aby * aby)
    if ab_len_sq == 0:
        return math.sqrt((apx * apx) + (apy * apy)), 0.0

    t = (apx * abx + apy * aby) / ab_len_sq
    t = 1 if t > 1 else t
    t = 0 if t < 0 else t

    closest_x = ax + t * abx
    closest_y = ay + t * aby

    dx = px - closest_x
    dy = py - closest_y
    return math.sqrt((dx * dx) + (dy * dy)), t


def geo_in_obst(geometry: sp.Geometry, obstacles: list[sp.Geometry]) -> bool:
    """Check whether a geometry object is inside the given set of obstacles."""
    for obs in obstacles:
        if obs.contains(geometry):
            return True

    return False


def geo_inter_obst(geometry: sp.Geometry, obstacles: list[sp.Geometry]) -> bool:
    """Check whether a geometry object is intersecting the given set of obstacles."""
    for obs in obstacles:
        if obs.intersects(geometry):
            return True

    return False


def gen_pt_space(x_min: float, y_min: float, x_max: float, y_max: float) -> tuple[float, float]:
    """Generate a point in a rectangular space given by the limits"""
    x = rnd.uniform(x_min, x_max)
    y = rnd.uniform(y_min, y_max)

    return (x, y)


def gen_pt_free_space(x_min: float, y_min: float, x_max: float, y_max: float,
                      obstacles: list[sp.Geometry]) -> tuple[float, float]:
    """Generate a point in a rectangular space given by the limits that isnt inside an obstacle, assumes that enough free space exists (otherwise can create an infinite loop)"""

    contains = True
    point = None

    while contains:
        point = gen_pt_space(x_min, y_min, x_max, y_max)
        point_obj = sp.Point(point)

        contains = geo_in_obst(point_obj, obstacles)

    return point


def graph_from_paths(paths: dict) -> dict:
    """Generates 'abstract' (id based) graph from path dict as dictionary of neighbors for each point"""
    graph = {}

    for p1, p2 in paths:

        if p1 in graph:
            graph[p1].add(p2)
        else:
            graph[p1] = {p2}
        
        if p2 in graph:
            graph[p2].add(p1)
        else:
            graph[p2] = {p1}

    return graph


def all_paths_DFS(id_graph: dict, start_id: int, end_id: int) -> list:
    """Finds all possible paths in a graph from start point to end point"""
    paths = []
    open_paths = [[start_id]]

    if start_id not in id_graph or end_id not in id_graph:
        return []

    while open_paths:
        curr = open_paths.pop()

        if curr[-1] == end_id:
            paths.append(curr)
            continue
        
        for neighbor in id_graph[curr[-1]]:
            if neighbor not in curr: # could be optimised by using sets but kinda useless in our case
                open_paths.append(curr + [neighbor])
        
    return paths


def remove_duplicate_points(path: list) -> list:
    """Removes unncessary 'loose' ends which are often created by SFF"""
    cleaned_path = []
    points = set()

    for point in path:
        if point in points:
            while cleaned_path:
                if cleaned_path[-1] == point:
                    break
                else:
                    removed = cleaned_path.pop()
                    points.remove(removed)
        else:
            cleaned_path.append(point)
            points.add(point)
    
    return cleaned_path


def get_paths(id_paths: list, point_segments: list) -> list:
    """Creates point paths based on id paths and point segments connecting the ids"""
    paths = []

    for id_path in id_paths:
        options = []

        for i in range(len(id_path) - 1):
            tree_a, tree_b = id_path[i], id_path[i + 1]
            reverse = tree_a > tree_b
            key = (tree_b, tree_a) if reverse else (tree_a, tree_b)
            raw_segs = point_segments[key]

            if reverse:
                segs = [list(reversed(copy.deepcopy(seg))) for seg in raw_segs]
            else:
                segs = copy.deepcopy(raw_segs)

            options.append(segs)

        for choice in product(*options):
            path = []
            for seg in choice:
                path.extend(seg)
            paths.append(remove_duplicate_points(path))

    return paths


def repair_segment(p_start: tuple[float, float], p_end: tuple[float, float], obstacles: sp.Geometry,
                   boundary: sp.Geometry, max_depth: int, depth: int = 0) -> list:
    """Recursively subdivides segment in an attempt to fix obstacle intersection"""
    if depth > max_depth: 
        return [p_start, p_end]
        
    segment = sp.LineString([p_start, p_end])
    
    if segment.intersects(obstacles):
        mid_x = (p_start[0] + p_end[0]) / 2
        mid_y = (p_start[1] + p_end[1]) / 2
        mid_point = sp.Point(mid_x, mid_y)
        
        safe_mid, _ = nearest_points(boundary, mid_point)
        p_mid = (safe_mid.x, safe_mid.y)
        
        left_half = repair_segment(p_start, p_mid, obstacles, boundary, max_depth, depth + 1)
        right_half = repair_segment(p_mid, p_end, obstacles, boundary, max_depth, depth + 1)
        
        return left_half[:-1] + right_half
        
    return [p_start, p_end]


def obstacle_intersection_repair(path: list, obstacles: list[sp.Geometry], safety_margin: float) -> list:
    """Pushes all points away from obstacles and subvidivides segments if they
    are intersecting obstacles"""
    obs_buffers = [obs.buffer(safety_margin * 0.01, join_style=3) for obs in obstacles]
    obstacle_zone = unary_union(obs_buffers)
    
    push_buffers = [obs.buffer(safety_margin, join_style=3) for obs in obstacles]
    push_boundary = unary_union(push_buffers).boundary

    fixed_path = [path[0]]

    for i in range(1, len(path)):
        p1 = path[i]
        p_point = sp.Point(p1)
        
        if p_point.intersects(obstacle_zone):
            safe_p, _ = nearest_points(push_boundary, p_point)
            p1 = (safe_p.x, safe_p.y)
            
        p2 = fixed_path[-1]
        repaired_segment = repair_segment(p2, p1, obstacle_zone, push_boundary, 10)
        fixed_path.extend(repaired_segment[1:])
        
    return fixed_path


def moving_avg_path_smoothing(path: list, obstacles: list, margin: int, window_size: int) -> list:
    """Smoothes path by assigning the average of the nearby 'window_size' points (on each side) to
    each point"""
    obs_buffers = [obs.buffer(margin, join_style=3) for obs in obstacles]
    obs_union = unary_union(obs_buffers)
    left_pad = [path[0]] * window_size
    right_pad = [path[-1]] * window_size
    path_padded = left_pad + path + right_pad
    
    smoothed = []
    total_win_len = 2 * window_size + 1
    
    for i in range(len(path)):
        vals = path_padded[i : i + total_win_len]
        x_vals, y_vals = zip(*vals)
        
        x_avg = sum(x_vals) / total_win_len
        y_avg = sum(y_vals) / total_win_len

        point = sp.Point(x_avg, y_avg)
        
        if geo_in_obst(point, obs_buffers):
            safe_p, _ = nearest_points(obs_union, point)
            x_avg, y_avg = safe_p.x, safe_p.y
        
        smoothed.append((x_avg, y_avg))
        
    smoothed[0] = path[0]
    smoothed[-1] = path[-1]
    
    return smoothed


def redistribute_points(path: list, n_points: int) -> np.ndarray:
    """Evenly redistributes fixed amount of paths along the path using linear interpolation"""
    path = np.array(path)
    distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
    cumulative_dist = np.concatenate(([0], np.cumsum(distances)))
    total_dist = cumulative_dist[-1]

    target_dists = np.linspace(0, total_dist, n_points)

    x_resampled = np.interp(target_dists, cumulative_dist, path[:, 0])
    y_resampled = np.interp(target_dists, cumulative_dist, path[:, 1])

    path = np.vstack((x_resampled, y_resampled)).T
    return path


def resample_paths(paths: list[list], obstacles: list[sp.Geometry], config: dict, smooth: bool) -> list[np.ndarray]:
    """Smoothes and resamples all paths based on config file settings"""
    path_c = config["path"]
    resampled_paths = []
    n_points = config["path"]["n_points"]
    safety_margin = path_c["safety_margin"]

    for path in paths:
        if smooth:
            path = moving_avg_path_smoothing(path, obstacles, safety_margin, path_c["avg_smoothing_window"])
            path = obstacle_intersection_repair(path, obstacles, safety_margin)

        path = redistribute_points(path, n_points)
        resampled_paths.append(path)

    return resampled_paths