import utils as ut
import shapely as sp
import numpy as np
from shapely.ops import nearest_points, unary_union


def gen_pt_free_space(x_min: float, y_min: float, x_max: float, y_max: float,
                      obstacles: list[sp.Geometry]) -> tuple[float, float]:
    """Generate a point in a rectangular space given by the limits that isnt inside an obstacle"""

    contains = True
    point = None

    while contains:
        point = ut.gen_pt_space(x_min, y_min, x_max, y_max)
        point_obj = sp.Point(point)

        contains = ut.geo_in_obst(point_obj, obstacles)

    return point

def gen_seeds(start: tuple[float, float], end: tuple[float, float],
              obs: list[sp.Geometry], gen_c: dict) -> tuple[list, int, int]:
    """Generates seeds for running SFF in a 2D space, returns a list of seeds as well as the indexes of
    the passed start point and end point"""

    seeds = [start] # id 0

    for _ in range (gen_c["n_seeds"]):
        seeds.append(gen_pt_free_space(0, 0, gen_c["x_max"], gen_c["y_max"], obs))

    seeds.append(end) # id max index

    return seeds, 0, gen_c["n_seeds"] + 1

def graph_from_paths(paths: dict) -> dict:
    """Generates 'abstract' (id based) graph from path dict as dictionary of neighbors for each point"""

    graph = {}

    for p1, p2 in paths:

        if p1 in graph:
            graph[p1].append(p2)
        else:
            graph[p1] = [p2]
        
        if p2 in graph:
            graph[p2].append(p1)
        else:
            graph[p2] = [p1]

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
    
def get_paths(id_paths: list, point_paths: dict) -> list:
    """Creates actual point paths from path segments created by SFF marked with IDs based on 'abstract' ID paths
    created for some start and goal"""

    paths = []

    for id_path in id_paths:
        path = []

        for i in range(len(id_path) - 1):
            if id_path[i] < id_path[i+1]:
                id = (id_path[i], id_path[i+1])
                path_seg = point_paths[id]
            else:
                id = (id_path[i+1], id_path[i])
                path_seg = list(reversed(point_paths[id]))
            
            path.extend(path_seg)
        
        path = remove_duplicate_points(path)
        paths.append(path)

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

    obs_buffers = [obs.buffer(safety_margin - 0.1, join_style=3) for obs in obstacles]
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

def moving_avg_path_smoothing(path: list, window_size: int) -> list:
    """Smoothes path by assigning the average of the nearby 'window_size' points (on each side) to
    each point"""

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
        
        smoothed.append((x_avg, y_avg))
        
    smoothed[0] = path[0]
    smoothed[-1] = path[-1]
    
    return smoothed

def physical_path_smoothing(path: list, obstacles: list[sp.Geometry], safety_margin: float, influence_radius: float,
                            iterations: int, alpha: float, beta: float) -> list:
    """Smoothes path using physical based 'spring' like relaxation alongside force-field based obstacle avoidance"""

    obs_union = unary_union(obstacles)
    
    for _ in range(iterations):
        new_path = list(path)
        
        for i in range(1, len(path) - 1):
            p = path[i]
            p_point = sp.Point(p)
            dist = float(sp.distance(p_point, obs_union))
            
            repulsion_x = 0
            repulsion_y = 0

            if dist < 0.001:
                pass # get normal vector of nearest obstacle edge
            elif dist < influence_radius:
                nearest_geom, _ = nearest_points(obs_union, p_point)
                vec_x = p[0] - nearest_geom.x
                vec_y = p[1] - nearest_geom.y

                normed_x  = vec_x / dist
                normed_y  = vec_y / dist
                
                if dist <= safety_margin:
                    repulsion_x = normed_x * (safety_margin - dist + 1) * 2
                    repulsion_y = normed_y * (safety_margin - dist + 1) * 2
                else:
                    ratio = (influence_radius - dist) / (influence_radius - safety_margin)
                    repulsion_x = normed_x * beta * (ratio ** 2)
                    repulsion_y = normed_y * beta * (ratio ** 2)
                    
            spring_x = alpha * (path[i-1][0] + path[i+1][0] - 2 * p[0])
            spring_y = alpha * (path[i-1][1] + path[i+1][1] - 2 * p[1])

            new_x = p[0] + repulsion_x + spring_x
            new_y = p[1] + repulsion_y + spring_y

            new_path[i] = (new_x, new_y)
            
        path = new_path
        
    return path

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

def resample_paths(paths: list[list], obstacles: list[sp.Geometry], gen_c: dict) -> list[np.ndarray]:
    """Smoothes and resamples all paths based on config file settings"""

    resampled_paths = []
    n_points = gen_c["n_points"]
    safety_margin = (gen_c["obs_margin"] / 2) - gen_c["safety_margin_reduction"]
    influence_radius = gen_c["influence_radius"]

    for path in paths:
        for _ in range(gen_c["init_smoothing_iters"]):
            path = moving_avg_path_smoothing(path, gen_c["avg_smoothing_window"])
            path = obstacle_intersection_repair(path, obstacles, safety_margin)
    
        path = physical_path_smoothing(
            path, obstacles, safety_margin, influence_radius, iterations=gen_c["relaxation_iters"],
            alpha=gen_c["smoothing_c"], beta=gen_c["repulsion_c"])
        
        path = obstacle_intersection_repair(path, obstacles, safety_margin)
        path = redistribute_points(path, n_points)

        resampled_paths.append(path)

    return resampled_paths