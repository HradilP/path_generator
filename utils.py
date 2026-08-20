import math
import random as rnd
from pathlib import Path
from typing import Any
import shapely as sp

import yaml
from shapely import Geometry


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as a YAML file."""
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file)


def gen_pt_radius(
    x: float, y: float, max_rad: float, min_rad: float = 0
) -> tuple[tuple[float, float], float]:
    """Generate a new point inside a circle of set radius around a given center point."""
    angle = rnd.uniform(0, 2 * math.pi)
    radius = math.sqrt(rnd.uniform(min_rad**2, max_rad**2))

    x += radius * math.cos(angle)
    y += radius * math.sin(angle)

    return (x, y), radius


def get_pt_seg_dist(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
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


def geo_in_obst(geometry: Geometry, obstacles: list[Geometry]) -> bool:
    """Check whether a geometry object is inside the given set of obstacles."""
    for obs in obstacles:
        if obs.contains(geometry):
            return True

    return False


def geo_inter_obst(geometry: Geometry, obstacles: list[Geometry]) -> bool:
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


def gen_c_obs(num_obs: int, max_size: float, min_size: float, margin: float,
               x_max: float, y_max: float, max_attempts: int) -> tuple[list, list]:
    """Generate n circular obstacles inside a rectangular space given by coordinate limits
    with a guaranteed free margin on the edges"""

    obs = []
    obs_repr = [] # might have to actually order it
    generated = 0
    attempts = 0
    free_min = max_size + margin

    while generated < num_obs and attempts < max_attempts:
        attempts += 1
        center = gen_pt_space(free_min, free_min, x_max - free_min, y_max - free_min)
        radius = rnd.uniform(min_size, max_size)

        point = sp.Point(center)
        obs_w_margin = point.buffer(radius + margin)

        if geo_inter_obst(obs_w_margin, obs):
            continue

        obs.append(point.buffer(radius))
        obs_repr.append(((center), radius))

        generated += 1
    
    return obs, obs_repr

def gen_p_obs(num_obs: int, max_size: float, min_size: float, margin: float, x_max: float, y_max: float,
              verts_max: int, verts_min: int, min_angle: float, max_attempts: int) -> tuple[list, list]:
    """Generate n polygonal obstacles inside a rectangular space given by coordinate limits
    with a guaranteed free margin on the edges"""

    obs = []
    obs_repr = []
    generated = 0
    attempts = 0
    free_min = max_size + margin

    while generated < num_obs and attempts < max_attempts:
        attempts += 1
        points = []
        center = gen_pt_space(free_min, free_min, x_max - free_min, y_max - free_min)
        num_verts = rnd.randint(verts_min, verts_max)
        free_angle = 2 * math.pi - num_verts * min_angle
        angles = sorted([rnd.uniform(0, free_angle) for _ in range(num_verts)])

        for i in range(len(angles)):
            radius = rnd.uniform(min_size, max_size)
            x = center[0] + math.cos(angles[i] + (i) * min_angle) * radius
            y = center[1] + math.sin(angles[i] + (i) * min_angle) * radius
            points.append((x,y))

        poly = sp.Polygon(points)
        obs_w_margin = poly.buffer(margin, join_style=3)

        if geo_inter_obst(obs_w_margin, obs) or not poly.is_valid:
            continue

        obs.append(poly)
        obs_repr.append(points)

        generated += 1
    
    return obs, obs_repr

def gen_obstacles(gen_c: dict) -> tuple[list, list]:
    """Generate obstacles and their representation based on config file"""

    obs = []
    obs_repr = []
    num_obst = rnd.randint(gen_c["min_n_obs"], gen_c["max_n_obs"])

    x_max = gen_c["x_max"]
    y_max = gen_c["y_max"]
    max_size = gen_c["max_size_obs"]
    min_size = gen_c["min_size_obs"]
    margin = gen_c["obs_margin"]
    max_attempts = num_obst * gen_c["attempts_per_obs"]
    obs_type = gen_c["obs_type"]

    if obs_type == "c":
        obs, obs_repr = gen_c_obs(num_obst, max_size, min_size, margin, x_max, y_max, max_attempts)
    elif obs_type == "p":
        verts_min = gen_c["verts_min"]
        verts_max = gen_c["verts_max"]
        min_angle = math.radians(gen_c["min_angle"])
        obs, obs_repr = gen_p_obs(num_obst, max_size, min_size, margin, x_max, y_max,
                                  verts_max, verts_min, min_angle, max_attempts)
    else:
        pass

    return obs, obs_repr