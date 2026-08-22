import utils as ut
import random as rnd
import shapely as sp


DIRS = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]


class Grid():
    """A simple grid lookup class to speed up processing when only points in some proximity need to be processed"""

    def __init__(self, res: float, x_max: float, y_max: float) -> None:
        self.res = res
        self.grid = {(i, j):[] for i in range (int(x_max // res) + 1)  for j in range (int(y_max // res) + 1)}

    def get_grid_coord(self, point: tuple[float, float]) -> tuple[int, int]:
        """Get grid coords (keys) of a given point"""

        return (int(point[0] // self.res), int(point[1] // self.res))

    def add_to_grid(self, point: tuple[float, float], tree_id: int) -> None:
        """Add a point to the lookup grid"""

        coord = self.get_grid_coord(point)
        self.grid[coord].append((point, tree_id))
    
    def get_cell_points(self, x:int, y:int) -> list:
        """Get points at the given grid key (if such a key exists)"""

        if (x, y) not in self.grid:
            return []
        else:
            return self.grid[(x,y)]
    
    def add_seeds(self, seeds: list) -> None:
        """Add SFF seed points to the grid"""

        for i in range(len(seeds)):
            self.add_to_grid(seeds[i], i)
    
    def get_close_points(self, point:tuple[float, float]) -> list:
        """Get all points from cells nearby to a given point"""

        points = []
        
        x, y = self.get_grid_coord(point)

        for dx, dy in DIRS:
            points.extend(self.get_cell_points(x + dx, y + dy))
        
        return points



class SFFsolver():
    """Space Filling Forest solver class for running SFF in a 2D space with obstacles"""

    def __init__(self, seeds: list, obstacles: list, config: dict) -> None:
        self.seeds = seeds
        self.obs = obstacles
        self.config = config

    def get_dist_in_tree(self, curr_tree_id: int, point: tuple[float, float]) -> float:
        """Get the distance of a point to the nearest point belonging to the given tree"""

        min_dist = float('inf')
        tree = self.trees[curr_tree_id]

        for edge_s, tree_id in self.grid.get_close_points(point):

            if tree_id != curr_tree_id:
                continue

            edge_e = tree[edge_s]
            dist, _ = ut.get_pt_seg_dist(point, edge_s, edge_e)
            
            if dist < min_dist:
                min_dist = dist
        
        return min_dist

    def get_closest_other_trees(self, point: tuple[float, float], curr_id: int) -> tuple:
        """Get the distance, coords and tree_id of the nearest point (from another tree) to the given point"""

        min_dist = float('inf')
        min_id = None
        min_point = None

        for edge_s, tree_id in self.grid.get_close_points(point):

                if tree_id == curr_id:
                    continue
                
                edge_e = self.trees[tree_id][edge_s]
                dist, t = ut.get_pt_seg_dist(point, edge_s, edge_e)

                if dist < min_dist:
                    min_dist = dist
                    min_id = tree_id

                    if t < 0.5:
                        min_point = edge_s
                    else:
                        min_point = edge_e

        return min_dist, min_point, min_id

    def backtrack(self, start: tuple[float, float], point_tree: dict) -> list:
        """Backtrack a path from a given start point through the given tree to its origin"""

        path = []
        curr_p = start
        
        while curr_p != point_tree[curr_p]:
            parent = point_tree[curr_p]
            path.append(curr_p)
            curr_p = parent
        
        path.append(curr_p) #append origin
        
        return path

    def construct_path(self, end_point1: tuple[float, float], end_point2: tuple[float, float],
                       tree1: dict, tree2: dict, reverse: bool) -> list:
        """Constructs a path from origin of one tree to origin of another tree based on given end points"""

        path = []

        path.extend(self.backtrack(end_point1, tree1))
        path.reverse()
        path.extend(self.backtrack(end_point2, tree2))
        
        if reverse:
            path.reverse() # make sure the first point of path_id is first in path
        
        return path

    def check_intersect(self, new_point: tuple[float, float], origin_point: tuple[float, float],
                        line: sp.LineString, curr_tree_id: int) -> bool:
        """Check whether a line from point 1 to point 2 intersects any edges in a its own tree"""

        tree = self.trees[curr_tree_id]
        edges = self.edges[curr_tree_id]

        for edge_s, tree_id in self.grid.get_close_points(new_point):

                if curr_tree_id != tree_id:
                    continue

                edge_e = tree[edge_s]

                if edge_s == origin_point or edge_e == origin_point:
                    continue

                edge = edges[edge_s]

                if edge.intersects(line):
                    return True
        
        return False    

    def sff(self) -> tuple:
        """Run the actual SFF algorithm"""

        seeds = self.seeds
        gen_c = self.config["generation"]
        sff_c = self.config["sff"]

        self.trees = {i:{seeds[i]:seeds[i]} for i in range(len(seeds))} # tree_id:{child:parent}
        frontier = [(seeds[i], i) for i in range(len(seeds))] # (point, tree_id)
        paths = {} # (tree1, tree2):path
        bridge_points = {}

        for i in range(len(seeds)):
            for j in range(0, i):
                bridge_points[(j, i)] = set()
        
        self.edges = {i:{seeds[i]:sp.LineString((seeds[i], seeds[i]))} for i in range(len(seeds))}
        self.grid = Grid(sff_c["grid_res"], gen_c["x_max"], gen_c["y_max"]) # coord:[(point, tree_id)]
        self.grid.add_seeds(seeds)

        while frontier:
            success = False
            rnd_id = rnd.randint(0, len(frontier) - 1)
            point, tree_id = frontier[rnd_id]
            trial = 0

            tree = self.trees[tree_id]
            edges = self.edges[tree_id]

            while trial < sff_c["trial_lim"]:
                trial += 1
                new_point, new_point_dist = ut.gen_pt_radius(point[0], point[1], sff_c["max_rad"])
                new_edge = sp.LineString((new_point, point))

                if new_point[0] > gen_c["x_max"] or new_point[0] < 0 or new_point[1] > gen_c["y_max"] or new_point[1] < 0:
                    trial -= 1 # point out of bounds doesnt count as an unsuccessfull trial
                    continue

                nearest_dist = self.get_dist_in_tree(tree_id, new_point)
                nearest_tree_dist, nearest_point, nearest_tree_id = self.get_closest_other_trees(new_point, tree_id)

                if nearest_tree_dist > sff_c["tree_dist"]: # tree colission check
                    if (
                        nearest_dist >= new_point_dist and
                        not ut.geo_inter_obst(new_edge, self.obs) and
                        not self.check_intersect(new_point, point, new_edge, tree_id)
                    ): # tree ingrowth and obstacles check
                        tree[new_point] = point
                        frontier.append((new_point, tree_id))
                        self.grid.add_to_grid(new_point, tree_id)
                        edges[new_point] = new_edge
                        success = True
                        break
                else:
                    reverse = nearest_tree_id < tree_id
                    path_id = (nearest_tree_id, tree_id) if reverse else (tree_id, nearest_tree_id)
                    connect_line = sp.LineString((nearest_point, point))

                    if ut.geo_inter_obst(connect_line, self.obs) or point in bridge_points[path_id] or nearest_point in bridge_points[path_id]:
                        continue

                    min_dist = max(gen_c["x_max"], gen_c["y_max"])

                    for br_point in bridge_points[path_id]:
                        dist1 = ut.get_pt_pt_dist(br_point, point)
                        dist2 = ut.get_pt_pt_dist(br_point, nearest_point)
                        dist = min(dist1, dist2)

                        if dist < min_dist:
                            min_dist = dist

                    if min_dist < sff_c["bridge_thresh"] and bridge_points[path_id]:
                        continue
                    
                    bridge_points[path_id].add(point)
                    bridge_points[path_id].add(nearest_point)
                    path = self.construct_path(point, nearest_point, self.trees[tree_id],
                                               self.trees[nearest_tree_id], reverse)

                    if path_id not in paths:
                        paths[path_id] = [path]
                    else:
                        paths[path_id].append(path)

            if not success:
                frontier.pop(rnd_id)

        return paths, self.trees