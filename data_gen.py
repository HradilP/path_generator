import utils as ut
import sff

import shapely as sp
import time
import datetime

# basic working example
config = ut.load_yaml("dgen_cfg.yaml")

x_max = config["env"]["x_max"]
y_max = config["env"]["y_max"]

start = (0, 0)
end = (x_max, y_max)

obs = [sp.Point(3, 6).buffer(1), sp.Point(6, 3).buffer(1)] # can be a list of any shapely geometry

s_time = time.time()
n_paths = 10000
generated = 0
solver = sff.SFFsolver(obs, config)

while generated < n_paths:
    start_id, end_id = solver.gen_seeds(start, end)

    print("Running solver")
    point_paths = solver.sff()

    print("Creating graph")
    id_graph = ut.graph_from_paths(point_paths)
    print("Getting ID paths")
    id_paths = ut.all_paths_DFS(id_graph, start_id, end_id)
    print("Getting actual paths")
    paths = ut.get_paths(id_paths, point_paths)
    print(f"New paths: {len(paths)}")
    
    print("Resampling\n")
    resampled_paths = ut.resample_paths(paths, obs, config, smooth=False) # smooth=True for reasonably nice curves, slows the process a lot
    generated += len(resampled_paths)

taken = time.time() - s_time
print(f"Paths generated: {generated}")
print(f"Time taken: {datetime.timedelta(seconds=taken)}")