import utils as ut
import dgen_utils as dut
import sff

def main():
    config = ut.load_yaml("dgen_cfg.yaml")
    gen_c = config["generation"]

    start = (0, 0)
    x_max = gen_c["x_max"]
    y_max = gen_c["y_max"]
    end = (x_max, y_max)

    obs = [] # shapely geometry list

    # seeds, start_id, end_id = dut.gen_seeds(start, end, obs, gen_c)
    solver = sff.SFFsolver(obs, config)
    point_paths = solver.sff()

    id_graph = dut.graph_from_paths(point_paths)
    id_paths = dut.all_paths_DFS(id_graph, start_id, end_id)

    if id_paths == []:
        return

    paths = dut.get_paths(id_paths, point_paths)
    resampled_paths = dut.resample_paths(paths, obs, gen_c)