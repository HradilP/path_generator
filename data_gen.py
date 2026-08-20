from pathlib import Path
import argparse

import torch as tr
import concurrent.futures

from tqdm import tqdm
import utils as ut
import dgen_utils as dut
import sff


def generate_data_worker(n_samples, config):
    gen_c = config["generation"]

    n_points = gen_c["n_points"]
    point_dim = gen_c["point_dim"]

    data = {
        "paths":tr.empty((n_samples, n_points, point_dim), dtype=tr.float32),
        "obstacles":[],
    } # obstacles is going to be of variable size
    extra = {"trees": [], "num_paths": [], "raw_paths": [], "seeds": [], "raw_point_paths": []}

    start = (0, 0)
    x_max = gen_c["x_max"]
    y_max = gen_c["y_max"]
    end = (x_max, y_max)

    total_sampled = 0

    while total_sampled < n_samples: # possibly change to fixed amount of paths per obstacle config
        obs, obs_repr = ut.gen_obstacles(gen_c)
        seeds, start_id, end_id = dut.gen_seeds(start, end, obs, gen_c)

        solver = sff.SFFsolver(seeds, obs, config)
        point_paths, trees = solver.sff()

        id_graph = dut.graph_from_paths(point_paths)
        id_paths = dut.all_paths_DFS(id_graph, start_id, end_id)

        if id_paths == []:
            continue

        paths = dut.get_paths(id_paths, point_paths)
        resampled_paths = dut.resample_paths(paths, obs, gen_c)

        curr_sampled = 0

        for path in resampled_paths:
            if total_sampled >= n_samples:
                break

            data["paths"][total_sampled] = tr.tensor(path)
            data["obstacles"].append(obs_repr)
            extra["raw_point_paths"].append(paths[curr_sampled])
            total_sampled += 1
            curr_sampled += 1

        extra["trees"].append(trees)
        extra["num_paths"].append(curr_sampled)
        extra["raw_paths"].append(point_paths)
        extra["seeds"].append(seeds)
    
    return data, extra

def generate_data(config):
    gen_c = config["generation"]
    n_samples = gen_c["n_samples"]
    n_batches = gen_c["n_batches"]

    aggregated_data = {"paths": [], "obstacles": []}
    aggregated_extra = {"trees": [], "num_paths": [], "raw_paths": [], "seeds": [], "raw_point_paths" : []}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(generate_data_worker, n_samples, config) for _ in range(n_batches)]

        for future in tqdm(concurrent.futures.as_completed(futures), desc="Batch n.", total=n_batches):
            try:
                batch_data, batch_extra = future.result()
                
                aggregated_data["paths"].append(batch_data["paths"])
                aggregated_extra["raw_point_paths"].extend(batch_extra["raw_point_paths"])
                aggregated_data["obstacles"].extend(batch_data["obstacles"])
                aggregated_extra["trees"].extend(batch_extra["trees"])
                aggregated_extra["num_paths"].extend(batch_extra["num_paths"])
                aggregated_extra["raw_paths"].extend(batch_extra["raw_paths"])
                aggregated_extra["seeds"].extend(batch_extra["seeds"])

            except Exception as e:
                print(f"A worker encountered an error: {e}")

    if aggregated_data["paths"]:
            aggregated_data["paths"] = tr.cat(aggregated_data["paths"], dim=0)
    else:
        raise Exception

    return aggregated_data, aggregated_extra


if __name__ == "__main__":
    config = ut.load_yaml("dgen_cfg.yaml")
    parser = argparse.ArgumentParser(description="Generate dataset from config.")

    parser.add_argument(
            "-o", 
            type=str, 
            default="default", 
            help="Optional override for the output folder path (relative to working or absolute)"
        )
    
    parser.add_argument(
            "-n_samples", 
            type=int, 
            default=config["generation"]["n_samples"], 
            help="Optional override for number of samples"
        )
    
    parser.add_argument(
            "-n_batches", 
            type=int, 
            default=config["generation"]["n_batches"], 
            help="Optional override for number of batches"
        )
    
    parser.add_argument(
            "-obs_type", 
            type=str, 
            default=config["generation"]["obs_type"], 
            help="Optional override for type of obstacles"
        )
    
    args = parser.parse_args()
    config["generation"]["n_samples"] = args.n_samples
    config["generation"]["n_batches"] = args.n_batches
    config["generation"]["obs_type"] = args.obs_type
    out_dir = Path(config["paths"]["default_pt"]).joinpath(args.o)

    data, extra = generate_data(config)
    # data, extra = generate_data_worker(100, config)
    metakeys = [
        "n_samples", "n_batches", "n_points", "point_dim", "n_seeds", "x_max", "y_max", "obs_type", "min_n_obs", "max_n_obs"]
    metadata = {key:config["generation"][key] for key in metakeys}

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir.joinpath(config["paths"]["dset_fn"])
    extra_path = out_dir.joinpath(config["paths"]["extra_fn"])
    meta_path = out_dir.joinpath(config["paths"]["meta_fn"])

    tr.save(data, dataset_path)
    tr.save(extra, extra_path)
    ut.save_yaml(metadata, meta_path)
    print("Dataset saved!")