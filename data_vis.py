import argparse
from pathlib import Path
from sys import exit

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch as tr
import shapely as sp
import geopandas as gp
from utils import load_yaml
import dvis_utils as vut


class DatasetVisualizer:

    def __init__(self, meta, data, extra):
        self.meta = meta
        self.data = data
        self.extra = extra
        self.num_paths = extra["num_paths"]

        if self.meta["obs_type"] == "c":
            self.obst_drawer = self.c_drawer
        elif self.meta["obs_type"] == "p":
            self.obst_drawer = self.p_drawer
        else:
            pass

        self.small_ptr = 0
        self.big_ptr = 0
        self.mode = 0
        self.small_ptr_min = 0
        self.small_ptr_max = (meta["n_samples"] * meta["n_batches"]) - 1
        self.big_ptr_min = 0
        self.big_ptr_max = len(self.num_paths) - 1
        self.curr_num = self.num_paths[self.big_ptr]

        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(bottom=0.2)
        self.fig.subplots_adjust(top=0.95)
        self.fig.subplots_adjust(left=0.125)
        self.fig.subplots_adjust(right=0.925)

        self.axbut = self.fig.add_axes((0.35, 0.035, 0.3, 0.06))
        self.button = plt.Button(self.axbut, 'Path / SFF trees toggle', hovercolor='0.975')
        self.button.on_clicked(self.click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_press)

        self.draw_current()

    def c_drawer(self, obs_props, ax):
        for obs in obs_props:
            point = sp.Point(obs[0])
            obs_obj = point.buffer(obs[1])
            obs_drawable = gp.GeoSeries(obs_obj)
            obs_drawable.plot(ax=ax, color="gray")

    def p_drawer(self, obs_props, ax):
        for obs in obs_props:
            poly = sp.Polygon(obs)
            obs_drawable = gp.GeoSeries(poly)
            obs_drawable.plot(ax=ax, color="gray")

    def draw_current(self):
        self.ax.clear()
        self.ax.set_ylim(0, self.meta["y_max"])
        self.ax.set_xlim(0, self.meta["x_max"])

        obs_props = self.data["obstacles"][self.small_ptr]
        self.obst_drawer(obs_props, self.ax)

        if self.mode == 0:
            path = self.data["paths"][self.small_ptr]
            self.ax.plot(path[:, 0], path[:, 1], color="orange", zorder=5, marker="o", markersize=1)

        elif self.mode == 1:
            trees = self.extra["trees"][self.big_ptr]

            for i in range(self.meta["n_seeds"] + 2):
                color = vut.get_color(i)
                tree = trees[i]
                segs = []

                for edge_s in tree:
                    edge_e = tree[edge_s]
                    segs.append([edge_s, edge_e])

                line_segments = LineCollection(segs, linewidths=1, color=color, linestyle='solid')
                self.ax.add_collection(line_segments)

                seed = self.extra["seeds"][self.big_ptr][i]
                self.ax.scatter(seed[0], seed[1], color=color, edgecolors="black", s=15, zorder=10)
            
            paths = self.extra["raw_paths"][self.big_ptr]
    
            for path_id, path in paths.items():
                x, y = zip(*path)
                self.ax.plot(x, y, color="black", linestyle="dotted", zorder=7)
            
            taken_path = self.extra["raw_point_paths"][self.small_ptr]
            taken_x, taken_y = zip(*taken_path)
            self.ax.plot(taken_x, taken_y, color="black", linewidth=2, zorder=8)

        else: pass

        self.ax.set_title(f"Environment {self.big_ptr + 1} / {self.big_ptr_max + 1} | Path {self.small_ptr - (self.curr_num - self.num_paths[self.big_ptr]) + 1} / {self.num_paths[self.big_ptr]}")
        self.fig.canvas.draw_idle()
    
    def click(self, event):
        if self.mode == 0:
            self.mode = 1
        elif self.mode == 1:
            self.mode = 0
        else:
            raise Exception
        
        self.draw_current()

    def on_press(self, event):
        if event.key == "right" and self.small_ptr < self.small_ptr_max:
            self.small_ptr += 1

            if self.small_ptr == self.curr_num:
                self.big_ptr += 1
                self.curr_num += self.num_paths[self.big_ptr]

            self.draw_current()

        elif event.key == "left" and self.small_ptr > self.small_ptr_min:
            self.small_ptr -= 1

            if self.small_ptr == self.curr_num - self.num_paths[self.big_ptr] - 1:
                self.curr_num -= self.num_paths[self.big_ptr]
                self.big_ptr -= 1

            self.draw_current()

        elif event.key == "up" and self.big_ptr < self.big_ptr_max:
            self.small_ptr = self.curr_num
            self.big_ptr += 1
            self.curr_num += self.num_paths[self.big_ptr]

            self.draw_current()

        elif event.key == "down" and self.big_ptr > self.big_ptr_min:
            self.curr_num -= self.num_paths[self.big_ptr]
            self.big_ptr -= 1
            self.small_ptr = self.curr_num - self.num_paths[self.big_ptr]

            self.draw_current()
        
    def show(self):
        plt.show()


if __name__ == "__main__":
    config = load_yaml("dvis_cfg.yaml")
    parser = argparse.ArgumentParser(description="Visualize dataset.")
    parser.add_argument(
        "-i",
        type=str,
        default="default",
        help="Optional override for the input folder path",
    )
    args = parser.parse_args()
    in_dir = Path(config["paths"]["default_pt"]).joinpath(args.i)

    try:
        meta = load_yaml(in_dir.joinpath(config["paths"]["meta_fn"]))
        data = tr.load(in_dir.joinpath(config["paths"]["dset_fn"]))
        extra = tr.load(in_dir.joinpath(config["paths"]["extra_fn"]))
    except Exception as ex:
        print(f"File load unsuccessful: {ex}")
        exit()

    visualizer = DatasetVisualizer(meta, data, extra)
    visualizer.show()