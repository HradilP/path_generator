A simple working example is in data_gen.py

Config params:
Env:
- x_max, y_max: self explanatory, min is always assumed to be (0, 0)

Path:
- n_points: the number of points for final resampling
- safety margin: how far points will be pushed away from an obstacle during smoothing
- avg_smoothing_window: how many points on each side are used for moving avg

SFF:
- n_seeds: number of generated seeds (start and end are not included in this number)
- trial_lim: how many attempts at expanding a node are made before popping it from the frontier
- max_rad: maximum distance for node expansion
- min_rad: minimum distance for node expansion
- tree_dist: how close trees can grow to eachother before being bridged
- grid_res: size of a grid cell edge used for lookups of points in proximity to a point
- bridge_thresh: how close to each other bridges can be (for a given tree pair, other tree pairs are not included)

Other notes:
- max_rad <= tree_dist <= grid_res should be true to keep everything working correctly
- increasing n_seeds RAPIDLY increases the number of paths
- increasing bridge_thresh reduces the amount of bridges, but if a given tree pair can be connected, at least a single bridge will always exist