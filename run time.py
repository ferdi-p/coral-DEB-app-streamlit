from parameters import default_params, drivers_from_params
from plotting import plot_panel
from model_fast import simulate, compute_outputs_along_solution

import time

params = default_params()
params["tmax"] = 365.0

drivers = drivers_from_params(params)
start = time.time()
simulate(params, drivers=drivers)
end = time.time()

print(f"Runtime: {end - start:.3f} seconds")