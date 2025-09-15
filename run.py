from parameters import default_params, drivers_from_params
from plotting import plot_panel

if __name__ == "__main__":
    params = default_params()
    params["tmax"] = 365.0

    drivers = drivers_from_params(params)
    plot_panel(params, drivers=drivers)

