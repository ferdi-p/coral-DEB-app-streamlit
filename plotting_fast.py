# plotting_new.py (or replace your plot_panel in plotting_fast.py)
from __future__ import annotations
from typing import Dict, Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from model_fast import simulate, compute_outputs_along_solution
from helper_functions import Drivers

# --- meta (names + units) ---
NAMES_UNITS = {
    "L":  dict(name=r"Light ($L$)",                  unit="mol photons m⁻² d⁻¹"),
    "T":  dict(name=r"Temperature ($T$)",            unit="°C"),
    "Nu": dict(name=r"DIN ($N$)",                    unit="mol N L⁻¹"),
    "X":  dict(name=r"Prey ($X$)",                   unit="mol C L⁻¹"),

    "jCP": dict(name=r"Photosynthesis rate ($j_{CP}$)",        unit="mol C mol C⁻¹ d⁻¹"),
    "rho_C": dict(name=r"C shared by symbiont ($\rho_C$)",     unit="mol C mol C⁻¹ d⁻¹"),
    "rho_N": dict(name=r"N shared by host ($\rho_N$)",         unit="mol N mol C⁻¹ d⁻¹"),
    "jeL": dict(name=r"Excess light ($j_{eL}$)",               unit="mol photons mol C⁻¹ d⁻¹"),
    "jNPQ": dict(name=r"Non-photochem. quench. ($j_{NPQ}$)",   unit="mol photons mol C⁻¹ d⁻¹"),
    "cROS_increase": dict(name=r"ROS ($cROS - 1$)",            unit="–"),
    "expulsion": dict(name=r"Expulsion flux $b(cROS - 1)\cdot j_{ST,0}$",
                      unit="mol C mol C⁻¹ d⁻¹"),
    "jSG": dict(name=r"Symbiont biomass ($j_{SG}$)",           unit="mol C mol C⁻¹ d⁻¹"),
    "jHG": dict(name=r"Host biomass ($j_{HG}$)",               unit="mol C mol C⁻¹ d⁻¹"),
    "net_S": dict(name=r"Net symbiont growth ($S'/S$)",        unit="d⁻¹"),
    "net_H": dict(name=r"Net host growth ($H'/H$)",            unit="d⁻¹"),
    "S_over_H": dict(name=r"Symbiont/host ratio ($S/H$)",      unit="–"),
    "jCO2": dict(name=r"CO$_2$ flux ($j_{CO_2}$)",             unit="mol C mol C⁻¹ d⁻¹"),

    # NEW: temperature scalings
    "alpha": dict(name=r"Temperature scaling ($\alpha$)",      unit="–"),
    "jCPm":  dict(name=r"Max photosynthesis ($j_{CP,m}$)",     unit="mol C mol C⁻¹ d⁻¹"),
}

# which panels to show in the Simulation block
panel_KEYS = [
    "S_over_H","jCP","rho_C","rho_N",
    "jCO2","cROS_increase",
    "net_S","net_H",
    # NEW: add at the end (or wherever you like)
  #  "alpha","jCPm",
]

def _sample_driver(fn_or_const, t: np.ndarray) -> np.ndarray:
    if callable(fn_or_const):
        return np.array([fn_or_const(ti) for ti in t], dtype=float)
    return np.full_like(t, float(fn_or_const), dtype=float)

def _bbox_union(ax_list: List[plt.Axes]) -> Optional[Bbox]:
    boxes = [ax.get_position() for ax in ax_list if ax and ax.has_data()]
    if not boxes:
        return None
    x0 = min(b.x0 for b in boxes); y0 = min(b.y0 for b in boxes)
    x1 = max(b.x1 for b in boxes); y1 = max(b.y1 for b in boxes)
    return Bbox.from_extents(x0, y0, x1, y1)

def _alpha_and_jCPm(params: Dict[str, float], T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute alpha = Q10^((T - T0)/10) and jCPm = alpha * beta * jCPm0,
    where beta is the normalized sigmoid per your RULES (at T0 normalization).
    """
    Q10 = float(params["Q10"])
    T0  = float(params["T0"])
    steep = float(params["steepness"])
    ED50  = float(params["ED50"])
    sigmax = float(params["sigmax"])
    jCPm0  = float(params["jCPm0"])

    # alpha
    alpha = Q10 ** ((T - T0) / 10.0)

    # beta = sigmoid(T) / sigmoid(T0), with sigmoid(T) = sigmax * exp(-k*(T-ED50)) / (1 + exp(-k*(T-ED50)))
    def _sigmoid(temp):
        z = -steep * (temp - ED50)
        ez = np.exp(z)
        return sigmax * ez / (1.0 + ez)

    beta = _sigmoid(T) / (_sigmoid(T0) + 1e-12)

    # jCPm
    jCPm = alpha * beta * jCPm0
    return alpha, jCPm

def plot_panel(
    params: Dict[str, float],
    drivers: Optional[Drivers] = None,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    GridSpec layout:
      - 4 columns overall
      - Row 0 (Environment): cols 1–2 hold T and L (centered); cols 0 and 3 are empty
      - Rows 1.. (Simulation): fill all 4 columns with panel_KEYS
    Adds centered group labels with ample headroom.
    """
    if drivers is None:
        raise ValueError("plot_panel requires a Drivers instance.")

    # simulate + outputs
    sol = simulate(params=params, drivers=drivers)
    series = compute_outputs_along_solution(sol, params, drivers)
    t = series["t"]

    # environment series (only L and T shown)
    L_series = _sample_driver(drivers.L, t)
    T_series = _sample_driver(drivers.T, t)

    # NEW: temperature scalings from params & T(t)
    alpha_series, jCPm_series = _alpha_and_jCPm(params, T_series)

    # Make these available like other series for plotting
    series = dict(series)  # shallow copy
    series["alpha"] = alpha_series
    series["jCPm"]  = jCPm_series

    # grid bookkeeping
    ncols = 4
    rows_for_panels = (len(panel_KEYS) + ncols - 1) // ncols
    nrows = 1 + max(1, rows_for_panels)

    # build figure and GridSpec with more space between rows
    fig = plt.figure(figsize=(13, 4.0 * nrows))

    # Add an extra spacer row (0.3) between environment and simulations
    height_ratios = [1.0, 1.0] + [1.0] * rows_for_panels
    nrows_total = len(height_ratios)

    gs = fig.add_gridspec(
        nrows=nrows_total, ncols=ncols,
        left=0.06, right=0.985, bottom=0.07, top=0.92,
        hspace=0.6, wspace=0.28,
        height_ratios=height_ratios
    )

    axes = np.empty((nrows, ncols), dtype=object)

    # --- Row 0: Environment centered in cols 1–2 (T then L)
    env_map = {1: ("T", T_series), 2: ("L", L_series)}
    for c in range(ncols):
        if c in env_map:
            ax = fig.add_subplot(gs[0, c])
            k, arr = env_map[c]
            ax.plot(t, arr)
            ax.set_title(NAMES_UNITS[k]["name"])
            ax.set_ylabel(NAMES_UNITS[k]["unit"])
            ax.grid(True, alpha=0.3)
            axes[0, c] = ax
        else:
            axes[0, c] = None  # keep slots consistent

    # --- Rows 1.. : Simulation outputs across all 4 cols
    out_positions = []
    for r in range(1, nrows):
        for c in range(ncols):
            out_positions.append((r, c))

    for i, key in enumerate(panel_KEYS):
        r, c = out_positions[i]
        ax = fig.add_subplot(gs[r, c])
        ax.plot(t, series[key])
        meta = NAMES_UNITS.get(key, {"name": key, "unit": ""})
        ax.set_title(meta["name"])
        ax.set_ylabel(meta["unit"])
        ax.grid(True, alpha=0.3)
        axes[r, c] = ax

    # remove any extra axes beyond the panels
    for j in range(len(panel_KEYS), len(out_positions)):
        r, c = out_positions[j]
        axes[r, c] = None  # leave blank cell

    # shared x label on bottom row (only on axes that exist & have data)
    for c in range(ncols):
        ax = axes[-1, c]
        if ax is not None and ax.has_data():
            ax.set_xlabel("Time (days)")

    # --- Group labels (centered), using actual bounding boxes
    def _bbox_union_local(ax_list: List[plt.Axes]) -> Optional[Bbox]:
        boxes = [ax.get_position() for ax in ax_list if ax and ax.has_data()]
        if not boxes:
            return None
        x0 = min(b.x0 for b in boxes); y0 = min(b.y0 for b in boxes)
        x1 = max(b.x1 for b in boxes); y1 = max(b.y1 for b in boxes)
        return Bbox.from_extents(x0, y0, x1, y1)

    env_axes = [axes[0, c] for c in (1, 2) if axes[0, c] is not None]
    sim_axes = [axes[r, c] for r in range(1, nrows) for c in range(ncols)
                if axes[r, c] is not None and axes[r, c].has_data()]

    env_bb = _bbox_union_local(env_axes)
    sim_bb = _bbox_union_local(sim_axes)

    if env_bb:
        x_center = 0.5 * (env_bb.x0 + env_bb.x1)
        y_text = env_bb.y1 + 0.03
        fig.text(x_center, y_text, "Environment",
                 ha="center", va="bottom", fontsize=14, fontweight="bold")

    if sim_bb:
        x_center = 0.5 * (sim_bb.x0 + sim_bb.x1)
        y_text = sim_bb.y1 + 0.03
        fig.text(x_center, y_text, "Simulation",
                 ha="center", va="bottom", fontsize=14, fontweight="bold")

    if show:
        plt.show()

    return fig, axes
