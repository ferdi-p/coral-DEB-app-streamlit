# plotting.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from model import simulate, compute_outputs_along_solution
from helper_functions import Drivers

# Names & units (add environment drivers)
NAMES_UNITS = {
    # Environment
    "L":  dict(name="Light (L)",               unit="mol photons m⁻² d⁻¹"),
    "T":  dict(name="Temperature (T)",         unit="°C"),
    "Nu": dict(name="DIN (Nu)",                unit="mol N L⁻¹"),
    "X":  dict(name="Prey (X)",                unit="mol C L⁻¹"),
    # Outputs (Fig.5 second row and on)
    "jCP": dict(name="Photosynthesis rate (jCP)",           unit="mol C mol C⁻¹ d⁻¹"),
    "rho_C": dict(name="C shared by symbiont (ρC)",              unit="mol C mol C⁻¹ d⁻¹"),
    "rho_N": dict(name="N shared by host (ρN)",                unit="mol N mol C⁻¹ d⁻¹"),
    "jeL": dict(name="Excess light (jeL)",                  unit="mol photons mol C⁻¹ d⁻¹"),
    "jNPQ": dict(name="Non-photochem. quench. (jNPQ)",      unit="mol photons mol C⁻¹ d⁻¹"),
    "cROS_increase": dict(name="ROS increase (cROS − 1)",   unit="–"),
    "expulsion": dict(name="Expulsion flux b(cROS−1)·jST,0",unit="mol C mol C⁻¹ d⁻¹"),
    "jSG": dict(name="Symbiont biomass (jSG)",              unit="mol C mol C⁻¹ d⁻¹"),
    "jHG": dict(name="Host biomass (jHG)",                  unit="mol C mol C⁻¹ d⁻¹"),
    "net_S": dict(name="Net symbiont growth Ṡ/S",           unit="d⁻¹"),
    "net_H": dict(name="Net host growth Ḣ/H",               unit="d⁻¹"),
    "S_over_H": dict(name="Symbiont/host ratio S/H",        unit="–"),
}

# Fig-5 style keys (second row and on)
panel_KEYS = [
    "S_over_H","jCP","rho_C","rho_N",
    "jNPQ","cROS_increase",
    "net_S","net_H"
]

def _sample_driver(fn_or_const, t: np.ndarray) -> np.ndarray:
    if callable(fn_or_const):
        return np.array([fn_or_const(ti) for ti in t], dtype=float)
    return np.full_like(t, float(fn_or_const), dtype=float)

def plot_panel(
    params: Dict[str, float],
    drivers: Optional[Drivers] = None,
    show: bool = True,
) -> Tuple[object, tuple]:
    """
    First row: L, T, Nu, X (annual cycles).
    Remaining rows: model outputs in `panel_KEYS`.
    Time runs 0..params['tmax'].
    """
    if drivers is None:
        raise ValueError("plot_panel requires a Drivers instance.")

    # simulate state & outputs
    sol = simulate(params=params, drivers=drivers)
    series = compute_outputs_along_solution(sol, params, drivers)
    t = series["t"]

    # sample environment
    L_series  = _sample_driver(drivers.L,  t)
    T_series  = _sample_driver(drivers.T,  t)
    Nu_series = _sample_driver(drivers.Nu, t)
    X_series  = _sample_driver(drivers.X,  t)

    # Grid: 4 columns; rows = 1 (env) + enough to place all panels
    ncols = 4
    rows_for_panels = (len(panel_KEYS) + ncols - 1) // ncols
    nrows = 1 + max(1, rows_for_panels)

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.0*nrows), sharex=True)
    axes = np.array(axes).ravel()

    # Row 1: environment
    env_keys = [("L", L_series), ("T", T_series), ("Nu", Nu_series), ("X", X_series)]
    for i, (k, arr) in enumerate(env_keys):
        ax = axes[i]
        ax.plot(t, arr)
        ax.set_title(NAMES_UNITS[k]["name"])
        ax.set_ylabel(NAMES_UNITS[k]["unit"])
        ax.grid(True, alpha=0.3)

    # Remaining: model outputs
    start = ncols  # after the environment row
    for i, key in enumerate(panel_KEYS):
        ax = axes[start + i]
        ax.plot(t, series[key])
        meta = NAMES_UNITS[key]
        ax.set_title(meta["name"])
        ax.set_ylabel(meta["unit"])
        ax.grid(True, alpha=0.3)

    # Shared x label on bottom row, hide any unused axes
    last_row = (nrows - 1) * ncols
    for j in range(last_row, nrows*ncols):
        if j >= start + len(panel_KEYS):
            fig.delaxes(axes[j])

    for ax in axes[-ncols:]:
        if ax.has_data():
            ax.set_xlabel("Time (days)")

    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes
