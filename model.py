# model.py
from __future__ import annotations
from typing import Dict, Optional
import math
import numpy as np
from scipy.integrate import solve_ivp

from helper_functions import f, F, Drivers

# ---- Rule set adapted from Mathematica 'rules' ----
RULES_PY: Dict[str, str] = {
    # Base uptake & pools
    "jX":        "(jXm * X) / (X + KX)",
    "jN":        "(jNm * Nu) / (Nu + KN)",
    "jNH":       "jN + nNX * jX + rNH",
    "jHT":       "j0HT",
    "rNS":       "sigmaNS * nNS * j0ST",

    # F is the two-substrate synthesizing-unit
    "jHG":       "F(jHGm, yC * jHC, jNH / nNH)",

    # Carbon handling
    "jHC":       "jX + (S/H) * rhoC",
    "jeC":       "jHC - jHG / yC",
    "jCO2":      "jeC * kCO2",

    # Light handling
    "jL":        "A * astar * L",
    "rCH":       "(jHT + (jHG * (1 - yC)) / yC) * sigmaCH",
    "jeL":       "jL - jCP / yCL",
    "jNPQ":      "1.0 / (1.0/jeL + 1.0/kNPQ)",

    # Temperature stress on ST/HT
    "jST":       "(1 + b * cROS1) * j0ST",

    # Nitrogen handling
    "rNH":       "sigmaNH * nNH * jHT",

    # Light absorption cross-section A depends on S/H
    "A":         "1.256307 + 1.385969 * math.exp(-6.479055 * S / H)",

    # Carbon storage respiration
    "rCS":       "sigmaCS * (j0ST + (1 - yC) * jSG / yC)",

    # ROS
    "cROS1":     "max(0.0, jeL - jNPQ) * OneOverkROS",

    # Partitioning rates (rho)
    "rhoC":      "jCP - jSG / yC",
    "rhoN":      "jNH - jHG * nNH",

    # Aims for SG and CP (do not overwrite states)
    "jSGaim":    "F(jSGm, jCP * yC, (rNS + (H/S) * rhoN) / nNS)",
    "jCPaim":    "F(jCPm, jL * yCL, rCS + (jCO2 + rCH) * H / S) / (1 + cROS1)",

    # Temperature scalings
    "jHGm":      "alpha * jHGm0",
    "jNm":       "alpha * jNm0",
    "jSGm":      "alpha * jSGm0",
    "j0HT":      "alpha * j0HT0",
    "j0ST":      "alpha * j0ST0",
    "jXm":       "alpha * jXm0",
    "jCPm":      "alpha * beta * jCPm0",

    # alpha, beta
    "alpha":     "Q10 ** ((T - T0) / 10.0)",

    # Sigmoid(T) normalized to T0
    "sigmoidFunction": "sigmax * math.exp(-steepness * (T - ED50)) / (1.0 + math.exp(-steepness * (T - ED50)))",
    "beta":      "sigmoidFunction / (sigmax * math.exp(-steepness * (T0 - ED50)) / (1.0 + math.exp(-steepness * (T0 - ED50))))",
}

SAFE_GLOBALS = {
    "math": math,
    "np": np,
    "f": f,
    "F": F,
    "max": max,
}

def evaluate_rules(env: Dict[str, float], max_passes: int = 8) -> Dict[str, float]:
    """
    Multi-pass evaluation of RULES_PY.
    env starts with state, params, and drivers at time t.
    Returns env extended with computable intermediates (incl. jCPaim, jSGaim).
    """
    out = dict(env)
    for _ in range(max_passes):
        progressed = False
        for name, expr in RULES_PY.items():
            if name in out:
                continue
            try:
                val = eval(expr, SAFE_GLOBALS, out)
            except NameError:
                continue  # missing prerequisites; try later
            except ZeroDivisionError:
                val = float("inf")
            out[name] = float(val)
            progressed = True
        if not progressed:
            break
    return out

def rhs(t: float, y: np.ndarray, p: Dict[str, float], drivers: Drivers) -> np.ndarray:
    """
    y = [S, H, jCP, jSG]
    S'   = S * (jSG - jST)
    H'   = H * (jHG - jHT)
    jCP' = lam * (jCPaim - jCP)
    jSG' = lam * (jSGaim - jSG)
    """
    S, H, jCP, jSG = y
    env = dict(p)
    env.update(dict(S=S, H=H, jCP=jCP, jSG=jSG))
    env.update(drivers.at(t))  # L, T, X, Nu

    env = evaluate_rules(env)

    jST    = env.get("jST", 0.0)
    jHT    = env.get("jHT", 0.0)
    jHG    = env.get("jHG", 0.0)
    jCPaim = env.get("jCPaim", jCP)
    jSGaim = env.get("jSGaim", jSG)
    lam    = env.get("lam", p.get("lam", 1.0))

    dS   = S * (jSG - jST)
    dH   = H * (jHG - jHT)
    djCP = lam * (jCPaim - jCP)
    djSG = lam * (jSGaim - jSG)
    return np.array([dS, dH, djCP, djSG], dtype=float)

def simulate(
    params: Dict[str, float],
    drivers: Drivers,
    y0: Optional[np.ndarray | list[float]] = None,
    t_eval: Optional[np.ndarray] = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
):
    """
    Integrate the 4D ODE with automatic rule expansion.
    Uses params['tmax'] for the final time and params['y0'] if y0 is None.
    Returns scipy.integrate.OdeResult
    """
    if y0 is None:
        y0 = params.get("y0", [1.0, 0.1, 0.2, 0.2])
    y0 = np.array(y0, dtype=float)

    tmax = float(params.get("tmax", 30.0))
    if t_eval is None:
        steps_per_day = int(params.get("steps_per_day", 40))
        t_eval = np.linspace(0.0, tmax, int(tmax * steps_per_day) + 1)

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, params, drivers),
        t_span=(0.0, tmax),
        y0=y0,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        method="RK45",
        vectorized=False,
    )
    return sol

def compute_outputs_along_solution(sol, params: Dict[str, float], drivers: Drivers) -> Dict[str, np.ndarray]:
    """
    For each time in sol.t and state in sol.y, recompute intermediates via evaluate_rules
    and return time series for selected outputs (Fig.5, second row and on).
    Returns a dict of arrays keyed by variable names.
    """
    t = sol.t
    S, H, jCP, jSG = sol.y
    out = {
        "t": t,
        "jCP": np.zeros_like(t),
        "rho_C": np.zeros_like(t),      # fixed carbon shared with host (rhoC)
        "rho_N": np.zeros_like(t),      # nitrogen shared with symbiont (rhoN)
        "jeL": np.zeros_like(t),
        "jNPQ": np.zeros_like(t),
        "cROS_increase": np.zeros_like(t),   # cROS - 1 (our cROS1 already equals this increment)
        "expulsion": np.zeros_like(t),       # b*(cROS-1)*jST0
        "jSG": np.zeros_like(t),
        "jHG": np.zeros_like(t),
        "net_S": np.zeros_like(t),      # Sdot/S = jSG - jST
        "net_H": np.zeros_like(t),      # Hdot/H = jHG - jHT
        "S_over_H": np.zeros_like(t),
        "jCO2": np.zeros_like(t),       # <-- added
    }
    j0ST0 = params.get("j0ST0", params.get("j0ST", 0.03))  # fallback
    for i, ti in enumerate(t):
        env = dict(params)
        env.update(dict(S=S[i], H=H[i], jCP=jCP[i], jSG=jSG[i]))
        env.update(drivers.at(ti))
        env = evaluate_rules(env)

        out["jCP"][i] = env.get("jCP", jCP[i])
        out["rho_C"][i] = env.get("rhoC", np.nan)
        out["rho_N"][i] = env.get("rhoN", np.nan)
        out["jeL"][i] = env.get("jeL", np.nan)
        out["jNPQ"][i] = env.get("jNPQ", np.nan)
        cros1 = env.get("cROS1", 0.0)
        out["cROS_increase"][i] = cros1
        out["expulsion"][i] = params.get("b", 5.0) * cros1 * j0ST0
        out["jSG"][i] = env.get("jSG", jSG[i])
        out["jHG"][i] = env.get("jHG", np.nan)
        jST = env.get("jST", np.nan)
        jHT = env.get("jHT", np.nan)
        out["net_S"][i] = out["jSG"][i] - jST
        out["net_H"][i] = out["jHG"][i] - jHT
        out["S_over_H"][i] = S[i] / H[i] if H[i] != 0 else np.nan
        out["jCO2"][i] = env.get("jCO2", np.nan)  # <-- added
    return out
