# model_fast.py
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

EPS  = 0#1e-12
EPS2 = 0#1e-24

# ---------- Helpers ----------
def _as_func_on_time(arr_or_fn, t_grid: np.ndarray):
    if callable(arr_or_fn):
        return arr_or_fn
    arr = np.asarray(arr_or_fn, dtype=float)
    if arr.size == t_grid.size:
        return lambda tt: float(np.interp(tt, t_grid, arr))
    val = float(arr)
    return lambda tt: val

def _get_driver(drivers, name: str, t_grid: np.ndarray):
    src = drivers[name] if isinstance(drivers, dict) else getattr(drivers, name)
    return _as_func_on_time(src, t_grid)

# ---------- Vector field ----------
def rhs_fast(t: float, y: np.ndarray, params: dict, env: dict[str, callable]) -> np.ndarray:
    """
    States: y = [S, H, jCP, jSG]
    env: dict with callables L(t), T(t), Nu(t), X(t)
    """
    S, H, jCP, jSG = y

    # environment at time t
    L  = float(env["L"](t))
    T  = float(env["T"](t))
    Nu = float(env["Nu"](t))
    X  = float(env["X"](t))

    p = params

    # temperature scalings (original form)
    alpha = p["Q10"] ** ((T - p["T0"]) / 10.0)
    eT    = np.exp(-p["steepness"] * (T - p["ED50"]))
    eT0   = np.exp(-p["steepness"] * (p["T0"] - p["ED50"]))
    sigmoid_T  = p["sigmax"] * eT  / (1.0 + eT)
    sigmoid_T0 = p["sigmax"] * eT0 / (1.0 + eT0)
    Beta  = sigmoid_T / (sigmoid_T0 + EPS)

    # T-scaled maxima
    j0ST = alpha * p["j0ST0"]
    j0HT = alpha * p["j0HT0"]
    jXm  = alpha * p["jXm0"]
    jHGm = alpha * p["jHGm0"]
    jNm  = alpha * p["jNm0"]
    jSGm = alpha * p["jSGm0"]
    jCPm = alpha * Beta * p["jCPm0"]

    # resource uptakes
    Xterm  = (jXm * X)  / (p["KX"] + X  + EPS)
    NHterm = (jNm * Nu) / (p["KN"] + Nu + EPS)

    # host turnover
    jHT = j0HT

    # nitrogen totals
    rNH = p["sigmaNH"] * p["nNH"] * jHT
    jNH = NHterm + p["nNX"] * Xterm + rNH

    # carbon side
    rhoC = jCP - jSG / p["yC"]
    jHC  = Xterm + (rhoC * S) / (H + EPS)

    # geometry / light capture / photochemistry (original)
    Ageom = 1.25631 + 1.38597 * np.exp(-6.47906 * S / (H + EPS))
    jL    = p["astar"] * Ageom * L
    jeL   = jL - jCP / p["yCL"]
    jNPQ  = 1.0 / (1.0 / (jeL + EPS) + 1.0 / (p["kNPQ"] + EPS))
    cROS1 = np.maximum(0.0, jeL - jNPQ) * p["OneOverkROS"]

    # symbiont turnover (with ROS multiplier)
    jST = j0ST * (1.0 + p["b"] * cROS1)

    # jHG via F(max)
    C_big  = (p["nNX"] * Xterm + NHterm + p["nNH"] * j0HT)
    HC_big = Xterm + (rhoC * S) / (H + EPS)
    B_HG   = C_big / (p["nNH"] + EPS)
    A_HG   = p["yC"] * HC_big

    denom_F_HG = (B_HG**2) * jHGm + (A_HG**2) * (B_HG + jHGm) + (A_HG * B_HG) * (B_HG + jHGm)
    jHG = (A_HG * B_HG * (A_HG + B_HG) * jHGm) / (denom_F_HG + EPS2)

    # balances
    rhoN = jNH - jHG * p["nNH"]

    # jSG*
    rNS = p["sigmaNS"] * p["nNS"] * j0ST
    A_SG = p["yC"] * jCP
    B_SG = (rNS + (H / (S + EPS)) * rhoN) / (p["nNS"] + EPS)
    denom_F_SG = (B_SG**2) * jSGm + (A_SG**2) * (B_SG + jSGm) + (A_SG * B_SG) * (B_SG + jSGm)
    jSG_star = (A_SG * B_SG * (A_SG + B_SG) * jSGm) / (denom_F_SG + EPS2)
    djSG = p["lam"] * (jSG_star - jSG)

    # jCP* with CO2 and ROS
    jeC  = jHC - jHG / p["yC"]
    jCO2 = p["kCO2"] * jeC
    rCS  = p["sigmaCS"] * (jST + (1.0 - p["yC"]) * jSG / p["yC"])
    rCH  = p["sigmaCH"] * (j0HT + (jHG * (1.0 - p["yC"])) / p["yC"])
    A_CP = jL * p["yCL"]
    B_CP = rCS + (jCO2 + rCH) * (H / (S + EPS))
    denom_F_CP = (B_CP**2) * jCPm + (A_CP**2) * (B_CP + jCPm) + (A_CP * B_CP) * (B_CP + jCPm)
    jCP_star = (A_CP * B_CP * (A_CP + B_CP) * jCPm) / (denom_F_CP + EPS2)
    jCP_star = jCP_star / (1.0 + cROS1)
    djCP = p["lam"] * (jCP_star - jCP)

    # state derivatives
    dS = S * (jSG - jST)
    dH = H * (
        -j0HT
        + (
            jHGm * p["yC"] * C_big * HC_big * (B_HG + p["yC"] * HC_big)
            / (
                p["nNH"]
                * (
                    (jHGm * (C_big ** 2)) / (p["nNH"] ** 2 + EPS)
                    + (C_big / (p["nNH"] * p["yC"] + EPS)) * (jHGm + B_HG) * HC_big
                    + (p["yC"] ** 2) * (jHGm + B_HG) * (HC_big ** 2)
                )
                + EPS2
            )
        )
    )

    return np.array([dS, dH, djCP, djSG], dtype=float)

# ---------- Integrator ----------
def simulate(params: dict, drivers):
    tmax = float(params["tmax"])
    spd  = int(params["steps_per_day"])
    t_eval = np.linspace(0.0, tmax, int(tmax * spd) + 1)

    Lf  = _get_driver(drivers, "L",  t_eval)
    Tf  = _get_driver(drivers, "T",  t_eval)
    Nuf = _get_driver(drivers, "Nu", t_eval)
    Xf  = _get_driver(drivers, "X",  t_eval)

    env = {"L": Lf, "T": Tf, "Nu": Nuf, "X": Xf}

    def f(t, y):
        return rhs_fast(t, y, params, env)

    y0 = np.asarray(params["y0"], dtype=float)

    sol = solve_ivp(
        f, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval,
        method="BDF",
        rtol=float(params.get("rtol", 1e-5)),
        atol=float(params.get("atol", 1e-7)),
        max_step=float(params.get("max_step", np.inf)),
    )
    return sol

# ---------- Outputs ----------
def compute_outputs_along_solution(sol, params: dict, drivers) -> dict[str, np.ndarray]:
    EPS, EPS2 = 1e-12, 1e-24
    t = sol.t
    S, H, jCP, jSG = sol.y

    Lf  = _get_driver(drivers, "L",  t)
    Tf  = _get_driver(drivers, "T",  t)
    Nuf = _get_driver(drivers, "Nu", t)
    Xf  = _get_driver(drivers, "X",  t)

    out = {
        "t": t.copy(),
        "jCP": jCP.copy(),
        "rho_C": np.empty_like(t),
        "rho_N": np.empty_like(t),
        "jeL": np.empty_like(t),
        "jNPQ": np.empty_like(t),
        "cROS_increase": np.empty_like(t),
        "expulsion": np.empty_like(t),
        "jSG": jSG.copy(),
        "jHG": np.empty_like(t),
        "net_S": np.empty_like(t),
        "net_H": np.empty_like(t),
        "S_over_H": np.empty_like(t),
        "jST": np.empty_like(t),
        "jHT": np.empty_like(t),
        "jL":  np.empty_like(t),
        "jCO2": np.empty_like(t),
    }

    p = params
    for i, ti in enumerate(t):
        Si, Hi, jCPi, jSGi = S[i], H[i], jCP[i], jSG[i]
        Li, Ti, Nui, Xi = Lf(ti), Tf(ti), Nuf(ti), Xf(ti)

        alpha = p["Q10"] ** ((Ti - p["T0"]) / 10.0)
        eT    = np.exp(-p["steepness"] * (Ti - p["ED50"]))
        eT0   = np.exp(-p["steepness"] * (p["T0"] - p["ED50"]))
        sigmoid_T  = p["sigmax"] * eT  / (1.0 + eT)
        sigmoid_T0 = p["sigmax"] * eT0 / (1.0 + eT0)
        Beta  = sigmoid_T / (sigmoid_T0 + EPS)

        j0ST = alpha * p["j0ST0"]
        j0HT = alpha * p["j0HT0"]
        jXm  = alpha * p["jXm0"]
        jHGm = alpha * p["jHGm0"]
        jNm  = alpha * p["jNm0"]
        jCPm = alpha * Beta * p["jCPm0"]

        Xterm  = (jXm * Xi) / (p["KX"] + Xi + EPS)
        NHterm = (jNm * Nui) / (p["KN"] + Nui + EPS)

        jHT = j0HT
        rNH = p["sigmaNH"] * p["nNH"] * jHT
        jNH = NHterm + p["nNX"] * Xterm + rNH

        rhoC = jCPi - jSGi / p["yC"]
        jHC  = Xterm + (rhoC * Si) / (Hi + EPS)

        Ageom = 1.25631 + 1.38597 * np.exp(-6.47906 * Si / (Hi + EPS))
        jL    = p["astar"] * Ageom * Li
        jeL   = jL - jCPi / p["yCL"]
        jNPQ  = 1.0 / (1.0 / (jeL + EPS) + 1.0 / (p["kNPQ"] + EPS))
        cROS1 = np.maximum(0.0, jeL - jNPQ) * p["OneOverkROS"]

        jST = j0ST * (1.0 + p["b"] * cROS1)

        A_HG = p["yC"] * jHC
        B_HG = jNH / (p["nNH"] + EPS)
        denom_F_HG = (B_HG**2) * jHGm + (A_HG**2) * (B_HG + jHGm) + (A_HG * B_HG) * (B_HG + jHGm)
        jHG = (A_HG * B_HG * (A_HG + B_HG) * jHGm) / (denom_F_HG + EPS2)

        rhoN = jNH - jHG * p["nNH"]
        net_S = jSGi - jST
        net_H = jHG  - j0HT

        jeC  = jHC - jHG / p["yC"]
        jCO2 = p["kCO2"] * jeC

        out["rho_C"][i] = rhoC
        out["rho_N"][i] = rhoN
        out["jeL"][i]   = jeL
        out["jNPQ"][i]  = jNPQ
        out["cROS_increase"][i] = cROS1
        out["expulsion"][i] = p.get("b", 5.0) * cROS1 * p.get("j0ST0", 0.03)
        out["jHG"][i]   = jHG
        out["net_S"][i] = net_S
        out["net_H"][i] = net_H
        out["S_over_H"][i] = (Si / Hi) if Hi != 0 else np.nan
        out["jST"][i] = jST
        out["jHT"][i] = j0HT
        out["jL"][i]  = jL
        out["jCO2"][i] = jCO2

    return out
