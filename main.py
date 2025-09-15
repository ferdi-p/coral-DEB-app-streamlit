# main.py
import os
from nicegui import ui
from parameters import default_params, drivers_from_params
from plotting import plot_panel

# import your simulation helpers
params = default_params()
drivers = drivers_from_params(params)

# Run a simple initial figure (optional)
fig, _ = plot_panel(params, drivers=drivers, show=False)
ui.image(fig)  # quick placeholder, or replace with your NiceGUI UI code

# Start NiceGUI on the port Render provides
port = int(os.getenv('PORT', 8000))
ui.run(host='0.0.0.0', port=port, reload=False, title='Coral Heat Stress')

# # main.py
# import os
# from nicegui import ui
# from parameters import default_params, drivers_from_params
# from plotting import plot_panel
#
# # import your simulation helpers
# params = default_params()
# drivers = drivers_from_params(params)
#
# # Run a simple initial figure (optional)
# fig, _ = plot_panel(params, drivers=drivers, show=False)
# ui.image(fig)  # quick placeholder, or replace with your NiceGUI UI code
#
# # Start NiceGUI on the port Render provides
# port = int(os.getenv('PORT', 8000))
# ui.run(host='0.0.0.0', port=port, reload=False, title='Coral Heat Stress')


# # coral_heat_app_single.py
# # Coral heat-stress model (4 ODEs) + automatic rule expansion + quick plot check.
#
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Callable, Dict, Optional
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
#
# # --------------------------
# # Helper “biochemistry” functions (your Mathematica defs)
# # --------------------------
#
# def f(max_, A):
#     """f[max][A] := (A max)/(A + max)"""
#     return (A * max_) / (A + max_)
#
# def F(max_, A, B):
#     """F[max][A,B] := (A B (A+B) max)/(B^2 max + A^2 (B+max) + A B (B+max))"""
#     denom = (B * B * max_) + (A * A * (B + max_)) + (A * B * (B + max_))
#     return (A * B * (A + B) * max_) / denom if denom != 0 else 0.0
#
# # --------------------------
# # Drivers (constants or callables)
# # --------------------------
#
# @dataclass
# class Drivers:
#     # Light, temperature, carbon substrate proxy X, nutrient pool Nu
#     L: Callable[[float], float] | float = 0.0
#     T: Callable[[float], float] | float = 28.0
#     X: Callable[[float], float] | float = 1.0
#     Nu: Callable[[float], float] | float = 1.0
#
#     def at(self, t: float) -> Dict[str, float]:
#         def val(x): return x(t) if callable(x) else float(x)
#         return dict(L=val(self.L), T=val(self.T), X=val(self.X), Nu=val(self.Nu))
#
# # --------------------------
# # Automatic rule expansion
# # --------------------------
#
# RULES_PY = {
#     # Base uptake & pools
#     "jX":        "(jXm * X) / (X + KX)",
#     "jN":        "(jNm * Nu) / (Nu + KN)",
#     "jNH":       "(jN + nNX * jX + rNH)",
#     "jHT":       "j0HT",
#     "rNS":       "sigmaNS * nNS * j0ST",
#
#     # F is the two-substrate synthesizing-unit (your F[max][A,B])
#     "jHG":       "F(jHGm, yC * jHC, jNH / nNH)",
#
#     # Carbon handling
#     "jHC":       "jX + (S/H) * rhoC",
#     "jeC":       "jHC - jHG / yC",
#     "jCO2":      "jeC * kCO2",
#
#     # Light handling
#     "jL":        "A * astar * L",
#     "rCH":       "(jHT + (jHG * (1 - yC)) / yC) * sigmaCH",
#     "jeL":       "jL - jCP / yCL",
#     "jNPQ":      "1.0 / (1.0/jeL + 1.0/kNPQ)",
#
#     # Temperature stress on ST/HT
#     "jST":       "(1 + b * cROS1) * j0ST",
#
#     # Nitrogen handling
#     "rNH":       "sigmaNH * nNH * jHT",
#
#     # Light absorption cross-section A depends on S/H
#     "A":         "1.256307 + 1.385969 * math.exp(-6.479055 * S / H)",
#
#     # Carbon storage respiration
#     "rCS":       "sigmaCS * (j0ST + (1 - yC) * jSG / yC)",
#
#     # ROS
#     "cROS1":     "max(0.0, jeL - jNPQ) * OneOverkROS",
#
#     # Partitioning rates (rho)
#     "rhoC":      "jCP - jSG / yC",
#     "rhoN":      "jNH - jHG * nNH",
#
#     # Aims for SG and CP (do not overwrite states)
#     "jSGaim":    "F(jSGm, jCP * yC, (rNS + (H/S) * rhoN) / nNS)",
#     "jCPaim":    "F(jCPm, jL * yCL, rCS + (jCO2 + rCH) * H / S) / (1 + cROS1)",
#
#     # Temperature scalings
#     "jHGm":      "alpha * jHGm0",
#     "jNm":       "alpha * jNm0",
#     "jSGm":      "alpha * jSGm0",
#     "j0HT":      "alpha * j0HT0",
#     "j0ST":      "alpha * j0ST0",
#     "jXm":       "alpha * jXm0",
#     "jCPm":      "alpha * beta * jCPm0",
#
#     # alpha, beta
#     "alpha":     "Q10 ** ((T - T0) / 10.0)",
#
#     # Sigmoid(T) normalized to T0
#     "sigmoidFunction": "sigmax * math.exp(-steepness * (T - ED50)) / (1.0 + math.exp(-steepness * (T - ED50)))",
#     "beta":      "sigmoidFunction / (sigmax * math.exp(-steepness * (T0 - ED50)) / (1.0 + math.exp(-steepness * (T0 - ED50))))",
# }
#
# SAFE_GLOBALS = {
#     "math": math,
#     "np": np,
#     "f": f,
#     "F": F,
#     "max": max,
# }
#
# def evaluate_rules(env: Dict[str, float], max_passes: int = 8) -> Dict[str, float]:
#     """
#     Multi-pass evaluation of RULES_PY.
#     env starts with state, params, and drivers at time t.
#     Returns env extended with computable intermediates (incl. jCPaim, jSGaim).
#     """
#     out = dict(env)
#     for _ in range(max_passes):
#         progressed = False
#         for name, expr in RULES_PY.items():
#             if name in out:
#                 continue
#             try:
#                 val = eval(expr, SAFE_GLOBALS, out)
#             except NameError:
#                 continue  # missing prerequisites; try in a later pass
#             except ZeroDivisionError:
#                 val = float("inf")
#             out[name] = float(val)
#             progressed = True
#         if not progressed:
#             break
#     return out
#
# # --------------------------
# # ODE system
# # --------------------------
#
# def rhs(t: float, y: np.ndarray, p: Dict[str, float], drivers: Drivers) -> np.ndarray:
#     """
#     y = [S, H, jCP, jSG]
#     S'   = S * (jSG - jST)
#     H'   = H * (jHG - jHT)
#     jCP' = lam * (jCPaim - jCP)
#     jSG' = lam * (jSGaim - jSG)
#     """
#     S, H, jCP, jSG = y
#     env = dict(p)
#     env.update(dict(S=S, H=H, jCP=jCP, jSG=jSG))
#     env.update(drivers.at(t))  # L, T, X, Nu
#
#     env = evaluate_rules(env)
#
#     jST    = env.get("jST", 0.0)
#     jHT    = env.get("jHT", 0.0)
#     jHG    = env.get("jHG", 0.0)
#     jCPaim = env.get("jCPaim", jCP)
#     jSGaim = env.get("jSGaim", jSG)
#     lam    = env.get("lam", p.get("lam", 1.0))
#
#     dS   = S * (jSG - jST)
#     dH   = H * (jHG - jHT)
#     djCP = lam * (jCPaim - jCP)
#     djSG = lam * (jSGaim - jSG)
#     return np.array([dS, dH, djCP, djSG], dtype=float)
#
# # --------------------------
# # Public simulate() API
# # --------------------------
#
# def simulate(
#     t_span=(0.0, 365),
#     y0: np.ndarray | list[float] = (1.0, 1.0, 0.1, 0.1),
#     params: Optional[Dict[str, float]] = None,
#     drivers: Optional[Drivers] = None,
#     t_eval: Optional[np.ndarray] = None,
#     rtol: float = 1e-7,
#     atol: float = 1e-9,
# ):
#     """
#     Integrate the 4D ODE with automatic rule expansion.
#     Returns scipy.integrate.OdeResult
#     """
#     if drivers is None:
#         drivers = Drivers(L=30.0, T=28.0, X=2e-7, Nu=2e-7)  # constants for now
#     if params is None:
#         params = default_params()
#     if t_eval is None:
#         t_eval = np.linspace(t_span[0], t_span[1], 1001)
#
#     y0 = np.array(y0, dtype=float)
#     sol = solve_ivp(
#         fun=lambda t, y: rhs(t, y, params, drivers),
#         t_span=t_span,
#         y0=y0,
#         t_eval=t_eval,
#         rtol=rtol,
#         atol=atol,
#         method="RK45",
#         vectorized=False,
#     )
#     return sol
#
# # --------------------------
# # Parameter defaults (paper-style; adjust later as needed)
# # --------------------------
#
# def default_params() -> Dict[str, float]:
#     """
#     Baseline parameter set (paper-style). Replace with your calibrated table if needed.
#     """
#     return dict(
#         # Stoichiometry / yields
#         yC=0.8,
#         yCL=0.1,
#
#         # Synthesizing-unit maxima at baseline temperature
#         jHGm0=1.0,
#         jSGm0=0.25,
#         jCPm0=2.8,
#         jXm0=0.13,
#         jNm0=0.035,
#
#         # Baseline turnover
#         j0HT0=0.03,
#         j0ST0=0.03,
#
#         # Half-saturation & constants
#         KX=1e-6,
#         KN=1.5e-6,
#         nNH=0.18,
#         nNS=0.13,
#         nNX=0.2,
#
#         # Optical/geometric
#         astar=1.34,
#         kCO2=10.0,
#         kNPQ=112.0,
#
#         # Repair / respiration coeffs
#         sigmaCH=0.02,
#         sigmaCS=0.02,
#         sigmaNH=0.02,
#         sigmaNS=0.02,
#
#         # ROS / bleaching
#         OneOverkROS=1.0/80.0,
#         b=5.0,
#
#         # Temperature scaling
#         Q10=1.88,
#         T0=28.0,
#         sigmax=1.0,
#         steepness=0.58,
#         ED50=29.7,
#
#         # Aim relaxation
#         lam=0.5,
#
#         # placeholders (computed by rules)
#         alpha=1.0,
#         beta=1.0,
#     )
#
# # --------------------------
# # Quick plot to check
# # --------------------------
#
# def plot_baseline():
#     """
#     Simulate 30 days at constant L, T, Nu, X; plot S, H, jCP, jSG.
#     """
#     p = default_params()
#     drivers = Drivers(L=30.0, T=28.0, Nu=2e-7, X=2e-7)  # constants
#     t_eval = np.linspace(0, 365, 1201)  # 30 days
#
#     # initial states (tune as needed)
#     y0 = [1.0, 0.1, p["jCPm0"] * 0.8, p["jSGm0"] * 0.8]
#
#     sol = simulate(t_span=(0, 365), y0=y0, params=p, drivers=drivers, t_eval=t_eval)
#
#     if not sol.success:
#         print("Solver failed:", sol.message)
#
#     t = sol.t
#     S, H, jCP, jSG = sol.y
#
#     fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
#
#     # Biomass
#     axes[0].plot(t, S, label="Symbiont S")
#     axes[0].plot(t, H, label="Host H")
#     axes[0].set_ylabel("Biomass (arb.)")
#     axes[0].legend(loc="best")
#     axes[0].grid(True, alpha=0.3)
#
#     # Fluxes
#     axes[1].plot(t, jCP, label="jCP")
#     axes[1].plot(t, jSG, label="jSG")
#     axes[1].set_xlabel("Time (days)")
#     axes[1].set_ylabel("Flux")
#     axes[1].legend(loc="best")
#     axes[1].grid(True, alpha=0.3)
#
#     fig.tight_layout()
#     plt.show()
#
# # --------------------------
# # Run quick check
# # --------------------------
#
# if __name__ == "__main__":
#     plot_baseline()
