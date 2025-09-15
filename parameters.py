# parameters.py
from __future__ import annotations
from typing import Dict
from dataclasses import dataclass
from helper_functions import Drivers
import math


def default_params() -> Dict[str, float]:
    """
    Baseline parameter set (paper-style placeholders).
    Replace with your exact table when ready.
    """
    return dict(
        # --- model parameters ---
        yC=0.8,
        yCL=0.1,
        jHGm0=1.0,
        jSGm0=0.25,
        jCPm0=2.8,
        jXm0=0.13,
        jNm0=0.035,
        j0HT0=0.03,
        j0ST0=0.03,
        KX=1e-6,
        KN=1.5e-6,
        nNH=0.18,
        nNS=0.13,
        nNX=0.2,
        astar=1.34,
        kCO2=10.0,
        kNPQ=112.0,
        sigmaCH=0.1,
        sigmaCS=0.9,
        sigmaNH=0.9,
        sigmaNS=0.9,
        OneOverkROS=1.0/80.0,
        b=6.0,
        Q10=1.67,
        T0=28.0,
        sigmax=1.0,
        steepness=0.578183,
        ED50=29.6526,
        lam=1,
        # alpha=1.0,
        # beta=1.0,

        # environment cycle parameters
        L_mean=30.0, L_amp=0.0, L_phase=172.0,  # peak at midsummer (~day 172 = June 21)
        T_mean=27.0, T_amp=7.0, T_phase=182.0,  # peak later in the summer
        Nu_mean=2e-7, Nu_amp=0, Nu_phase=30.0,  # DIN
        X_mean=2e-7, X_amp=0, X_phase=60.0,  # Prey
        # --- simulation settings ---
        tmax=365.0,          # days


        steps_per_day=10,   # resolution
        y0=[0.1,1, 2.8*0.2, 0.25*0.1],  # [S, H, jCP, jSG] initial values

    )


@dataclass
class AnnualCycle:
    mean: float
    amplitude: float
    phase_day: float   # day of year when maximum occurs

    def __call__(self, t: float) -> float:
        return self.mean + self.amplitude * math.cos(
            2 * math.pi * (t - self.phase_day) / 365.0
        )

def drivers_from_params(params):
    L  = AnnualCycle(params["L_mean"],  params["L_amp"],  params["L_phase"])
    T  = AnnualCycle(params["T_mean"],  params["T_amp"],  params["T_phase"])
    Nu = AnnualCycle(params["Nu_mean"], params["Nu_amp"], params["Nu_phase"])
    X  = AnnualCycle(params["X_mean"],  params["X_amp"],  params["X_phase"])
    return Drivers(L=L, T=T, Nu=Nu, X=X)

def default_drivers() -> Drivers:
    """Constant light & temperature (and Nu, X) for now."""
    return Drivers(L=30.0, T=28.0, Nu=2e-7, X=2e-7)


