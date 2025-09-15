# helper_functions.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict
import math

# ---- Helper “biochemistry” functions (from Mathematica defs) ----
def f(max_, A):
    """f[max][A] := (A max)/(A + max)"""
    return (A * max_) / (A + max_)

def F(max_, A, B):
    """F[max][A,B] := (A B (A+B) max)/(B^2 max + A^2 (B+max) + A B (B+max))"""
    denom = (B * B * max_) + (A * A * (B + max_)) + (A * B * (B + max_))
    return (A * B * (A + B) * max_) / denom if denom != 0 else 0.0

# # ---- Drivers (constants or callables) ----
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


@dataclass
class AnnualCycle:
    mean: float
    amplitude: float
    phase_day: float   # day of year when maximum occurs

    def __call__(self, t: float) -> float:
        return self.mean + self.amplitude * math.cos(
            2 * math.pi * (t - self.phase_day) / 365.0
        )

@dataclass
class Drivers:
    L: Callable[[float], float] | float = 0.0
    T: Callable[[float], float] | float = 28.0
    X: Callable[[float], float] | float = 1.0
    Nu: Callable[[float], float] | float = 1.0

    def at(self, t: float) -> Dict[str, float]:
        def val(x): return x(t) if callable(x) else float(x)
        return dict(L=val(self.L), T=val(self.T), X=val(self.X), Nu=val(self.Nu))
