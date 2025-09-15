# streamlit_app.py
from __future__ import annotations
import io
import json
import numpy as np
import streamlit as st

from parameters import default_params, drivers_from_params
from plotting import plot_panel
from model import simulate, compute_outputs_along_solution

st.set_page_config(page_title="Coral Heat Stress — Annual Drivers", layout="wide")

st.title("Coral Heat Stress Model")
st.caption("Annual cycles for L, T, DIN (Nu), and Prey (X); Fig.5-style outputs below.")

# --- Load & tweak params from UI ---
params = default_params().copy()

with st.sidebar:
    st.header("Simulation")
    params["tmax"] = st.slider("Simulation length (days)", 30, 1460, int(params.get("tmax", 365.0)), step=5)
    params["steps_per_day"] = st.slider("Steps per day", 10, 200, int(params.get("steps_per_day", 40)), step=5)

    st.divider()
    st.header("Initial Conditions (y0)")
    s0 = st.number_input("S0 (symbiont)", value=float(params["y0"][0]), format="%.6g")
    h0 = st.number_input("H0 (host)", value=float(params["y0"][1]), format="%.6g")
    jcp0 = st.number_input("jCP0", value=float(params["y0"][2]), format="%.6g")
    jsg0 = st.number_input("jSG0", value=float(params["y0"][3]), format="%.6g")
    params["y0"] = [s0, h0, jcp0, jsg0]

    st.divider()
    st.header("Annual Cycles (365 d period)")

    st.subheader("Light L")
    params["L_mean"]  = st.number_input("L mean (mol photons m⁻² d⁻¹)", value=float(params.get("L_mean", 30.0)))
    params["L_amp"]   = st.number_input("L amplitude (± around mean)", value=float(params.get("L_amp", 25.0)))
    params["L_phase"] = st.number_input("L peak day (0–365)", value=float(params.get("L_phase", 172.0)))

    st.subheader("Temperature T")
    params["T_mean"]  = st.number_input("T mean (°C)", value=float(params.get("T_mean", 28.0)))
    params["T_amp"]   = st.number_input("T amplitude (± °C)", value=float(params.get("T_amp", 2.0)))
    params["T_phase"] = st.number_input("T peak day (0–365)", value=float(params.get("T_phase", 200.0)))

    st.subheader("DIN Nu")
    params["Nu_mean"]  = st.number_input("Nu mean (mol N L⁻¹)", value=float(params.get("Nu_mean", 2e-7)), format="%.6g")
    params["Nu_amp"]   = st.number_input("Nu amplitude (±)", value=float(params.get("Nu_amp", 1e-7)), format="%.6g")
    params["Nu_phase"] = st.number_input("Nu peak day (0–365)", value=float(params.get("Nu_phase", 30.0)))

    st.subheader("Prey X")
    params["X_mean"]  = st.number_input("X mean (mol C L⁻¹)", value=float(params.get("X_mean", 2e-7)), format="%.6g")
    params["X_amp"]   = st.number_input("X amplitude (±)", value=float(params.get("X_amp", 5e-8)), format="%.6g")
    params["X_phase"] = st.number_input("X peak day (0–365)", value=float(params.get("X_phase", 60.0)), format="%.6g")

    st.divider()
    st.caption("Tip: to stress temperature more, try raising T_mean or T_amp; "
               "ensure your model recomputes α and β from T(t) each step.")

# Build drivers from current params
drivers = drivers_from_params(params)

# --- Run simulation & make figure ---
fig, axes = plot_panel(params, drivers=drivers, show=False)
st.pyplot(fig, clear_figure=True)

# --- Optional: downloadable CSV of key series ---
with st.expander("Download time series as CSV"):
    sol = simulate(params=params, drivers=drivers)
    series = compute_outputs_along_solution(sol, params, drivers)

    # sample environment at the solver times:
    t = series["t"]
    L = np.array([drivers.L(ti) if callable(drivers.L) else drivers.L for ti in t])
    T = np.array([drivers.T(ti) if callable(drivers.T) else drivers.T for ti in t])
    Nu = np.array([drivers.Nu(ti) if callable(drivers.Nu) else drivers.Nu for ti in t])
    X  = np.array([drivers.X(ti)  if callable(drivers.X)  else drivers.X  for ti in t])

    # choose columns to export
    export_keys = ["jCP", "rho_C", "rho_N", "jeL", "jNPQ", "cROS_increase",
                   "expulsion", "jSG", "jHG", "net_S", "net_H", "S_over_H"]
    header = "t,L,T,Nu,X," + ",".join(export_keys)
    rows = np.column_stack([t, L, T, Nu, X] + [series[k] for k in export_keys])

    # to CSV
    buf = io.StringIO()
    np.savetxt(buf, rows, delimiter=",", header=header, comments="")
    st.download_button("Download CSV", buf.getvalue(), file_name="coral_timeseries.csv", mime="text/csv")
