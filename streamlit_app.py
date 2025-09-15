# streamlit_app.py
from __future__ import annotations
import base64
import streamlit as st
import numpy as np

from parameters import default_params, drivers_from_params
from plotting_fast import plot_panel  # row labels handled inside plotting code

st.set_page_config(page_title="Coral DEB model", layout="wide")

st.title("Coral DEB model")
st.caption("Dynamic Energy Budget model for coral symbiosis")

# ---------------------------
# Helpers
# ---------------------------
def reset_params() -> dict:
    return default_params().copy()

@st.cache_data(show_spinner=False)
def build_drivers_cached(params: dict):
    return drivers_from_params(params)

def slider_auto(label, value, lo=None, hi=None, step=None):
    """
    Streamlit slider with optional auto-range.
    Pass lo/hi to override; otherwise a sensible range is inferred.
    """
    v = float(value)

    # Infer range if not provided
    if lo is None or hi is None:
        if v == 0:
            lo, hi = 0.0, 1.0
        else:
            span = abs(v) if abs(v) > 1e-12 else 1.0
            lo = 0.0 if v >= 0 else -2.0 * span
            hi = 2.5 * span
            # Special-case tiny positives
            if 0 <= v < 1e-9:
                lo, hi = 0.0, max(5e-9, 5 * v if v > 0 else 1e-9)

    # Infer step if not provided
    if step is None:
        step = (hi - lo) / 100.0 if hi > lo else 1.0

    return st.slider(
        label,
        min_value=float(lo),
        max_value=float(hi),
        value=float(v),
        step=float(step),
    )

# ---------------------------
# Params & Sidebar UI
# ---------------------------
if "params" not in st.session_state:
    st.session_state["params"] = reset_params()

params = st.session_state["params"]

with st.sidebar:
    # --- Environment first ---
    st.header("Environment")

    st.subheader("Temperature T")
    params["T_mean"]  = slider_auto("T mean (°C)", params.get("T_mean", 28.0), lo=26, hi=35, step=0.1)
    params["T_amp"]   = slider_auto("T amplitude (± °C)", params.get("T_amp", 2.0), lo=0, hi=10, step=0.1)
    params["T_phase"] = st.slider(
        "T peak day (0–365)",
        min_value=0.0, max_value=365.0,
        value=float(params.get("T_phase", 200.0))
    )

    st.subheader("Light L")
    params["L_mean"]  = slider_auto("L mean (mol photons m⁻² d⁻¹)", params.get("L_mean", 30.0), lo=0, hi=80, step=1.0)
    params["L_amp"]   = slider_auto("L amplitude (± around mean)",  params.get("L_amp", 25.0), lo=0, hi=40, step=1.0)
    params["L_phase"] = st.slider(
        "L peak day (0–365)",
        min_value=0.0, max_value=365.0,
        value=float(params.get("L_phase", 172.0))
    )

    st.subheader("Other environment factors")
    # DIN abundance (mean only)
    params["Nu_mean"] = slider_auto(
        "DIN abundance (mol N L⁻¹)",
        params.get("Nu_mean", 2e-7),
        lo=0.0, hi=8e-7, step=1e-8
    )
    params["Nu_amp"] = 0.0
    params["Nu_phase"] = 0.0

    # Prey abundance (mean only)
    params["X_mean"] = slider_auto(
        "Prey abundance (mol C L⁻¹)",
        params.get("X_mean", 2e-7),
        lo=0.0, hi=8e-7, step=1e-8
    )
    params["X_amp"] = 0.0
    params["X_phase"] = 0.0

    st.divider()

    # --- Initial Conditions (only S and H shown) ---
    st.header("Initial Conditions")
    s0 = slider_auto("S(0) (symbiont)", params["y0"][0])
    h0 = slider_auto("H(0) (host)",     params["y0"][1])
    # Hidden (kept at current/default values)
    jcp0 = params["y0"][2]
    jsg0 = params["y0"][3]
    params["y0"] = [s0, h0, jcp0, jsg0]

    st.divider()

    # --- tmax last (default 365, max 730) ---
    params["tmax"] = st.slider(
        "Simulation length (days)",
        min_value=14, max_value=730,
        value=int(params.get("tmax", 365)),
        step=5
    )

    # Reset button
    if st.button("Reset"):
        st.session_state["params"] = reset_params()
        st.experimental_rerun()

# ---------------------------
# Drivers, Simulation, Plot
# ---------------------------
drivers = build_drivers_cached(params)

tabs = st.tabs(["📈 Plots", "ℹ️ About"])

with tabs[0]:
    with st.spinner("Running simulation and rendering figure..."):
        try:
            fig, axes = plot_panel(params, drivers=drivers, show=False)
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.error(f"Plotting failed: {e}")

with tabs[1]:

    # Show the static diagram as an image (cleaner than embedding a PDF viewer)
    st.image("coral-heat-diagram-Fv-Fm-Arr.svg", use_container_width=True)

    st.markdown(
        """
**Model details**

Heat stress and bleaching in corals: a bioenergetic model.

Pfab, Ferdinand, A. Raine Detmer, Holly V. Moeller, Roger M. Nisbet, Hollie M. Putnam, and Ross Cunning.

Coral Reefs (2024).  
<https://link.springer.com/article/10.1007/s00338-024-02561-1>
        """
    )
