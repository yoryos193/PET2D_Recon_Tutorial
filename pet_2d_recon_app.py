import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

#from pet_2d_recon_app import (
#    make_brain_phantom,
#    make_roi_masks,
#    make_projector,
#    simulate_measurements,
#    run_reconstruction,
#    run_map,
#    NOISE_PRESETS
#)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="PET Reconstruction Lab",
    layout="wide"
)

st.title("🧠 PET Image Reconstruction Lab")
st.markdown("Interactive learning tool for MLEM, OSEM, and MAP reconstruction")

# -------------------------------------------------
# CACHE HEAVY SETUP
# -------------------------------------------------
@st.cache_resource
def load_system():
    N = 64
    phantom = make_brain_phantom(N)
    roi_masks = make_roi_masks(N)
    P, _, _ = make_projector(N)
    return N, phantom, roi_masks, P

N, PHANTOM_2D, ROI_MASKS, P_MATRIX = load_system()

# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.radio("Mode", ["Student", "Advanced"])

algorithm = st.sidebar.selectbox("Algorithm", ["MLEM", "OSEM", "MAP"])

noise_level = st.sidebar.selectbox(
    "Noise level", list(NOISE_PRESETS.keys()), index=2
)

scatter = st.sidebar.slider("Scatter fraction", 0.0, 0.5, 0.1)
seed = st.sidebar.number_input("Random seed", value=42)

iterations = st.sidebar.slider("Iterations", 1, 100, 20)

# OSEM
subsets = 1
if algorithm == "OSEM":
    subsets = st.sidebar.slider("Subsets", 1, 16, 4)

# MAP
beta = 0.05
prior = None
if algorithm == "MAP":
    prior = st.sidebar.selectbox(
        "Prior",
        [
            "Quadratic (smooth, blurs edges)",
            "MRP (edge-preserving, local median)",
            "TV  (edge-preserving, piecewise-flat)"
        ]
    )
    beta = st.sidebar.slider("Beta", 0.0, 1.0, 0.05)

compare = st.sidebar.checkbox("Compare with MLEM")

run_button = st.sidebar.button("▶ Run Reconstruction")

# -------------------------------------------------
# HELP SECTION (TEACHING)
# -------------------------------------------------
with st.expander("📘 What is MLEM?"):
    st.write("""
MLEM maximises the Poisson likelihood:

xₖ₊₁ = xₖ · (Aᵀ (y / Ax)) / (Aᵀ 1)

It iteratively improves the image so projections match measured data.
""")

# -------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------
if run_button:

    st.subheader("Running reconstruction...")

    v = PHANTOM_2D.flatten()

    tc = NOISE_PRESETS[noise_level]
    noisy = tc is not None

    y, n_vec, _, _ = simulate_measurements(
        P_MATRIX,
        v,
        total_counts=tc if noisy else 1_000_000,
        scatter_frac=scatter,
        noisy=noisy,
        seed=int(seed),
    )

    # Run selected algorithm
    if algorithm == "MAP":
        results = run_map(
            P_MATRIX,
            y,
            n_vec,
            v,
            ROI_MASKS,
            N=N,
            prior=prior,
            beta=beta,
            n_iter=iterations,
        )
    else:
        results = run_reconstruction(
            P_MATRIX,
            y,
            n_vec,
            v,
            ROI_MASKS,
            N=N,
            algorithm=algorithm,
            n_iter=iterations,
            n_subsets=subsets,
        )

    # Optional comparison
    if compare and algorithm != "MLEM":
        res_mlem = run_reconstruction(
            P_MATRIX,
            y,
            n_vec,
            v,
            ROI_MASKS,
            N=N,
            algorithm="MLEM",
            n_iter=iterations,
            n_subsets=1,
        )
    else:
        res_mlem = None

    st.success("✅ Reconstruction complete")

    # -------------------------------------------------
    # IMAGE DISPLAY
    # -------------------------------------------------
    st.subheader("Reconstruction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(PHANTOM_2D, caption="True Phantom", clamp=True)

    with col2:
        st.image(results["x_final"], caption=algorithm, clamp=True)

    with col3:
        diff = PHANTOM_2D - results["x_final"]
        st.image(diff, caption="Difference", clamp=True)

    # -------------------------------------------------
    # ITERATION VIEWER
    # -------------------------------------------------
    st.subheader("Iteration Viewer")

    if len(results["images"]) > 0:
        iter_view = st.slider(
            "Select iteration",
            min(results["images"].keys()),
            max(results["images"].keys()),
            max(results["images"].keys()),
        )
        st.image(results["images"][iter_view], caption=f"Iteration {iter_view}")

    # -------------------------------------------------
    # METRICS
    # -------------------------------------------------
    st.subheader("Metrics")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.plot(results["sse"], label=algorithm)
        if res_mlem is not None:
            ax.plot(res_mlem["sse"], "--", label="MLEM")
        ax.set_title("SSE")
        ax.legend()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.semilogy(results["kl"], label=algorithm)
        if res_mlem is not None:
            ax.semilogy(res_mlem["kl"], "--", label="MLEM")
        ax.set_title("KL Divergence")
        ax.legend()
        st.pyplot(fig)

    # -------------------------------------------------
    # ROI PLOTS
    # -------------------------------------------------
    st.subheader("ROI Convergence")

    fig, ax = plt.subplots()

    for name in ["hot", "tumour"]:
        ax.plot(results["roi"][name], label=name)

    ax.set_title("ROI Mean")
    ax.legend()
    st.pyplot(fig)
