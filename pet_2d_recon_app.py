import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def make_brain_phantom(N):
    # Example placeholder; replace with your actual code
    return np.zeros((N, N))

def make_roi_masks(N):
    # Example placeholder; replace with your actual code
    return {"hot": np.zeros((N, N)), "tumour": np.zeros((N, N))}

def make_projector(N):
    # Example placeholder; replace with your actual code
    P = np.eye(N*N)  # dummy projection matrix
    return P, None, None

def simulate_measurements(P, v, total_counts=1e6, scatter_frac=0.1, noisy=True, seed=42):
    # Example placeholder; replace with your actual code
    y = v
    n_vec = np.ones_like(v)
    return y, n_vec, None, None

def run_reconstruction(P, y, n_vec, v, roi_masks, N=64, algorithm="MLEM", n_iter=20, n_subsets=1):
    # Example placeholder; replace with your actual code
    results = {"x_final": v.reshape(N, N), "images": {i: v.reshape(N, N) for i in range(n_iter)}, "sse": [1]*n_iter, "kl": [1]*n_iter, "roi": {"hot": [0]*n_iter, "tumour": [0]*n_iter}}
    return results

def run_map(P, y, n_vec, v, roi_masks, N=64, prior=None, beta=0.05, n_iter=20):
    # Example placeholder; replace with your actual code
    return run_reconstruction(P, y, n_vec, v, roi_masks, N=N, algorithm="MAP", n_iter=n_iter)

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
NOISE_PRESETS = {
    "low": 1e5,
    "medium": 1e6,
    "high": 1e7
}

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
noise_level = st.sidebar.selectbox("Noise level", list(NOISE_PRESETS.keys()), index=2)
scatter = st.sidebar.slider("Scatter fraction", 0.0, 0.5, 0.1)
seed = st.sidebar.number_input("Random seed", value=42)
iterations = st.sidebar.slider("Iterations", 1, 100, 20)

subsets = 1
if algorithm == "OSEM":
    subsets = st.sidebar.slider("Subsets", 1, 16, 4)

beta = 0.05
prior = None
if algorithm == "MAP":
    prior = st.sidebar.selectbox(
        "Prior",
        ["Quadratic (smooth, blurs edges)", "MRP (edge-preserving, local median)", "TV  (edge-preserving, piecewise-flat)"]
    )
    beta = st.sidebar.slider("Beta", 0.0, 1.0, 0.05)

compare = st.sidebar.checkbox("Compare with MLEM")
run_button = st.sidebar.button("▶ Run Reconstruction")

# -------------------------------------------------
# HELP SECTION
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

    y, n_vec, _, _ = simulate_measurements(P_MATRIX, v, total_counts=tc if noisy else 1_000_000,
                                           scatter_frac=scatter, noisy=noisy, seed=int(seed))

    if algorithm == "MAP":
        results = run_map(P_MATRIX, y, n_vec, v, ROI_MASKS, N=N, prior=prior, beta=beta, n_iter=iterations)
    else:
        results = run_reconstruction(P_MATRIX, y, n_vec, v, ROI_MASKS, N=N, algorithm=algorithm,
                                     n_iter=iterations, n_subsets=subsets)

    res_mlem = None
    if compare and algorithm != "MLEM":
        res_mlem = run_reconstruction(P_MATRIX, y, n_vec, v, ROI_MASKS, N=N, algorithm="MLEM",
                                      n_iter=iterations, n_subsets=1)

    st.success("✅ Reconstruction complete")

    # IMAGE DISPLAY
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(PHANTOM_2D, caption="True Phantom", clamp=True)
    with col2:
        st.image(results["x_final"], caption=algorithm, clamp=True)
    with col3:
        st.image(PHANTOM_2D - results["x_final"], caption="Difference", clamp=True)

    # ITERATION VIEWER
    if len(results["images"]) > 0:
        iter_view = st.slider("Select iteration", min(results["images"].keys()),
                              max(results["images"].keys()), max(results["images"].keys()))
        st.image(results["images"][iter_view], caption=f"Iteration {iter_view}")

    # METRICS
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

    # ROI PLOTS
    st.subheader("ROI Convergence")
    fig, ax = plt.subplots()
    for name in ["hot", "tumour"]:
        ax.plot(results["roi"][name], label=name)
    ax.set_title("ROI Mean")
    ax.legend()
    st.pyplot(fig)
