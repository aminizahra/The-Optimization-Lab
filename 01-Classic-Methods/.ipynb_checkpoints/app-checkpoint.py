import streamlit as st
import numpy as np
import plotly.graph_objects as go
from benchmarks import Sphere, Rosenbrock, Rastrigin
from optimizers import GradientDescent, NewtonsMethod, VanillaSGD

# --- Page Configuration ---
st.set_page_config(page_title="Optimization Lab", layout="wide")
st.title("🚀 Optimization Lab: Visualizing Convergence")
st.markdown("""
This dashboard visualizes how different optimization algorithms navigate mathematical landscapes to find global minima.
Created as part of the **Pattern Recognition Portfolio**.
""")

# --- Sidebar: Configuration ---
st.sidebar.header("🛠️ Settings")

# 1. Function Selection
func_option = st.sidebar.selectbox(
    "Select Benchmark Function",
    ("Sphere (Convex)", "Rosenbrock (Narrow Valley)", "Rastrigin (Multi-modal)")
)

if func_option == "Sphere (Convex)":
    func = Sphere()
elif func_option == "Rosenbrock (Narrow Valley)":
    func = Rosenbrock()
else:
    func = Rastrigin()

# 2. Algorithm Selection
opt_option = st.sidebar.selectbox(
    "Select Optimizer",
    ("Gradient Descent", "Newton's Method", "Vanilla SGD")
)

# 3. Hyperparameters
col1, col2 = st.sidebar.columns(2)
with col1:
    lr = st.number_input("Learning Rate", value=0.01, step=0.001, format="%.3f")
with col2:
    iters = st.number_input("Iterations", value=50, step=10)

# 4. Starting Point (Dynamic based on function bounds)
st.sidebar.subheader("📍 Starting Point")
start_x = st.sidebar.slider("Start X", float(func.bounds[0]), float(func.bounds[1]), float(func.bounds[0]*0.8))
start_y = st.sidebar.slider("Start Y", float(func.bounds[0]), float(func.bounds[1]), float(func.bounds[1]*0.8))

# --- Engine: Solve Optimization ---
if st.sidebar.button("Run Optimization"):
    # Initialize Optimizer
    if opt_option == "Gradient Descent":
        optimizer = GradientDescent(learning_rate=lr, n_iterations=iters)
    elif opt_option == "Newton's Method":
        optimizer = NewtonsMethod(n_iterations=iters) # Newton doesn't strictly need LR but can be added
    else:
        optimizer = VanillaSGD(learning_rate=lr, n_iterations=iters, noise_level=0.1)

    # Run
    history = optimizer.solve(func, start_x, start_y)
    
    # --- Main Panel: Visualization ---
    tab1, tab2 = st.tabs(["📊 3D Surface View", "🗺️ 2D Contour View"])
    
    # Prepare Mesh Data
    X, Y, Z = func.get_mesh(resolution=100)
    path_x = history[:, 0]
    path_y = history[:, 1]
    path_z = func(path_x, path_y)

    with tab1:
        fig_3d = go.Figure(data=[
            go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.8),
            go.Scatter3d(x=path_x, y=path_y, z=path_z, mode='lines+markers',
                         line=dict(color='red', width=4),
                         marker=dict(size=3, color='black'))
        ])
        fig_3d.update_layout(title=f"3D Path on {func.name}", autosize=False,
                             width=800, height=800, margin=dict(l=65, r=50, b=65, t=90))
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab2:
        fig_contour = go.Figure(data=[
            go.Contour(x=X[0], y=Y[:,0], z=Z, colorscale='Viridis'),
            go.Scatter(x=path_x, y=path_y, mode='lines+markers',
                       line=dict(color='white', width=2),
                       marker=dict(size=6, color='red'))
        ])
        fig_contour.update_layout(title=f"2D Convergence Path", xaxis_title="X", yaxis_title="Y")
        st.plotly_chart(fig_contour, use_container_width=True)

    # --- Metrics & Insights ---
    st.subheader("📝 Execution Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Final X", f"{path_x[-1]:.4f}")
    c2.metric("Final Y", f"{path_y[-1]:.4f}")
    c3.metric("Final Value", f"{path_z[-1]:.4e}")

    st.info(f"**Insight:** {opt_option} reached the final value in {iters} iterations. "
            f"Observe how the path {'oscillates' if opt_option == 'Vanilla SGD' else 'straightens'} "
            f"depending on the function's curvature.")
else:
    st.info("👈 Adjust settings in the sidebar and click 'Run Optimization' to see the magic!")