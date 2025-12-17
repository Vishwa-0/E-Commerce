import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import pairwise_distances

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="ShopScope",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- Global CSS ----------------
st.markdown("""
<style>
body { background-color: #0f172a; }

.block-container { padding-top: 2rem; }

h1, h2, h3 { color: #e5e7eb; }
p, li { color: #cbd5e1; }

.hero {
    padding: 2.5rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #1e293b, #020617);
    margin-bottom: 2rem;
}

.glass-card {
    background: rgba(30, 41, 59, 0.65);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(255,255,255,0.05);
}

.result-card {
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
    background: linear-gradient(135deg, #22c55e, #15803d);
}

.outlier-card {
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
    background: linear-gradient(135deg, #ef4444, #991b1b);
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, "dbscan_ecommerce.pkl"), "rb") as f:
        dbscan = pickle.load(f)
    with open(os.path.join(base, "scaler_ecommerce.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return dbscan, scaler

dbscan, scaler = load_models()

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    return pd.read_csv(os.path.join(base, "data.csv"))

df = load_data()

X_train = df[["TotalQuantity", "TotalSpending"]]
X_train_scaled = scaler.transform(X_train)

# ---------------- DBSCAN Assignment ----------------
def assign_cluster(dbscan, X_train_scaled, X_new_scaled):
    core_samples = X_train_scaled[dbscan.core_sample_indices_]
    core_labels = dbscan.labels_[dbscan.core_sample_indices_]

    distances = pairwise_distances(X_new_scaled, core_samples)
    min_dist = distances.min()
    nearest_idx = distances.argmin()

    if min_dist <= dbscan.eps:
        return core_labels[nearest_idx]
    else:
        return -1

# ---------------- Slider Ranges (Percentiles) ----------------
qty_min = int(df["TotalQuantity"].quantile(0.05))
qty_max = int(df["TotalQuantity"].quantile(0.95))
qty_med = int(df["TotalQuantity"].median())

sp_min = float(df["TotalSpending"].quantile(0.05))
sp_max = float(df["TotalSpending"].quantile(0.95))
sp_med = float(df["TotalSpending"].median())

# ---------------- Hero ----------------
st.markdown("""
<div class="hero">
    <h1>ShopScope</h1>
    <h3>E-Commerce Customer Segmentation</h3>
    <p>
        This system uses DBSCAN to discover natural purchasing patterns.
        Clusters represent real customer behaviors, while outliers indicate
        unusual spending activity.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- Metrics ----------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Algorithm", "DBSCAN")
m2.metric("Features", "Quantity & Spending")
m3.metric("Scaling", "StandardScaler")
m4.metric("Outlier Detection", "Enabled")

# ---------------- Layout ----------------
left, center, right = st.columns([1.3, 1.8, 1.4])

# -------- Left --------
with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("How to Read Results")

    st.write("""
    - **Clusters** represent common purchasing patterns  
    - **Outliers** behave very differently from most customers  
    - The model is **pre-trained** and not re-fitted live  
    """)

    st.write("""
    This reflects the *learned structure of historical data*.
    New customers are evaluated against that structure.
    """)

    st.markdown('</div>', unsafe_allow_html=True)

# -------- Center --------
with center:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("Test a Customer")

    preset = st.selectbox(
        "Customer Profile",
        ["Custom", "Low Engagement", "Typical Customer", "High Value", "Extreme Case"]
    )

    if preset == "Low Engagement":
        qty = int(qty_min + 0.15 * (qty_max - qty_min))
        spending = sp_min + 0.15 * (sp_max - sp_min)
    elif preset == "Typical Customer":
        qty = qty_med
        spending = sp_med
    elif preset == "High Value":
        qty = int(qty_min + 0.75 * (qty_max - qty_min))
        spending = sp_min + 0.75 * (sp_max - sp_min)
    elif preset == "Extreme Case":
        qty = qty_min
        spending = sp_max
    else:
        qty = st.slider("Total Quantity Purchased", qty_min, qty_max, qty_med)
        spending = st.slider("Total Spending", sp_min, sp_max, sp_med)

    analyze = st.button("Analyze Customer", use_container_width=True)

    if analyze:
        X_new = np.array([[qty, spending]])
        X_new_scaled = scaler.transform(X_new)

        cluster = assign_cluster(
            dbscan,
            X_train_scaled,
            X_new_scaled
        )

        if cluster == -1:
            st.markdown("""
            <div class="outlier-card">
                <h2>Unusual Customer</h2>
                <p>
                This purchasing pattern does not match typical customers.
                It may represent rare or exceptional behavior.
                </p>
            </div>
            """, unsafe_allow_html=True)

        elif cluster == 0:
            st.markdown("""
            <div class="result-card">
                <h2>Low Engagement Customer</h2>
                <p>
                Limited purchasing activity and lower spending levels.
                </p>
            </div>
            """, unsafe_allow_html=True)

        elif cluster == 1:
            st.markdown("""
            <div class="result-card">
                <h2>High Value Customer</h2>
                <p>
                Frequent purchases and consistently higher spending.
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------- Right --------
with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("Cluster Summary")

    with st.expander("Low Engagement"):
        st.write("Infrequent purchases and low total spending.")

    with st.expander("High Value"):
        st.write("Strong purchasing patterns with higher revenue contribution.")

    with st.expander("Outliers"):
        st.write("Customers with rare or abnormal behavior patterns.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("""
<div class="footer">
    ShopScope • Customer Segmentation • Educational & Analytical Use
</div>
""", unsafe_allow_html=True)
