import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="ShopScope",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- Global CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #e5e7eb;
}

p, li {
    color: #cbd5e1;
}

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
    with open("dbscan_ecommerce.pkl", "rb") as f:
        dbscan = pickle.load(f)
    with open("scaler_ecommerce.pkl", "rb") as f:
        scaler = pickle.load(f)
    return dbscan, scaler

dbscan, scaler = load_models()

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data():
    return pd.read_csv("customer_summary.csv")

df = load_data()

# ---------------- Slider Ranges (Percentiles) ----------------
qty_min = int(df["TotalQuantity"].quantile(0.05))
qty_max = int(df["TotalQuantity"].quantile(0.95))
qty_med = int(df["TotalQuantity"].median())

spend_min = float(df["TotalSpending"].quantile(0.05))
spend_max = float(df["TotalSpending"].quantile(0.95))
spend_med = float(df["TotalSpending"].median())

# ---------------- Hero ----------------
st.markdown("""
<div class="hero">
    <h1>ShopScope</h1>
    <h3>Customer Segmentation with DBSCAN</h3>
    <p>
        This app groups customers based on purchasing behavior using density-based
        clustering. It helps identify low-engagement users, regular shoppers,
        high-value customers, and unusual outliers.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- Metrics ----------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Algorithm", "DBSCAN")
m2.metric("Features", "2")
m3.metric("Scaling", "StandardScaler")
m4.metric("Outlier Detection", "Enabled")

# ---------------- Layout ----------------
left, center, right = st.columns([1.3, 1.8, 1.4])

# -------- Left Panel --------
with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("What This Means")

    st.write("""
    **DBSCAN** groups customers based on *behavior similarity*:

    - **Cluster 0 / 1** → Real customer segments  
    - **Outliers (-1)** → Unusual behavior patterns  

    Clusters are formed automatically using density, not predefined labels.
    """)

    st.write("""
    Typical use cases:
    - Marketing personalization
    - Loyalty analysis
    - Detecting abnormal spenders
    """)

    st.markdown('</div>', unsafe_allow_html=True)

# -------- Center Panel --------
with center:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("Test a Customer")

    preset = st.selectbox(
        "Quick Profiles",
        ["Custom", "Low Engagement", "Regular Customer", "High Value", "Extreme Outlier"]
    )

    if preset == "Low Engagement":
        qty = int(qty_min + (qty_max - qty_min) * 0.15)
        spending = spend_min + (spend_max - spend_min) * 0.15
    elif preset == "Regular Customer":
        qty = qty_med
        spending = spend_med
    elif preset == "High Value":
        qty = int(qty_min + (qty_max - qty_min) * 0.75)
        spending = spend_min + (spend_max - spend_min) * 0.75
    elif preset == "Extreme Outlier":
        qty = qty_min
        spending = spend_max
    else:
        qty = st.slider(
            "Total Quantity Purchased",
            qty_min, qty_max, qty_med
        )
        spending = st.slider(
            "Total Spending",
            spend_min, spend_max, spend_med
        )

    analyze = st.button("Analyze Customer", use_container_width=True)

    if analyze:
        X_input = np.array([[qty, spending]])
        X_scaled = scaler.transform(X_input)
        cluster = dbscan.fit_predict(X_scaled)[0]

        if cluster == -1:
            st.markdown("""
            <div class="outlier-card">
                <h2>Outlier Detected</h2>
                <p>This customer behaves very differently from typical shoppers.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card">
                <h2>Customer Cluster {cluster}</h2>
                <p>This customer belongs to a recognized behavior group.</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------- Right Panel --------
with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("Cluster Interpretation")

    with st.expander("Cluster 0"):
        st.write("Lower engagement customers with limited purchasing activity.")

    with st.expander("Cluster 1"):
        st.write("Consistent and higher-value customers with regular purchases.")

    with st.expander("Outliers (-1)"):
        st.write("Unusual buying patterns that don’t match typical customer behavior.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("""
<div class="footer">
    ShopScope • E-commerce Customer Clustering • Educational & Analytical Use
</div>
""", unsafe_allow_html=True)
