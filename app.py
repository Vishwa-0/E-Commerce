import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="MarketScope",
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
    background: linear-gradient(135deg, #16a34a, #166534);
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Artifacts ----------------
@st.cache_resource
def load_models():
    with open("dbscan_ecommerce.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler_ecommerce.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

dbscan, scaler = load_models()

customer_df = pd.read_csv("customer_summary.csv")
X = customer_df[["TotalQuantity", "TotalSpending"]]
X_scaled = scaler.transform(X)

# ---------------- Hero Section ----------------
st.markdown("""
<div class="hero">
    <h1>MarketScope</h1>
    <h3>AI-driven Customer Behavior Segmentation</h3>
    <p>
        MarketScope analyzes historical purchasing behavior to discover
        natural customer groups using density-based clustering.
        Built for insight, strategy, and exploration — not rigid categorization.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- Metrics ----------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Algorithm", "DBSCAN")
m2.metric("Features", "2")
m3.metric("Data Type", "Transactional")
m4.metric("Inference", "< 1 sec")

# ---------------- Main Layout ----------------
left, center, right = st.columns([1.3, 1.8, 1.4])

# -------- Left --------
with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("About the Model")
    st.write("""
    MarketScope uses **DBSCAN**, a density-based clustering algorithm.

    Instead of forcing customers into fixed segments, it identifies
    natural behavior patterns based on purchasing similarity.

    Customers who do not fit common patterns are labeled as *unusual*
    rather than incorrectly grouped.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# -------- Center --------
with center:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("Customer Analysis")

    qty = st.slider(
        "Total Quantity Purchased",
        int(X["TotalQuantity"].min()),
        int(X["TotalQuantity"].max()),
        int(X["TotalQuantity"].median())
    )

    spending = st.slider(
        "Total Spending",
        int(X["TotalSpending"].min()),
        int(X["TotalSpending"].max()),
        int(X["TotalSpending"].median())
    )

    analyze = st.button("Analyze Customer", use_container_width=True)

    if analyze:
        labels = dbscan.fit_predict(X_scaled)
        customer_df["Cluster"] = labels

        user_point = np.array([[qty, spending]])
        user_scaled = scaler.transform(user_point)

        combined = np.vstack([X_scaled, user_scaled])
        combined_labels = dbscan.fit_predict(combined)
        user_cluster = combined_labels[-1]

        cluster_descriptions = {
            0: "Regular customers with steady purchasing behavior",
            1: "High-value customers with strong spending patterns",
            2: "Low-engagement customers with infrequent purchases"
        }

        if user_cluster == -1:
            st.markdown(
                """
                <div class="result-card">
                    <h2>Unusual Customer Pattern</h2>
                    <p>This customer does not closely match any common group.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-card">
                    <h2>{cluster_descriptions.get(user_cluster)}</h2>
                    <p>Based on similarity to existing customer behavior</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        fig, ax = plt.subplots(figsize=(6, 4))
        for c in set(labels):
            subset = customer_df[customer_df["Cluster"] == c]
            label = "Unusual" if c == -1 else "Customer Group"
            ax.scatter(
                subset["TotalQuantity"],
                subset["TotalSpending"],
                s=50,
                alpha=0.6,
                label=label
            )

        ax.scatter(
            qty, spending,
            s=220,
            c="gold",
            edgecolors="black",
            linewidths=1.5,
            label="Selected Customer"
        )

        ax.set_xlabel("Total Quantity")
        ax.set_ylabel("Total Spending")
        ax.set_title("Customer Behavior Map")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# -------- Right --------
with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("Customer Groups")

    with st.expander("High-Value Customers"):
        st.write("Frequent purchases and high total spending.")

    with st.expander("Regular Customers"):
        st.write("Consistent purchasing behavior over time.")

    with st.expander("Low-Engagement Customers"):
        st.write("Rare or minimal purchase activity.")

    with st.expander("Unusual Patterns"):
        st.write("Customers whose behavior does not fit common patterns.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("""
<div class="footer">
    MarketScope • Customer Segmentation • Educational & Analytical Use Only
</div>
""", unsafe_allow_html=True)
