import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="E-Commerce DBSCAN Segmentation",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.glass {
    background: rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 22px;
    border: 1px solid rgba(255,255,255,0.3);
}
.metric {
    font-size: 26px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown(
    "<h1 style='text-align:center;'>🛒 E-Commerce Customer Segmentation</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Interactive DBSCAN clustering with density control</p>",
    unsafe_allow_html=True
)

st.divider()

# ---------------- Load Data ----------------
customer_df = pd.read_csv("customer_summary.csv")

X = customer_df[["TotalQuantity", "TotalSpending"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- Controls ----------------
with st.container():
    c1, c2, c3 = st.columns(3)

    with c1:
        qty = st.slider(
            "Total Quantity Purchased",
            int(X["TotalQuantity"].min()),
            int(X["TotalQuantity"].max()),
            int(X["TotalQuantity"].median())
        )

    with c2:
        spending = st.slider(
            "Total Spending",
            int(X["TotalSpending"].min()),
            int(X["TotalSpending"].max()),
            int(X["TotalSpending"].median())
        )

    with c3:
        eps = st.slider(
            "DBSCAN eps (density radius)",
            0.2, 1.5, 0.6, 0.05
        )

run = st.button("Run DBSCAN", use_container_width=True)

st.divider()

# ---------------- Action ----------------
if run:
    dbscan = DBSCAN(eps=eps, min_samples=5)

    labels = dbscan.fit_predict(X_scaled)
    customer_df["Cluster"] = labels

    user_point = np.array([[qty, spending]])
    user_scaled = scaler.transform(user_point)

    combined = np.vstack([X_scaled, user_scaled])
    combined_labels = dbscan.fit_predict(combined)
    user_cluster = combined_labels[-1]

    # -------- KPIs --------
    k1, k2, k3 = st.columns(3)

    k1.markdown(
        f"<div class='glass metric'>Customers<br>{len(customer_df)}</div>",
        unsafe_allow_html=True
    )

    k2.markdown(
        f"<div class='glass metric'>Clusters<br>{len(set(labels)) - (1 if -1 in labels else 0)}</div>",
        unsafe_allow_html=True
    )

    k3.markdown(
        f"<div class='glass metric'>Outliers<br>{list(labels).count(-1)}</div>",
        unsafe_allow_html=True
    )

    st.divider()

    # -------- Result + Plot --------
    left, right = st.columns([1, 1.4])

    with left:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("📌 Customer Result")

        if user_cluster == -1:
            st.error("Outlier customer with rare purchasing behavior.")
        else:
            st.success(f"Customer belongs to Cluster {user_cluster}")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        fig, ax = plt.subplots(figsize=(7, 5))

        for c in set(labels):
            subset = customer_df[customer_df["Cluster"] == c]
            label = "Outliers" if c == -1 else f"Cluster {c}"

            ax.scatter(
                subset["TotalQuantity"],
                subset["TotalSpending"],
                label=label,
                s=55,
                alpha=0.6
            )

        ax.scatter(
            qty, spending,
            s=260,
            c="yellow",
            edgecolors="black",
            linewidths=2,
            label="User"
        )

        ax.set_xlabel("Total Quantity")
        ax.set_ylabel("Total Spending")
        ax.set_title("DBSCAN Clustering Map")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

else:
    st.info("Adjust sliders and click **Run DBSCAN** to explore customer clusters.")

