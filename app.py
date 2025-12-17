import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Customer Segmentation Insight",
    layout="centered"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.glass {
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.25);
}
.metric {
    font-size: 24px;
    font-weight: 700;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown(
    "<h1 style='text-align:center;'>Customer Behavior Segmentation</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Customer groups learned using DBSCAN (eps fixed via k-distance analysis)</p>",
    unsafe_allow_html=True
)

st.divider()

st.info(
    "This model has already learned the natural structure of historical customer behavior. "
    "The density parameter (eps) was selected during training using k-distance analysis and "
    "is fixed to ensure consistent evaluation."
)

# ---------------- Load Data & Model ----------------
customer_df = pd.read_csv("customer_summary.csv")

X = customer_df[["TotalQuantity", "TotalSpending"]]

scaler = joblib.load("scaler.pkl")
dbscan = joblib.load("dbscan_model.pkl")

X_scaled = scaler.transform(X)

# ---------------- User Inputs ----------------
st.subheader("Test a New Customer")

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

run = st.button("Analyze Customer", use_container_width=True)

st.divider()

# ---------------- Action ----------------
if run:
    labels = dbscan.fit_predict(X_scaled)
    customer_df["Cluster"] = labels

    user_point = np.array([[qty, spending]])
    user_scaled = scaler.transform(user_point)

    combined = np.vstack([X_scaled, user_scaled])
    combined_labels = dbscan.fit_predict(combined)

    user_cluster = combined_labels[-1]

    # ---------------- Cluster Meaning ----------------
    cluster_descriptions = {
        0: "Regular customers with moderate purchase volume and spending",
        1: "High-value customers with strong purchasing power",
        2: "Low-engagement customers with infrequent purchases"
    }

    # ---------------- KPIs ----------------
    k1, k2, k3 = st.columns(3)

    k1.markdown(
        f"<div class='glass metric'>Total Customers<br>{len(customer_df)}</div>",
        unsafe_allow_html=True
    )

    k2.markdown(
        f"<div class='glass metric'>Identified Groups<br>{len(set(labels)) - (1 if -1 in labels else 0)}</div>",
        unsafe_allow_html=True
    )

    k3.markdown(
        f"<div class='glass metric'>Unusual Patterns<br>{list(labels).count(-1)}</div>",
        unsafe_allow_html=True
    )

    st.divider()

    # ---------------- Result ----------------
    st.subheader("Customer Insight")

    if user_cluster == -1:
        st.warning(
            "This customer shows an unusual purchasing pattern and does not closely match "
            "any common customer group."
        )
    else:
        st.success(
            f"This customer fits into a group of **{cluster_descriptions.get(user_cluster, 'similar customers')}**."
        )

    st.divider()

    # ---------------- Visualization ----------------
    fig, ax = plt.subplots(figsize=(6, 4))

    for c in set(labels):
        subset = customer_df[customer_df["Cluster"] == c]
        label = "Unusual Customers" if c == -1 else f"Customer Group"

        ax.scatter(
            subset["TotalQuantity"],
            subset["TotalSpending"],
            s=50,
            alpha=0.6,
            label=label
        )

    ax.scatter(
        qty,
        spending,
        s=220,
        c="yellow",
        edgecolors="black",
        linewidths=2,
        label="Selected Customer"
    )

    ax.set_xlabel("Total Quantity Purchased")
    ax.set_ylabel("Total Spending")
    ax.set_title("Customer Segmentation Map")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

else:
    st.info("Enter customer purchase details and click **Analyze Customer** to see where they fit.")
