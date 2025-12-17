import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="E-Commerce Customer Segmentation (DBSCAN)",
    layout="wide"
)

# ---------------- Title ----------------
st.title("🛒 E-Commerce Customer Segmentation using DBSCAN")
st.caption("Density-based clustering on customer purchasing behavior")

# ---------------- Load PROCESSED Data ---------------
customer_df = pd.read_csv("customer_summary.csv")

# ---------------- Load Model ----------------
with open("dbscan_ecommerce.pkl", "rb") as f:
    dbscan = pickle.load(f)

# ---------------- Feature Selection ----------------
X = customer_df[["TotalQuantity", "TotalSpending"]]

# ---------------- Scaling ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

labels = dbscan.fit_predict(X_scaled)
customer_df["Cluster"] = labels

# ---------------- Sidebar Input ----------------
st.sidebar.header("New Customer Input")

qty = st.sidebar.number_input(
    "Total Quantity Purchased",
    min_value=1.0,
    value=float(customer_df["TotalQuantity"].median())
)

spending = st.sidebar.number_input(
    "Total Spending",
    min_value=1.0,
    value=float(customer_df["TotalSpending"].median())
)

user_point = np.array([[qty, spending]])
user_scaled = scaler.transform(user_point)

X_combined = np.vstack([X_scaled, user_scaled])
labels_combined = dbscan.fit_predict(X_combined)
user_cluster = labels_combined[-1]

# ---------------- Results ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Classification")

    if user_cluster == -1:
        st.warning("This customer is an outlier (unusual purchasing behavior).")
    else:
        st.success(f"This customer belongs to cluster {user_cluster}")

    st.metric("Total Customers", customer_df.shape[0])
    st.metric(
        "Clusters Found",
        len(set(labels)) - (1 if -1 in labels else 0)
    )

with col2:
    st.subheader("Cluster Visualization")

    fig, ax = plt.subplots(figsize=(7, 5))

    for c in set(labels):
        subset = customer_df[customer_df["Cluster"] == c]
        label = "Outliers" if c == -1 else f"Cluster {c}"

        ax.scatter(
            subset["TotalQuantity"],
            subset["TotalSpending"],
            label=label,
            s=60,
            alpha=0.6
        )

    ax.scatter(
        qty,
        spending,
        c="yellow",
        s=220,
        edgecolors="black",
        label="User"
    )

    ax.set_xlabel("Total Quantity Purchased")
    ax.set_ylabel("Total Spending")
    ax.set_title("DBSCAN Clusters with User Input")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

# ---------------- Dataset Preview ----------------
st.subheader("Customer Dataset Preview")
st.dataframe(customer_df.head(10))
