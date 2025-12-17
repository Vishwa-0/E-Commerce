import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Customer Segmentation",
    layout="centered"
)

# ---------------- Header ----------------
st.title("🛒 Customer Segmentation")
st.caption("DBSCAN clustering with parameters fixed via k-distance analysis")

st.divider()

# ---------------- Load Data & Artifacts ----------------
customer_df = pd.read_csv("customer_summary.csv")
X = customer_df[["TotalQuantity", "TotalSpending"]]

with open("dbscan_ecommerce.pkl", "rb") as f:
    dbscan = pickle.load(f)

with open("scaler_ecommerce.pkl", "rb") as f:
    scaler = pickle.load(f)

X_scaled = scaler.transform(X)

# ---------------- Context (collapsible, calm) ----------------
with st.expander("How this model works"):
    st.write(
        """
        This application uses **DBSCAN**, a density-based clustering algorithm.

        Instead of forcing customers into fixed groups, DBSCAN discovers natural
        patterns based on how similar customers are to one another.

        The clustering parameters were selected during training using
        k-distance analysis and are fixed for consistent evaluation.
        """
    )

# ---------------- User Input ----------------
st.subheader("Test a customer profile")

qty = st.slider(
    "Total quantity purchased",
    int(X["TotalQuantity"].min()),
    int(X["TotalQuantity"].max()),
    int(X["TotalQuantity"].median())
)

spending = st.slider(
    "Total spending",
    int(X["TotalSpending"].min()),
    int(X["TotalSpending"].max()),
    int(X["TotalSpending"].median())
)

run = st.button("Analyze customer", use_container_width=True)

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

    # ---------------- Interpretation ----------------
    cluster_descriptions = {
        0: "customers with moderate purchasing activity",
        1: "high-value customers who purchase frequently",
        2: "customers with low purchasing engagement"
    }

    st.subheader("Customer insight")

    if user_cluster == -1:
        st.warning(
            "This customer shows an unusual purchasing pattern and does not closely "
            "match any common customer group."
        )
    else:
        st.success(
            f"This customer most closely matches "
            f"**{cluster_descriptions.get(user_cluster, 'similar customers')}**."
        )

    st.divider()

    # ---------------- Visualization ----------------
    fig, ax = plt.subplots(figsize=(6, 4))

    for c in set(labels):
        subset = customer_df[customer_df["Cluster"] == c]
        label = "Unusual customers" if c == -1 else "Customer group"

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
        s=200,
        c="gold",
        edgecolors="black",
        linewidths=1.5,
        label="Selected customer"
    )

    ax.set_xlabel("Total quantity purchased")
    ax.set_ylabel("Total spending")
    ax.set_title("Customer behavior map")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

else:
    st.info("Adjust the values above and click **Analyze customer**.")
