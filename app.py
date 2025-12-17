import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="E-Commerce DBSCAN Segmentation",
    page_icon="🛒",
    layout="wide"
)

# ---------------- Header ----------------
st.markdown(
    """
    <h1 style='text-align: center;'>🛒 E-Commerce Customer Segmentation</h1>
    <p style='text-align: center; font-size: 18px;'>
    Density-based clustering using DBSCAN to identify customer behavior patterns
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- Load Data ----------------
customer_df = pd.read_csv("customer_summary.csv")

with open("dbscan_ecommerce.pkl", "rb") as f:
    dbscan = pickle.load(f)

X = customer_df[["TotalQuantity", "TotalSpending"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

labels = dbscan.fit_predict(X_scaled)
customer_df["Cluster"] = labels

# ---------------- Sidebar ----------------
st.sidebar.header("Customer Simulation Panel")

qty = st.sidebar.number_input(
    "Total Quantity Purchased",
    min_value=1.0,
    value=float(customer_df["TotalQuantity"].median()),
    step=10.0
)

spending = st.sidebar.number_input(
    "Total Spending",
    min_value=1.0,
    value=float(customer_df["TotalSpending"].median()),
    step=100.0
)

run_btn = st.sidebar.button("Run DBSCAN Analysis")

# ---------------- KPI Section ----------------
with st.container():
    k1, k2, k3 = st.columns(3)

    k1.metric("Total Customers", customer_df.shape[0])
    k2.metric(
        "Clusters Found",
        len(set(labels)) - (1 if -1 in labels else 0)
    )
    k3.metric(
        "Outliers Detected",
        list(labels).count(-1)
    )

st.divider()

# ---------------- Button Logic ----------------
if run_btn:
    user_point = np.array([[qty, spending]])
    user_scaled = scaler.transform(user_point)

    X_combined = np.vstack([X_scaled, user_scaled])
    labels_combined = dbscan.fit_predict(X_combined)
    user_cluster = labels_combined[-1]

    col1, col2 = st.columns([1, 1.2])

    # -------- Result Card --------
    with col1:
        st.subheader("Customer Classification")

        if user_cluster == -1:
            st.error(
                "This customer is an **outlier**.\n\n"
                "Their purchasing behavior is rare compared to the existing population."
            )
        else:
            st.success(
                f"This customer belongs to **Cluster {user_cluster}**.\n\n"
                "Their behavior matches a dense customer group."
            )

        st.markdown("### Input Summary")
        st.write(f"- **Total Quantity:** {qty}")
        st.write(f"- **Total Spending:** {spending}")

    # -------- Visualization --------
    with col2:
        st.subheader("Cluster Map")

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
            qty,
            spending,
            c="yellow",
            s=250,
            edgecolors="black",
            linewidths=2,
            label="User Input"
        )

        ax.set_xlabel("Total Quantity Purchased")
        ax.set_ylabel("Total Spending")
        ax.set_title("DBSCAN Customer Segmentation")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

else:
    st.info(
        "Enter customer values in the sidebar and click **Run DBSCAN Analysis** "
        "to classify the customer and visualize clusters."
    )

st.divider()

# ---------------- Data Preview ----------------
with st.expander("View Processed Customer Dataset"):
    st.dataframe(customer_df.head(15))
