import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

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
.sidebox {
    background: rgba(255, 255, 255, 0.12);
    border-radius: 14px;
    padding: 16px;
    border: 1px solid rgba(255,255,255,0.2);
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown(
    "<h1 style='text-align:center;'>🛒 Customer Behavior Segmentation</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>DBSCAN clustering with parameters fixed via k-distance analysis</p>",
    unsafe_allow_html=True
)

st.divider()

# ---------------- Load Data & Artifacts ----------------
customer_df = pd.read_csv("customer_summary.csv")
X = customer_df[["TotalQuantity", "TotalSpending"]]

with open("dbscan_ecommerce.pkl", "rb") as f:
    dbscan = pickle.load(f)

with open("scaler_ecommerce.pkl", "rb") as f:
    scaler = pickle.load(f)

X_scaled = scaler.transform(X)

# ---------------- Layout Columns ----------------
left, center, right = st.columns([1, 3, 1])

# ---------------- Left Panel ----------------
with left:
    st.markdown("<div class='sidebox'>", unsafe_allow_html=True)
    st.markdown("### About the Model")
    st.markdown(
        """
        This application uses **DBSCAN**, a density-based clustering algorithm.

        It discovers natural customer behavior patterns based on similarity,
        rather than forcing customers into predefined groups.

        Customers who do not fit well into any pattern are marked as *unusual*
        instead of being incorrectly grouped.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Center Panel ----------------
with center:
    st.info(
        "The clustering structure was learned during model training. "
        "The density parameter (eps) was selected using k-distance analysis "
        "and is fixed to ensure consistent evaluation."
    )

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

# ---------------- Right Panel ----------------
with right:
    st.markdown("<div class='sidebox'>", unsafe_allow_html=True)
    st.markdown("###How to Read Results")
    st.markdown(
        """
        • Each dot represents a customer  
        • Distance indicates similarity  
        • Groups form where behavior is similar  
        • The yellow dot shows the selected customer  

        If a customer is labeled *unusual*, it simply means
        their behavior is rare compared to others.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

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
        0: "regular customers with moderate purchasing behavior",
        1: "high-value customers who purchase frequently and spend more",
        2: "low-engagement customers with infrequent or minimal purchases"
    }

    # ---------------- KPIs ----------------
    k1, k2, k3 = st.columns(3)

    k1.markdown(
        f"<div class='glass metric'>Total Customers<br>{len(customer_df)}</div>",
        unsafe_allow_html=True
    )

    k2.markdown(
        f"<div class='glass metric'>Customer Groups<br>{len(set(labels)) - (1 if -1 in labels else 0)}</div>",
        unsafe_allow_html=True
    )

    k3.markdown(
        f"<div class='glass metric'>Unusual Customers<br>{list(labels).count(-1)}</div>",
        unsafe_allow_html=True
    )

    st.divider()

    # ---------------- Result ----------------
    st.subheader("Customer Insight")

    if user_cluster == -1:
        st.warning(
            "This customer shows an unusual purchasing pattern and does not closely "
            "match any common customer group."
        )
    else:
        st.success(
            f"This customer belongs to a group of "
            f"**{cluster_descriptions.get(user_cluster, 'similar customers')}**."
        )

    st.divider()

    # ---------------- Visualization ----------------
    fig, ax = plt.subplots(figsize=(6, 4))

    for c in set(labels):
        subset = customer_df[customer_df["Cluster"] == c]
        label = "Unusual Customers" if c == -1 else "Customer Group"

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
    st.info("Enter purchase details and click **Analyze Customer** to see where the customer fits.")
