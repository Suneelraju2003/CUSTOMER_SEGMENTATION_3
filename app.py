import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# -----------------------------
# Load model and dataset
# -----------------------------

model = joblib.load("customer_segmentation_model.pkl")

df = pd.read_csv("final_customer_dataset.csv")

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("Customer Personality Analysis - Customer Segmentation")

st.markdown(
"""
This dashboard predicts which **customer segment (cluster)** a user belongs to
based on demographic and purchasing behaviour.
"""
)

# -----------------------------
# Sidebar Inputs
# -----------------------------

st.sidebar.header("Customer Input")

education = st.sidebar.selectbox(
    "Education",
    sorted(df["Education"].unique())
)

marital_status = st.sidebar.selectbox(
    "Marital Status",
    sorted(df["Marital_Status"].unique())
)

income = st.sidebar.number_input(
    "Income",
    min_value=1000,
    max_value=200000,
    value=50000
)

recency = st.sidebar.slider(
    "Recency (Days since last purchase)",
    0,100,30
)

age = st.sidebar.slider(
    "Age",
    18,80,35
)

children = st.sidebar.slider(
    "Children",
    0,5,1
)

total_spent = st.sidebar.number_input(
    "Total Spent",
    min_value=0,
    max_value=3000,
    value=500
)

campaign = st.sidebar.slider(
    "Campaigns Accepted",
    0,6,1
)

total_purchases = st.sidebar.slider(
    "Total Purchases",
    0,30,10
)

# -----------------------------
# Encode categorical values
# -----------------------------

education_map = {"UG":0,"PG":1}
marital_map = {"Single":0,"Relationship":1}

education_val = education_map.get(education,0)
marital_val = marital_map.get(marital_status,0)

# -----------------------------
# Create input dataframe
# -----------------------------

input_data = pd.DataFrame({
    "Education":[education_val],
    "Marital_Status":[marital_val],
    "Income":[income],
    "Recency":[recency],
    "Age":[age],
    "Children":[children],
    "TotalSpent":[total_spent],
    "TotalCampaignAccepted":[campaign],
    "TotalPurchases":[total_purchases]
})

# -----------------------------
# Prediction
# -----------------------------

if st.sidebar.button("Predict Customer Cluster"):

    prediction = model.predict(input_data)[0]

    st.subheader("Predicted Customer Segment")

    st.success(f"This customer belongs to **Cluster {prediction}**")

# -----------------------------
# Dataset Overview
# -----------------------------

st.subheader("Dataset Overview")

col1,col2 = st.columns(2)

with col1:

    fig1 = px.histogram(
        df,
        x="Age",
        title="Age Distribution",
        color_discrete_sequence=["skyblue"]
    )

    st.plotly_chart(fig1, use_container_width=True)

with col2:

    fig2 = px.histogram(
        df,
        x="Income",
        title="Income Distribution",
        color_discrete_sequence=["orange"]
    )

    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Spending Insights
# -----------------------------

st.subheader("Spending Behaviour")

col3,col4 = st.columns(2)

with col3:

    fig3 = px.histogram(
        df,
        x="TotalSpent",
        title="Total Spending Distribution",
        color_discrete_sequence=["green"]
    )

    st.plotly_chart(fig3, use_container_width=True)

with col4:

    fig4 = px.histogram(
        df,
        x="Recency",
        title="Recency Distribution",
        color_discrete_sequence=["pink"]
    )

    st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# Cluster Visualization
# -----------------------------

st.subheader("Customer Segments")

fig5 = px.scatter(
    df,
    x="Income",
    y="TotalSpent",
    color="cluster",
    title="Income vs TotalSpent by Cluster"
)

st.plotly_chart(fig5, use_container_width=True)

# -----------------------------
# Age vs Spending
# -----------------------------

fig6 = px.scatter(
    df,
    x="Age",
    y="TotalSpent",
    color="cluster",
    title="Age vs TotalSpent by Cluster"
)

st.plotly_chart(fig6, use_container_width=True)

# -----------------------------
# Cluster Distribution
# -----------------------------

st.subheader("Cluster Distribution")

fig7 = px.histogram(
    df,
    x="cluster",
    title="Customer Segment Distribution"
)

st.plotly_chart(fig7, use_container_width=True)

# -----------------------------
# Cluster Profile Table
# -----------------------------

st.subheader("Cluster Profile")

cluster_profile = df.groupby("cluster").mean()

st.dataframe(cluster_profile)
