import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model
model = joblib.load("customer_segmentation_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset
df = pd.read_csv("final_customer_dataset.csv")

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("Customer Segmentation Dashboard")

st.write("Predict which customer segment a new user belongs to.")

# -----------------------
# Sidebar Inputs
# -----------------------

st.sidebar.header("Customer Inputs")

age = st.sidebar.slider("Age",18,80,35)

income = st.sidebar.number_input("Income",1000,200000,50000)

total_spent = st.sidebar.number_input("Total Spent",0,3000,500)

education = st.sidebar.selectbox(
    "Education",
    sorted(df["Education"].unique())
)

children = st.sidebar.slider("Children",0,5,1)

marital_status = st.sidebar.selectbox(
    "Marital Status",
    sorted(df["Marital_Status"].unique())
)

# -----------------------
# Create Input Data
# -----------------------

input_data = pd.DataFrame({
    "Age":[age],
    "Income":[income],
    "TotalSpent":[total_spent],
    "Education":[education],
    "Children":[children],
    "Marital_Status":[marital_status]
})

# Scale data
scaled_input = scaler.transform(input_data)

# -----------------------
# Prediction
# -----------------------

if st.sidebar.button("Predict Customer Segment"):

    cluster = model.predict(scaled_input)[0]

    st.subheader("Predicted Customer Cluster")

    st.success(f"Customer belongs to Cluster {cluster}")

# -----------------------
# Dataset Visualizations
# -----------------------

st.subheader("Dataset Overview")

col1,col2 = st.columns(2)

with col1:

    fig1 = px.histogram(
        df,
        x="Income",
        title="Income Distribution"
    )

    st.plotly_chart(fig1)

with col2:

    fig2 = px.histogram(
        df,
        x="TotalSpent",
        title="Total Spending Distribution"
    )

    st.plotly_chart(fig2)

# -----------------------
# Cluster Visualization
# -----------------------

st.subheader("Customer Segments")

fig3 = px.scatter(
    df,
    x="Income",
    y="TotalSpent",
    color="cluster",
    title="Income vs Spending by Cluster"
)

st.plotly_chart(fig3)

# -----------------------
# Cluster Distribution
# -----------------------

st.subheader("Cluster Distribution")

fig4 = px.histogram(
    df,
    x="cluster",
    title="Customer Segment Count"
)

st.plotly_chart(fig4)