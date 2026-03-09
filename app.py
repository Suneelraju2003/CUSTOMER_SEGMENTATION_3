import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -----------------------------
# Load Model and Dataset
# -----------------------------

model = joblib.load("customer_segmentation_model.pkl")
df = pd.read_csv("final_customer_dataset.csv")

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# -----------------------------
# Sidebar Navigation
# -----------------------------

st.sidebar.title("Customer Segmentation Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Dataset & Model Analysis","Prediction Tool","Visualization Dashboard"]
)

# Cluster color scheme
cluster_colors = {0:"red",1:"green",2:"blue"}

# --------------------------------------------------
# PAGE 1 : DATASET + MODEL ANALYSIS
# --------------------------------------------------

if page == "Dataset & Model Analysis":

    st.title("Customer Dataset & Model Analysis")

    # KPI METRICS
    st.header("Business Overview")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Total Customers",len(df))
    col2.metric("Average Income",int(df["Income"].mean()))
    col3.metric("Average Spending",int(df["TotalSpent"].mean()))
    col4.metric("Average Purchases",int(df["TotalPurchases"].mean()))

    st.divider()

    # DATASET PREVIEW
    st.header("Dataset Preview")
    st.dataframe(df.head())

    st.header("Dataset Statistics")
    st.dataframe(df.describe())

    st.divider()

    # CORRELATION
    st.header("Feature Correlation")

    corr = df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        title="Customer Data Correlation Heatmap"
    )

    st.plotly_chart(fig,use_container_width=True)

    st.divider()

    # CLUSTER DISTRIBUTION
    st.header("Cluster Distribution")

    fig2 = px.histogram(
        df,
        x="cluster",
        color="cluster",
        color_discrete_map=cluster_colors
    )

    st.plotly_chart(fig2,use_container_width=True)

    st.divider()

    # CLUSTER PROFILE
    st.header("Cluster Profile")

    cluster_profile = df.groupby("cluster").mean()

    st.dataframe(cluster_profile)

    st.divider()

    # CLUSTER GUIDE
    st.header("Customer Segment Guide")

    cluster_info = pd.DataFrame({

        "Cluster":[0,1,2],

        "Customer Type":[
        "Least Active Customer",
        "Highly Active Customer",
        "Moderately Active Customer"
        ],

        "Description":[
        "Low engagement and low spending customers",
        "High value loyal customers",
        "Moderate engagement and moderate spending"
        ]

    })

    st.table(cluster_info)

# --------------------------------------------------
# PAGE 2 : PREDICTION TOOL
# --------------------------------------------------

elif page == "Prediction Tool":

    st.title("Customer Segmentation Prediction Tool")

    st.write("Enter customer details to predict their segment.")

    st.subheader("Input Encoding Guide")

    st.markdown("""
**Education Encoding**

PG → **0**  
UG / Lower → **1**

**Marital Status Encoding**

Relationship → **0**  
Single → **1**
""")

    st.divider()

    col1,col2 = st.columns(2)

    with col1:

        education = st.selectbox(
        "Education",
        ["PG","UG"]
        )

        marital = st.selectbox(
        "Marital Status",
        ["Relationship","Single"]
        )

        income = st.number_input(
        "Income",
        1000,200000,50000
        )

        recency = st.slider(
        "Recency (days since last purchase)",
        0,100,30
        )

        age = st.slider(
        "Age",
        18,80,35
        )

    with col2:

        children = st.slider(
        "Children",
        0,5,1
        )

        total_spent = st.number_input(
        "Total Spent",
        0,3000,500
        )

        campaign = st.slider(
        "Campaign Accepted",
        0,6,1
        )

        purchases = st.slider(
        "Total Purchases",
        0,30,10
        )

    # Encoding based on notebook
    edu_map = {"PG":0,"UG":1}
    mar_map = {"Relationship":0,"Single":1}

    edu = edu_map.get(education)
    mar = mar_map.get(marital)

    input_data = pd.DataFrame({

        "Education":[edu],
        "Marital_Status":[mar],
        "Income":[income],
        "Recency":[recency],
        "Age":[age],
        "Children":[children],
        "TotalSpent":[total_spent],
        "TotalCampaignAccepted":[campaign],
        "TotalPurchases":[purchases]

    })

    if st.button("Predict Customer Segment"):

        cluster = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        st.success(f"Predicted Cluster: {cluster}")

        # Cluster interpretation
        if cluster == 1:

            st.success("Customer Type: Highly Active Customer")

        elif cluster == 2:

            st.info("Customer Type: Moderately Active Customer")

        else:

            st.warning("Customer Type: Least Active Customer")

        st.divider()

        st.subheader("Future Visit Prediction (Recency Based)")

        if recency <= 30:

            st.success("Customer is likely to return soon.")

        elif recency <= 60:

            st.info("Customer has moderate probability of returning.")

        else:

            st.warning("Customer may not return soon.")

# --------------------------------------------------
# PAGE 3 : VISUALIZATION DASHBOARD
# --------------------------------------------------

elif page == "Visualization Dashboard":

    st.title("Customer Behaviour Visualization Dashboard")

    st.header("Customer Overview")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Customers",len(df))
    col2.metric("Avg Income",int(df["Income"].mean()))
    col3.metric("Avg Spending",int(df["TotalSpent"].mean()))
    col4.metric("Avg Purchases",int(df["TotalPurchases"].mean()))

    st.divider()

    col1,col2 = st.columns(2)

    with col1:

        fig1 = px.histogram(df,x="Age",title="Age Distribution")

        st.plotly_chart(fig1,use_container_width=True)

    with col2:

        fig2 = px.histogram(df,x="Income",title="Income Distribution")

        st.plotly_chart(fig2,use_container_width=True)

    col3,col4 = st.columns(2)

    with col3:

        fig3 = px.histogram(df,x="TotalSpent",title="Total Spending Distribution")

        st.plotly_chart(fig3,use_container_width=True)

    with col4:

        fig4 = px.histogram(df,x="Recency",title="Recency Distribution")

        st.plotly_chart(fig4,use_container_width=True)

    st.divider()

    st.header("Cluster Visualizations")

    col5,col6 = st.columns(2)

    with col5:

        fig5 = px.scatter(
        df,
        x="Income",
        y="TotalSpent",
        color="cluster",
        color_discrete_map=cluster_colors
        )

        st.plotly_chart(fig5,use_container_width=True)

    with col6:

        fig6 = px.scatter(
        df,
        x="Age",
        y="TotalSpent",
        color="cluster",
        color_discrete_map=cluster_colors
        )

        st.plotly_chart(fig6,use_container_width=True)

    st.divider()

    st.header("3D Customer Segmentation")

    fig3d = px.scatter_3d(
        df,
        x="Income",
        y="TotalSpent",
        z="Age",
        color="cluster",
        color_discrete_map=cluster_colors
    )

    st.plotly_chart(fig3d,use_container_width=True)

    st.divider()

    st.header("Cluster Distribution")

    fig7 = px.histogram(
        df,
        x="cluster",
        color="cluster",
        color_discrete_map=cluster_colors
    )

    st.plotly_chart(fig7,use_container_width=True)
