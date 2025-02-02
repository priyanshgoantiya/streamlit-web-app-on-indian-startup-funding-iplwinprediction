import streamlit as st
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np

st.set_page_config(layout='wide', page_title="Startup Analysis")

# Load data
df = pd.read_csv('data/startup_funding (2).csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Investors'] = df['Investors'].astype(str)
df.dropna(subset=['Round'], inplace=True)
df.drop_duplicates(['Startup', 'Date'])

# Split the Investors column by comma and flatten the list
split_investors = df['Investors'].str.split(',')
flat_investors = chain.from_iterable(split_investors)
investors_list = [investor.strip() for investor in flat_investors]
unique_investors = sorted(set(investors_list))

def load_Overall_analysis():
    st.title("Overall Analysis")

    # Total Invested Amount
    total = round(df['amount'].sum())
    # Max Funding Startup
    max_Funding_Startup = df.groupby("Startup")['amount'].max().sort_values(ascending=False).head(1).values[0]
    # Avg Funding Amount
    Avg_funding_amount = df.groupby("Startup")['amount'].sum().mean()
    # TOTAL FUNDED STARTUP
    TOTAL_FUNDED_STARTUP = df['Startup'].nunique()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Invested Amount", f"{total:.2f} CR")
    with col2:
        st.metric("Max Funding Startup", f"{max_Funding_Startup:.2f} CR")
    with col3:
        st.metric("AVG Funding Amount", f"{Avg_funding_amount:.2f} CR")
    with col4:
        st.metric("TOTAL FUNDED STARTUP", TOTAL_FUNDED_STARTUP)

    st.subheader("MOM Graph")
    selected_option = st.selectbox('Select Type', ['Total', 'Count'])
    if selected_option == 'Total':
        temp_df = df.groupby(['Year', 'Month'])['amount'].sum().reset_index()
    else:
        temp_df = df.groupby(['Year', 'Month'])['amount'].count().reset_index()

    st.write("DataFrame after groupby operation:")
    st.write(temp_df)

    temp_df.columns = ['Year', 'Month', 'amount']
    temp_df['x_axis'] = temp_df['Month'].astype(str) + '-' + temp_df['Year'].astype(str)
    fig, ax = plt.subplots()
    ax.plot(temp_df['x_axis'], temp_df['amount'], marker='o')
    ax.set_xlabel("Month and Respective Year", fontsize=12)
    ax.set_ylabel("Amount (IN crore)", fontsize=12)
    ax.set_title('MOM Graph Investment Amount')
    step = max(1, len(temp_df) // 20)  # Every third month to avoid mismatch of 'x axis'
    ax.set_xticks(range(0, len(temp_df), step))
    ax.set_xticklabels(temp_df['x_axis'][::step], rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    st.pyplot(fig)
    plt.close()

    # Sector Graph
    st.subheader("Sector Wise Investment")
    selected_option1 = st.selectbox('Select Type', ['Total', 'Count'])
    if selected_option1 == 'Total':
        temp_df1 = df.groupby('Vertical')['amount'].sum().reset_index().sort_values(by='amount', ascending=False).head(20)
    else:
        temp_df1 = df.groupby('Vertical')['amount'].count().reset_index().sort_values(by='amount', ascending=False).head(20)

    st.write("DataFrame after groupby operation:")
    st.write(temp_df1)
    fig1, ax1 = plt.subplots(figsize=(12, 18))

    bars = ax1.bar(temp_df1['Vertical'], temp_df1['amount'], color='skyblue')
    ax1.set_xlabel('Vertical')
    ax1.set_ylabel('Amount')
    plt.xticks(rotation=45, ha='right')
    ax1.set_title('Investment Distribution by Sector')
    total_amount = temp_df1['amount'].sum()

    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_amount) * 100
        ax1.annotate(f'{percentage:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height))

    st.pyplot(fig1)
    plt.close()

    # Funding Heatmap
    heatmap_data = df.groupby(['Vertical', 'Round'])['amount'].sum().unstack(fill_value=0)
    top_verticals = heatmap_data.sum(axis=1).nlargest(20).index
    heatmap_data = heatmap_data.loc[top_verticals]

    heatmap_data_log = np.log1p(heatmap_data)

    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', linewidths=0.5)
    plt.title('Funding Amount Heatmap by Vertical and Round')
    plt.ylabel('Vertical')
    plt.xlabel('Round')
    st.subheader("Funding Heatmap")
    st.pyplot(plt)
    plt.close()


def find_similar_investors(selected_investor):
    categorical_features = ["City", "Round", "Vertical"]
    numerical_features = ["amount"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(), categorical_features)
        ]
    )

    X = preprocessor.fit_transform(df)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    st.title('Investor Clustering and Similar Investors')

    st.subheader(f"Selected Investor: {selected_investor}")
    investor_data = df[df['Investors'].str.contains(selected_investor)]
    st.write(investor_data)

    if not investor_data.empty:
        cluster = investor_data['Cluster'].iloc[0]
        similar_investors = df[df['Cluster'] == cluster]['Investors'].tolist()

        st.subheader(f"Investors similar to {selected_investor}:")
        for inv in similar_investors:
            st.write(inv)

        num_similar_investors = len(similar_investors)
        st.write(f"Number of investors similar to {selected_investor}: {num_similar_investors}")
    else:
        st.warning("No data available for the selected investor.")
