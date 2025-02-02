import streamlit as st
import pandas as pd
from itertools import chain
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
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
    # max Funding Startup
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

    # Debug: show the temp_df DataFrame
    st.write("DataFrame after groupby operation:")
    st.write(temp_df)

    # Ensure consistent column names
    temp_df.columns = ['Year', 'Month', 'amount']
    temp_df['x_axis'] = temp_df['Month'].astype(str) + '-' + temp_df['Year'].astype(str)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=temp_df['x_axis'], y=temp_df['amount'], marker='o', ax=ax)
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
    if selected_option1 == 'total':
        temp_df1 = df.groupby('Vertical')['amount'].sum().reset_index().sort_values(by='amount', ascending=False).head(20)
    else:
        temp_df1 = df.groupby('Vertical')['amount'].count().reset_index().sort_values(by='amount', ascending=False).head(20)

    # Debug: show the temp_df DataFrame
    st.write("DataFrame after groupby operation:")
    st.write(temp_df1)
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    st.subheader("Sector Graph")
    sns.barplot(x=temp_df1['Vertical'], y=temp_df1['amount'], ax=ax1, color='skyblue')
    ax1.set_xlabel('Vertical')
    ax1.set_ylabel('Amount')
    plt.xticks(rotation=45, ha='right')
    ax1.set_title('Investment Distribution by Sector')
    total_amount = temp_df1['amount'].sum()

    # Annotating bars with the percentage and labels
    for p in ax1.patches:
        height = p.get_height()
        percentage = (height / total_amount) * 100
        ax1.annotate(f'{percentage:.1f}%', (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom')
    st.pyplot(fig1)
    plt.close()

    # City Wise Funding
    st.subheader("City Wise Funding")
    selected_option2 = st.selectbox('Select Type', ['Total Investment in respective City(IN cr)', 'Count of Investment in respective City'])
    if selected_option2 == 'Total Investment in respective City(IN cr)':
        temp_df3 = df.groupby('City')['amount'].sum().reset_index().sort_values(by='amount', ascending=False).head(20)
    else:
        temp_df3 = df.groupby('City')['amount'].count().reset_index().sort_values(by='amount', ascending=False).head(20)

    st.write("DataFrame after groupby operation:")
    st.write(temp_df3)
    st.subheader("City Wise Funding Graph")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.barplot(x=temp_df3['City'], y=temp_df3['amount'], ax=ax2, color='skyblue')
    ax2.set_xlabel('City')
    ax2.set_ylabel('Amount')
    plt.xticks(rotation=45, ha='right')
    total_amount = temp_df3['amount'].sum()

    # Annotating bars with the percentage and labels
    for p in ax2.patches:
        height = p.get_height()
        percentage = (height / total_amount) * 100
        ax2.annotate(f'{percentage:.1f}%', (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom')
    st.pyplot(fig2)
    plt.close()

    # Top Startup Year Wise
    st.subheader("Top Startups")
    selected_option3 = st.selectbox('Select Type', ['Top Startup in respective Year(IN cr)', 'Overall Top startup'])
    if selected_option3 == 'Top Startup in respective Year(IN cr)':
        temp_df4 = df.groupby(['Year', 'Startup'])['amount'].sum().reset_index().sort_values(by=['Year', 'amount'], ascending=[True, False]).drop_duplicates('Year', keep='first')
    else:
        temp_df4 = df.groupby(['Startup'])['amount'].sum().reset_index().sort_values(by='amount', ascending=False)

    st.write("DataFrame after groupby operation:")
    st.write(temp_df4)

    # Top 100 investors
    st.subheader("Top Investors")
    temp_df5 = df.groupby('Investors')['amount'].sum().reset_index().sort_values(by='amount', ascending=False).head(100)
    st.write("DataFrame after groupby operation:")
    st.write(temp_df5)

    # Funding Heatmap
    heatmap_data = df.groupby(['Vertical', 'Round'])['amount'].sum().unstack(fill_value=0)
    top_verticals = heatmap_data.sum(axis=1).nlargest(20).index
    heatmap_data = heatmap_data.loc[top_verticals]

    # Apply log transformation
    heatmap_data_log = np.log1p(heatmap_data)

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', linewidths=0.5)
    plt.title('Funding Amount Heatmap by Vertical and Round')
    plt.ylabel('Vertical')
    plt.xlabel('Round')

    # Display plot in Streamlit
    st.subheader("Funding Heatmap")
    st.pyplot(plt)
    plt.close()

def load_investor_details(investor):
    st.title(investor)

    # Last 5 investments of the investor
    last5_df = df[df['Investors'].str.contains(investor)].sort_values('Date', ascending=False).head(5)[['Date', 'Startup', 'Vertical', 'City', 'Round', 'amount']]
    st.subheader('Most Recent Investments')
    st.dataframe(last5_df)

    col1, col2, col3, col4 = st.columns(4)

    # Biggest Investments
    with col1:
        biggest_investment = df[df['Investors'].str.contains(investor)].groupby('Startup')['amount'].sum().sort_values(ascending=False).head(20)
        st.subheader("Biggest Investments")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=biggest_investment.index, y=biggest_investment.values, ax=ax, color='skyblue')
        ax.set_xlabel("Startup", fontsize=12)
        ax.set_ylabel("Amount", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        ax.set_title('Top 20 Biggest Investments by Investor')
        st.pyplot(fig)
        plt.close()

    # Total Amount Invested
    with col2:
        total_investment = df[df['Investors'].str.contains(investor)]['amount'].sum()
        st.subheader("Total Amount Invested")
        st.write(f"₹ {total_investment:.2f} CR")

    # Average Investment
    with col3:
        avg_investment = df[df['Investors'].str.contains(investor)]['amount'].mean()
        st.subheader("Average Investment")
        st.write(f"₹ {avg_investment:.2f} CR")

    # Total Number of Investments
    with col4:
        num_investments = df[df['Investors'].str.contains(investor)]['amount'].count()
        st.subheader("Total Number of Investments")
        st.write(f"{num_investments}")

def app():
    menu = ['Home', 'Investor']
    choice = st.sidebar.selectbox("Select Option", menu)
    if choice == 'Home':
        load_Overall_analysis()
    else:
        investor = st.sidebar.selectbox('Select Investor', unique_investors)
        load_investor_details(investor)


