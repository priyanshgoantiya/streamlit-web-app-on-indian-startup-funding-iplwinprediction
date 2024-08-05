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
df = pd.read_csv('startup_funding (2).csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year
df['Month']=df['Date'].dt.month
df['Investors'] = df['Investors'].astype(str)
df.dropna(subset=['Round'],inplace=True)
df.drop_duplicates(['Startup', 'Date'])

# Split the Investors column by comma and flatten the list
split_investors = df['Investors'].str.split(',')
flat_investors = chain.from_iterable(split_investors)
investors_list = [investor.strip() for investor in flat_investors]
unique_investors = sorted(set(investors_list))

def load_Overall_analysis():
    st.title("overall Analysis")

    # Total Invested Amount
    total= round(df['amount'].sum())
    # max Funding Startup
    max_Funding_Startup=df.groupby("Startup")['amount'].max().sort_values(ascending=False).head(1).values[0]
    # Avg Funding Amount
    Avg_funding_amount = df.groupby("Startup")['amount'].sum().mean()
    # TOTAL FUNDED STARTUP
    TOTAL_FUNDED_STARTUP=df['Startup'].nunique()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Invested Amount", f"{total:.2f} CR")

    with col2:
        st.metric("Max Funding Startup", f"{max_Funding_Startup:.2f} CR")

    with col3:
        st.metric("AVG Funding Amount", f"{Avg_funding_amount:.2f} CR")
    with col4:
        st.metric("TOTAL FUNDED STARTUP",TOTAL_FUNDED_STARTUP)

    st.subheader("MOM Graph")
    selected_option=st.selectbox('Select Type',['Total','Count'])
    if selected_option=='Total':
        temp_df = df.groupby(['Year', 'Month'])['amount'].sum().reset_index()
    else:
        temp_df = df.groupby(['Year', 'Month'])['amount'].count().reset_index()
    # Debug: show the temp_df DataFrame
    st.write("DataFrame after groupby operation:")
    st.write(temp_df)

    # Ensure consistent column names
    temp_df.columns = ['Year', 'Month', 'amount']
    temp_df['x_axis'] = temp_df['Month'].astype(str) + '-' + temp_df['Year'].astype(str)
    fig, ax = plt.subplots()
    ax.plot(temp_df['x_axis'],temp_df['amount'], marker='o')
    ax.set_xlabel("month and respective year", fontsize=12)
    ax.set_ylabel("Amount (IN crore)", fontsize=12)
    ax.set_title('MOM graph investment amount ')
    step = max(1, len(temp_df) // 20) # Every third month to avoid mismatch of 'x axis'
    ax.set_xticks(range(0,len(temp_df),step))
    ax.set_xticklabels(temp_df['x_axis'][::step], rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    st.pyplot(fig)
    plt.close()

    # Sector Graph
    st.subheader("Sector Wise Investment")
    selected_option1= st.selectbox('Select Type', ['total', 'count'])
    if selected_option1 == 'total':
        temp_df1 = df.groupby('Vertical')['amount'].sum().reset_index().sort_values(by='amount', ascending=False).head(20)

    else:
        temp_df1 = df.groupby('Vertical')['amount'].count().reset_index().sort_values(by='amount', ascending=False).head(20)

    # Debug: show the temp_df DataFrame
    st.write("DataFrame after groupby operation:")
    st.write(temp_df1)
    fig1, ax1 = plt.subplots(figsize=(12,18))

    st.subheader("Sector Graph")
    bars=ax1.bar(temp_df1['Vertical'], temp_df1['amount'], color='skyblue')
    ax1.set_xlabel('Vertical')
    ax1.set_ylabel('Amount')
    plt.xticks(rotation=45, ha='right')
    ax1.set_title('Investment Distribution by Sector')
    total_amount = temp_df1['amount'].sum()

    # Annotating bars with the percentage and labels
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_amount) * 100
        ax1.annotate(f'{percentage:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height))
    st.pyplot(fig1)
    plt.close()
    # Type Of Funding

    st.subheader("Type Of Funding")
    unique_rounds = df['Round'].unique().tolist()
    st.write(unique_rounds)

    # City Wise Funding
    st.subheader("City Wise Funding ")
    selected_option2= st.selectbox('Select Type', ['Total Investment in respective City(IN cr)', 'count of Investment in respective City'])
    if selected_option2 == 'Total Investment in respective City(IN cr)':
        temp_df3 = df.groupby('City')['amount'].sum().reset_index().sort_values(by='amount', ascending=False).head(20)

    else:
        temp_df3 = df.groupby('City')['amount'].count().reset_index().sort_values(by='amount', ascending=False).head(20)

    st.write("DataFrame after groupby operation:")
    st.write(temp_df3)
    st.subheader("City Wise Funding Graph")
    fig2, ax2 = plt.subplots(figsize=(12, 18))
    bars1 = ax2.bar(temp_df3['City'], temp_df3['amount'], color='skyblue')
    ax2.set_xlabel('City')
    ax2.set_ylabel('Amount')
    plt.xticks(rotation=45, ha='right')
    total_amount = temp_df3['amount'].sum()

    # Annotating bars with the percentage and labels
    for bar in bars1:
        height = bar.get_height()
        percentage = (height / total_amount) * 100
        ax2.annotate(f'{percentage:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height))
    st.pyplot(fig2)
    plt.close()

    # Top Startup Year Wise
    st.subheader("Top Startups")
    selected_option3 = st.selectbox('Select Type', ['Top Startup in respective Year(IN cr)',
                                                    'overall Top startup'])
    if selected_option3 == 'Top Startup in respective Year(IN cr)':
        temp_df4 = df.groupby(['Year', 'Startup'])['amount'].sum().reset_index(
        ).sort_values(by=['Year', 'amount'], ascending=[True, False]).drop_duplicates('Year', keep='first')
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
    plt.figure(figsize=(12, 10))
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
    last5_df = df[df['Investors'].str.contains(investor)].sort_values('Date', ascending=False).head(5)[
        ['Date', 'Startup', 'Vertical', 'City', 'Round', 'amount']]
    st.subheader('Most recent investments')
    st.dataframe(last5_df)

    col1, col2, col3, col4 = st.columns(4)

    # Biggest Investments
    with col1:
        biggest_investment = df[df['Investors'].str.contains(investor)].groupby('Startup')['amount'].sum().sort_values(
            ascending=False).head(20)
        st.subheader("Biggest Investments")
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.bar(biggest_investment.index, biggest_investment.values)
        ax.set_xlabel("Startup", fontsize=12)
        ax.set_ylabel("Amount (IN crore)", fontsize=12)
        ax.set_title(f'Biggest Investments by {investor}')
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=12)
        st.pyplot(fig)
        plt.close()

    # Sector Invested In
    with col2:
        investment_by_vertical = df[df['Investors'].str.contains(investor)].groupby('Vertical')['amount'].sum().head(10)
        fig1, ax1 = plt.subplots()
        st.subheader("Sectors Invested In")
        ax1.pie(investment_by_vertical, labels=investment_by_vertical.index, autopct='%1.1f%%', pctdistance=0.75)
        ax1.set_title(f'Investments by {investor} by Vertical')
        st.pyplot(fig1)
        plt.close()

    # Round Invested in
    with col3:
        investment_by_round = df[df['Investors'].str.contains(investor)].groupby('Round')['amount'].sum()
        fig2, ax2 = plt.subplots()
        st.subheader("Rounds Invested in")
        ax2.pie(investment_by_round, labels=investment_by_round.index, autopct='%1.1f%%', pctdistance=0.85)
        ax2.set_title(f'Investments by {investor} by Round')
        st.pyplot(fig2)
        plt.close()

    # City With Most Investment
    with col4:
        investment_by_city = df[df['Investors'].str.contains(investor)].groupby('City')['amount'].sum()
        fig3, ax3 = plt.subplots()
        st.subheader("City With Most Investments")
        ax3.pie(investment_by_city, labels=investment_by_city.index, autopct='%1.1f%%', pctdistance=0.85)
        ax3.set_title(f'Investments by {investor} by City')
        st.pyplot(fig3)
        plt.close()

    # Yearly investment analysis

    yearly_investment = df[df['Investors'].str.contains(investor)].groupby('Year')['amount'].sum()
    fig4, ax4 = plt.subplots()
    st.subheader(f"Yearly Investment Trend for {investor}")
    ax4.plot(yearly_investment.index, yearly_investment.values, marker='o')
    st.pyplot(fig4)
    plt.close()

    # Similar investors
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


st.sidebar.title("Startup Funding Analysis")


def Startup_analysis(Startup):
    st.title(Startup)
    # Which Industry Startup belong
    st.subheader("Industry")
    Industry=df[df['Startup']==Startup]['Vertical'].values[0]
    st.write(Industry)
    # Which sub-Industry Startup belong
    st.subheader("Sub-Industry")
    Sub_Industry = df[df['Startup'] == Startup]['SubVertical'].values[0]
    st.write(Sub_Industry)
    # Startup Belong to which City
    st.subheader("Location")
    Location = df[df['Startup'] == Startup]['City'].values[0]
    st.write(Location)
    # Funding Rounds
    st.subheader('Funding Round')
    Stage = df[df['Startup'] == Startup]['Round'].values[0]
    st.write(Stage)
    # Startup Investors
    st.subheader('Investors')
    Investors = df[df['Startup'] == Startup]['Investors'].values[0]
    st.write(Investors)
    # Date of Opening
    st.subheader('Date of Opening')
    Date = df[df['Startup'] == Startup]['Date'].values[0]
    st.write(Date)
    # Startup Investment amount
    st.subheader('Startup Investment amount')
    amount = df[df['Startup'] == Startup]['amount'].values[0]
    st.write(amount)
# Similar Startup
def find_similar_Startup(selected_startup):
    categorical_features = ["City", "Round", "Vertical", "Startup"]  # Add "Startup" to categories
    numerical_features = ["amount"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Handle unknown startups
        ]
    )

    X = preprocessor.fit_transform(df)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    st.title('Startup Clustering and Similar Investors')

    st.subheader(f"Selected Startup: {selected_startup}")
    startup_data = df[df['Startup'] == selected_startup]  # Filter by exact startup name match
    st.write(startup_data)

    if not startup_data.empty:
        cluster = startup_data['Cluster'].iloc[0]
        similar_startups = df[df['Cluster'] == cluster]['Startup'].tolist()  # Find startups in the same cluster

        st.subheader(f"Startups similar to {selected_startup}:")
        for startup in similar_startups:
            st.write(startup)

        num_similar_startups = len(similar_startups)
        st.write(f"Number of startups similar to {selected_startup}: {num_similar_startups}")
    else:
        st.warning("No data available for the selected startup.")



option = st.sidebar.selectbox('Select One', ['Overall Analysis', 'Startup Analysis', 'Investor Analysis'])
if option == 'Overall Analysis':
    load_Overall_analysis()

elif option == 'Startup Analysis':
    selected_startup = st.sidebar.selectbox('Select Startup', sorted(df['Startup'].unique().tolist()))
    btn1 = st.sidebar.button("Find Startup details")
    if btn1:
        Startup_analysis(selected_startup)
        find_similar_Startup(selected_startup)


else:
    selected_investor = st.sidebar.selectbox('Select Investor', unique_investors)
    btn2 = st.sidebar.button("Find Investor details")
    if btn2:
        load_investor_details(selected_investor)
        find_similar_investors(selected_investor)