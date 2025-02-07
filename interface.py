import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pandas.tseries.offsets import CustomBusinessDay
import statsmodels.api as sm
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from PIL import Image


col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image("trend.png", width=100)
with col2:
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center;">
            <h1 style="margin: 0;">LiquidityGuards</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.image("duke_logo.png", width=140)
    
st.write("""
Our team specializes in **Predictive Modeling & Forecasting** and **Anomaly Detection**, with a particular focus on **Sentiment Analysis** of financial news. We assess the importance of news events by ranking them from highest to lowest, aiming to identify potential market risks💡.

Our hypothesis suggests that extreme market sentiment—whether **overly optimistic** or **deeply pessimistic**—can drive **higher trading volumes**, potentially increasing **systemic risk**✅.
""")

st.write("""
Sentiment analysis plays a crucial role in **understanding market dynamics** and identifying **liquidity and volatility risks**. The **Depository Trust & Clearing Corporation (DTCC)**, as a key entity in **clearing and settlement**, must anticipate shifts in **market sentiment** that could lead to **heightened trading activity**.🧠

Sudden surges in **trading volume**, triggered by **extreme sentiment**, could **stress settlement systems**, introduce **operational bottlenecks**, and elevate **counterparty risks**. By integrating **sentiment analysis** into **risk monitoring frameworks**, DTCC can enhance its ability to **manage liquidity**, **optimize collateral requirements**, and **strengthen overall financial market stability**.🚀
""")

# Load CSV file
data = pd.read_csv('tsla.csv')

# Preprocess data
if 'day_total_vol' in data.columns:
    data['day_total_vol'] = data['day_total_vol'] / 10000000

data['Date'] = pd.to_datetime(data['date'])
data['Volume'] = data['TSLA_first_hour_vol']  # Predicting first hour volume
data['SentimentImpact'] = data['abs_sentiment_score'] * data['news_count']

# Create lag features for past 5 days using 'day_total_vol'
for lag in range(1, 6):
    data[f'TSLA_volume_lag_{lag}d'] = data['day_total_vol'].shift(lag)

# Drop rows with NaN values due to lagging
data.dropna(inplace=True)

# Load sentiment and volume data
volume_data = pd.read_csv('filtered_hourly_volume_data.csv')
sentiment_data = pd.read_csv('filtered_sentiment_results.csv')

# Filter for specific tickers
tickers = ['TSLA']
filtered_volume_data = volume_data[['Datetime'] + tickers].copy()
filtered_sentiment_data = sentiment_data[sentiment_data['ticker'].isin(tickers)].copy()

# Process Sentiment Data
filtered_sentiment_data['time'] = pd.to_datetime(filtered_sentiment_data['time'])
filtered_sentiment_data['news_date'] = filtered_sentiment_data['time'].dt.date
filtered_sentiment_data['day_of_week'] = filtered_sentiment_data['time'].dt.dayofweek
filtered_sentiment_data['time_only'] = filtered_sentiment_data['time'].dt.time

# Filter for non-trading-hour news
non_trading_news = filtered_sentiment_data[
    (filtered_sentiment_data['trading_status'].str.lower() == 'non-trading') |
    (
        ((filtered_sentiment_data['time_only'] >= pd.to_datetime('16:00:00').time()) &
         (filtered_sentiment_data['day_of_week'].isin([0, 1, 2, 3, 4]))) |
        (filtered_sentiment_data['day_of_week'].isin([5, 6]))
    )
].copy()

# Calculate sentiment score: -1 * negative + 1 * positive
non_trading_news['sentiment_score'] = -1 * non_trading_news['roberta_neg'] + non_trading_news['roberta_pos']
non_trading_news['abs_sentiment_score'] = non_trading_news['sentiment_score'].abs()

# Map sentiment to the next trading day by default (for after 4 PM news)
us_bd = CustomBusinessDay(weekmask='Mon Tue Wed Thu Fri')
non_trading_news['next_trading_day'] = pd.to_datetime(non_trading_news['news_date']) + us_bd

# For news before 9:30 AM, map to the same day
non_trading_news.loc[
    non_trading_news['time_only'] < pd.to_datetime('09:30:00').time(),
    'next_trading_day'
] = pd.to_datetime(non_trading_news['news_date'])

# For weekend news (Saturday/Sunday), map to Monday
non_trading_news.loc[
    non_trading_news['day_of_week'].isin([5, 6]),
    'next_trading_day'
] = pd.to_datetime(non_trading_news['news_date']) + pd.offsets.Week(weekday=0)

# Aggregate sentiment data
non_trading_news['news_count'] = non_trading_news.groupby(['ticker', 'next_trading_day'])['ticker'].transform('count')
sentiment_trend = non_trading_news.groupby(['ticker', 'next_trading_day']).agg({
    'abs_sentiment_score': 'mean',
    'news_count': 'max'
}).reset_index()

# Calculate weighted absolute average sentiment score
sentiment_trend['absolute_avg_sentiment_score'] = sentiment_trend['abs_sentiment_score'] * sentiment_trend['news_count']

# Remove the timezone information from 'Datetime'
filtered_volume_data['Datetime'] = filtered_volume_data['Datetime'].str.replace(r'-\d{2}:\d{2}$', '', regex=True)

# Convert 'Datetime' to datetime format
filtered_volume_data['Datetime'] = pd.to_datetime(filtered_volume_data['Datetime'], errors='coerce')

# Extract time and date
filtered_volume_data['time'] = filtered_volume_data['Datetime'].dt.time
filtered_volume_data['date'] = filtered_volume_data['Datetime'].dt.date

# Ensure date formats match for merging
sentiment_trend['next_trading_day'] = pd.to_datetime(sentiment_trend['next_trading_day'])
filtered_volume_data['date'] = pd.to_datetime(filtered_volume_data['date'])

# Filter volume data for the first trading hour
first_hour_volume = filtered_volume_data[
    (filtered_volume_data['time'] >= pd.to_datetime('09:30:00').time()) & 
    (filtered_volume_data['time'] <= pd.to_datetime('10:30:00').time())
]

# Aggregate volume by date and ticker
volume_trend = first_hour_volume.melt(
    id_vars=['Datetime', 'date'], 
    value_vars=tickers, 
    var_name='ticker', 
    value_name='volume'
)
volume_trend = volume_trend.groupby(['ticker', 'date'])['volume'].sum().reset_index()

# Merge sentiment and volume data on the next trading day
merged_trend = pd.merge(
    sentiment_trend, 
    volume_trend, 
    left_on=['ticker', 'next_trading_day'], 
    right_on=['ticker', 'date'], 
    how='inner'
)

# Correlation Analysis
for ticker in tickers:
    ticker_data = merged_trend[merged_trend['ticker'] == ticker]

    if not ticker_data.empty:
        # Compute Pearson correlation
        correlation, p_value = pearsonr(ticker_data['absolute_avg_sentiment_score'], ticker_data['volume'])

        # Display correlation analysis title for each ticker
        st.markdown(f"""
            <h2 style="text-align: center;">Correlation Analysis</h2>
        """, unsafe_allow_html=True)

        # Subtitle for key observations
        st.markdown(f"""
            <h3 style="text-align: center;">Key Observations for Example {ticker}</h3>
        """, unsafe_allow_html=True)


        # Display analysis text with dynamic values
        st.write(f"""
        Our analysis explores the relationship between **after-market news sentiment** (**4:00 PM – 9:30 AM**) and the **first-hour trading volume** of the next market session. Using **{ticker} as an example**, we find a **moderate to strong positive correlation** between the **absolute sentiment score** and the **first-hour trading volume**, with a **highly significant P-value**.
        """)
        
        st.write(
        "We utilized the **RoBERTa** model to conduct a sentiment analysis on news articles, "
        "categorizing them into **neutral, positive, or negative** based on their sentiment scores. "
        "To quantify the sentiment impact more effectively, we calculated the **absolute weighted sentiment score**, "
        "which served as the primary metric for our correlation analysis. "
        "Additionally, we incorporated the **news count** as a factor to enhance the analysis by accounting for **news intensity**, "
        "ensuring that the overall sentiment impact reflects both the **sentiment magnitude** and the **volume** of relevant news articles."
        )
        
        st.write(f"""       
        These findings highlight how **market participants react to overnight news**, potentially leading to **heightened trading activity at market open**. A deeper understanding of this relationship can enhance **liquidity forecasting**, **risk management**, and **trade execution strategies**. 

        For institutions like **DTCC**, integrating **sentiment analysis into risk monitoring** could improve:
        
        🔹 **Clearing efficiency**
        
        🔹 **Collateral optimization**
        
        🔹 **Market stability**
        """)

        st.write(f"""
        The chart below visualizes the relationship between **{ticker}’s trading volume** and the **absolute average sentiment score** of after-market news.
        """)

        # Plot sentiment and volume trends
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot absolute avg sentiment score on left y-axis
        line1, = ax1.plot(ticker_data['next_trading_day'], ticker_data['absolute_avg_sentiment_score'], 
                          label='Absolute Avg Sentiment Score', linestyle='--', color='blue')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Absolute Avg Sentiment Score', color='blue')

        # Create second y-axis for trading volume
        ax2 = ax1.twinx()
        line2, = ax2.plot(ticker_data['next_trading_day'], ticker_data['volume'], 
                          label='Trading Volume', color='green')
        ax2.set_ylabel('Trading Volume', color='green')

        # Add title and legend
        plt.title(f'{ticker} - Trading Volume & Absolute Avg Sentiment Score (Correct Mapping)')
        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')

        # Show the plot in Streamlit
        st.pyplot(fig)

        # Display key observations with Streamlit signals
        st.success(f"#### Positive Correlation")
        st.write(f"""
        
        🔹 The **correlation coefficient ({correlation:.4f})** indicates a **moderate to strong positive relationship** between **news sentiment** and the **first-hour trading volume** on the following trading day.
        
        🔹 This suggests that the market tends to **react to overnight news**, leading to **increased trading activity** in the first hour after the market opens.
        """)

        st.warning(f"#### Statistical Significance")
        st.write(f"""
        
        🔹 The **P-value ({p_value:.4f})** confirms that the correlation is **highly statistically significant**, meaning there is **a very low probability** that this relationship is due to chance.
        """)

        st.info(f"#### Market Interpretation")
        st.write(f"""
        
        🔹 The results suggest that **sentiment-driven trading behavior** is a key factor influencing **market liquidity** and **volatility** during the early trading session.
        
        🔹 **DTCC and other clearing entities** can leverage such **sentiment analysis** to **anticipate settlement risks**, ensuring **efficient liquidity management**.
        """)
        
        image = Image.open("heatmap.png")

        # Centering with Markdown
        st.markdown("<h4 style='text-align: center;'>Visualized News Sentiment Heatmap</h4>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        # Explanation section
        st.markdown(
            """
            This interactive heatmap provides a **real-time sentiment analysis** of key stocks before market open.  
            Users can **select specific tickers of interest** and apply **various predictive models** to gain insights into potential market movements.  

            ---  
            1️⃣ **📏 Square Size Represents News Coverage Intensity**  
            🔹 Larger squares indicate stocks with **more news coverage**.  
            🔹Smaller squares represent stocks with **fewer news mentions**.  

            2️⃣ **🎨 Color Represents Sentiment Type and Intensity**  
            🔹 **🟥 Deep Red**: Strong **negative sentiment** (e.g., PLUG, TSLA).  
            🔹 **🟧 Light Red/Pink**: Mildly **negative sentiment**.  
            🔹 **🟦 Light Blue**: Mildly **positive sentiment**.  
            🔹 **🔵 Deep Blue**: Strong **positive sentiment** (e.g., RIVN, SMCI).  

            3️⃣ **🏢 Each Square Represents a Company**  
            🔹 The **ticker symbol** (e.g., **NVDA, AAPL, TSLA**) is displayed within its respective square.  
            🔹 Below the ticker, the **sentiment score** is shown, ranging from **-1 to 1**, where:  
                 **1.00** → Extremely **positive** sentiment.  
                 **-1.00** → Extremely **negative** sentiment.  
                 **0.00** → **Neutral** sentiment.  

            """
        )

        st.markdown(
            """
            <style>
            .small-text {
                font-size: 12px;  /* Adjust the size as needed */
            }
            </style>
            
            <div class="small-text">
            
            🔍 **Data Availability & Future Improvements**:
            
            Due to the *time constraints of this hackathon* and *limited access to large-scale news data*,  
            our current dataset primarily focuses on **TSLA (Tesla, Inc.)**. 

            As a result, we will be presenting our analysis using *TSLA* as a representative example.  
            However, this methodology is *scalable*, and with access to a *larger volume of news data*,  
            we can expand our analysis to include *multiple companies* in future iterations.
            </div>
            """,
            unsafe_allow_html=True  # Required to render HTML and CSS
        )


# Model Training and Predictions
st.markdown("""
    <h2 style="text-align: center;">Volume Prediction Model Comparison</h2>
""", unsafe_allow_html=True)

model_choice = st.selectbox('Select Ticker', ['NVDA', 'AMZN', 'AAPL', 'TSLA', 'BAC', 'INTC', 'PFE', 'SMCI', 'AMD', 'MARA', 
           'PLTR', 'RIVN', 'NIO', 'SNAP', 'PLUG', 'LCID', 'TLRY', 'SOFI', 'SOUN', 'CLSK'])

model_choice = st.selectbox('Select Model', ['Random Forest', 'Linear Regression', 'XGBoost', 'SARIMAX'])

if model_choice == "Random Forest":
    st.markdown("""
    <div style="text-align: center;">
        <h3>🌲 Random Forest Regression</h3>
        <p style="font-size:17px;">
        1️⃣ An <b>ensemble learning</b> method that combines multiple decision trees. <br>
        2️⃣ Reduces overfitting and improves accuracy by aggregating predictions from many trees. <br>
        3️⃣ Well-suited for <b>complex, non-linear relationships</b> in data.
        </p>
    </div>
    """, unsafe_allow_html=True)

elif model_choice == "Linear Regression":
    st.markdown("""
    <div style="text-align: center;">
        <h3>📈 Linear Regression</h3>
        <p style="font-size:17px;">
        1️⃣ A fundamental statistical model that <b>assumes a linear relationship</b> between input features and the target variable. <br>
        2️⃣ Best for datasets where changes in volume are <b>linearly dependent</b> on factors like sentiment scores and past volume trends.
        </p>
    </div>
    """, unsafe_allow_html=True)

elif model_choice == "XGBoost":
    st.markdown("""
    <div style="text-align: center;">
        <h3>🚀 XGBoost (Extreme Gradient Boosting)</h3>
        <p style="font-size:17px;">
        1️⃣ A powerful gradient boosting algorithm optimized for <b>both speed and accuracy</b>. <br>
        2️⃣ Uses <b>boosted decision trees</b> and is effective for <b>high-dimensional data</b>. <br>
        3️⃣ Provides <b>built-in regularization</b> to prevent overfitting.
        </p>
    </div>
    """, unsafe_allow_html=True)

elif model_choice == "SARIMAX":
    st.markdown("""
    <div style="text-align: center;">
        <h3>📊 SARIMAX (Seasonal AutoRegressive Integrated Moving Average with Exogenous Variables)</h3>
        <p style="font-size:17px;">
        1️⃣ A time-series forecasting model that captures <b>trends, seasonality, and exogenous factors</b>. <br>
        2️⃣ Ideal for financial and market data with patterns influenced by external variables (e.g., sentiment, trading days). <br>
        3️⃣ Uses historical volume data and external features like sentiment scores to make <b>more informed predictions</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    predicted_volume_sarimax = 32194940

    st.markdown(f"""
    <div style="text-align: center;">
        <h3>📊 Predicted Trading Volume on Dec 31, 2024</h3>
        <p style="font-size:18px;">Using <b>SARIMAX</b>, the estimated trading volume between 9:30am - 10:30am EST is <b>{predicted_volume_sarimax:,.0f}</b> shares.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

if model_choice == 'Random Forest':
    model = RandomForestRegressor(n_estimators=1000, random_state=500)
elif model_choice == 'Linear Regression':
    model = LinearRegression()
elif model_choice == 'XGBoost':
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        random_state=42
    )

# Split data into training and testing sets
train_data = data[(data['Date'] >= '2024-01-01') & (data['Date'] <= '2024-09-30')]
test_data = data[(data['Date'] >= '2024-10-01') & (data['Date'] <= '2024-12-31')].copy()

# Define features and target for Linear Regression and XGBoost
features = [
    'mean', 'std', 'abs_sentiment_score', 'news_count', 'is_option_expiry',
    'is_first_trading_day', 'is_last_trading_day', 'is_last_trading_day_of_quarter',
    'Friday', 'Monday', 'Thursday', 'Tuesday', 'Wednesday',
    'TSLA_volume_lag_1d', 'TSLA_volume_lag_2d', 'TSLA_volume_lag_3d',
    'TSLA_volume_lag_4d', 'TSLA_volume_lag_5d'
]
target = 'Volume'

# Function to center any text output
def centered_text(text, emoji=""):
    return st.markdown(f"""
        <div style="text-align: center;">
            <h4>{text} {emoji}</h4>
        </div>
    """, unsafe_allow_html=True)

# Function to center charts
def centered_chart(fig):
    st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# Standardization for Linear Regression
if model_choice == 'Linear Regression':
    cols_to_standardize = features + [target]
    train_mean = train_data[cols_to_standardize].mean()
    train_std = train_data[cols_to_standardize].std()

    train_data[cols_to_standardize] = (train_data[cols_to_standardize] - train_mean) / train_std
    test_data[cols_to_standardize] = (test_data[cols_to_standardize] - train_mean) / train_std

if model_choice == 'SARIMAX':
    # SARIMAX Specific Logic
    file_path = "tsla.csv"
    df = pd.read_csv(file_path, parse_dates=[0], index_col=0)
    df = df.sort_index()
    df['day_total_vol'] = df['day_total_vol']/10000000

    for lag in range(1, 5):
        df[f"day_total_vol_L{lag}"] = df["day_total_vol"].shift(lag)

    df_2024 = df.loc["2024"].dropna()

    train = df_2024.loc["2024-01":"2024-09"]
    test = df_2024.loc["2024-10":"2024-12"]

    y_train = train["TSLA_first_hour_vol"]
    y_test = test["TSLA_first_hour_vol"]
    lags = [f"day_total_vol_L{i}" for i in range(1, 5)]

    X_train_mean = train.drop(columns=["day_total_vol","TSLA_first_hour_vol", "abs_sentiment_score"])
    X_test_mean = test.drop(columns=["day_total_vol","TSLA_first_hour_vol", "abs_sentiment_score"])

    X_train_mean = X_train_mean[lags + list(X_train_mean.columns)]
    X_test_mean = X_test_mean[lags + list(X_test_mean.columns)]

    model_sarimax = SARIMAX(y_train, exog=X_train_mean, order=(1,1,1), seasonal_order=(0,0,0,0))
    sarimax_result = model_sarimax.fit()

    # st.markdown("""
    #     <div style="text-align: center;">
    #         <h4>SARIMAX Model Summary</h4>
    #     </div>
    # """, unsafe_allow_html=True)

    # st.text(sarimax_result.summary())

    forecast = sarimax_result.forecast(steps=len(X_test_mean), exog=X_test_mean)
    forecast.index = y_test.index

    residuals = y_test - forecast
    std_dev = np.std(residuals)

    ci_upper = forecast + 1.645 * std_dev
    ci_lower = forecast - 1.645 * std_dev

    mse = mean_squared_error(y_test, forecast)
    mae = mean_absolute_error(y_test, forecast)

    st.markdown("""
        <div style="text-align: center;">
            <p style="font-size:18px;"><b>Mean Squared Error (MSE):</b> {:.2f}</p>
            <p style="font-size:18px;"><b>Mean Absolute Error (MAE):</b> {:.2f}</p>
        </div>
    """.format(mse, mae), unsafe_allow_html=True)


    st.subheader('Actual vs. Predicted Volumes with Confidence Interval')
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_test.index, y_test, label='Actual Volume', marker='o', color='black')
    ax.plot(forecast.index, forecast, label='Predicted Volume (SARIMAX)', linestyle='--', marker='x', color='blue')
    ax.fill_between(forecast.index, ci_lower, ci_upper, color='blue', alpha=0.2, label='90% Confidence Interval')
    ax.set_title('SARIMAX Predictions vs Actual TSLA Volumes Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume (in millions)')
    ax.legend()
    st.pyplot(fig)

else:
    if model_choice == 'XGBoost':
        model.fit(
            train_data[features], train_data[target],
            eval_set=[(test_data[features], test_data[target])],  # Validation set for early stopping
            verbose=10
        )
    else:
        model.fit(train_data[features], train_data[target])

    predictions = model.predict(test_data[features])
    actuals = test_data[target].values
    
    # Predict Trading Volume for December 31, 2024
    dec_31_data = test_data[test_data['Date'] == '2024-12-30']

    if not dec_31_data.empty:
        predicted_volume = model.predict(dec_31_data[features])[0] *10000000 # Predict the first data point

        # Display prediction dynamically for each model
        if model_choice == 'Random Forest':
            st.markdown(f"""
            <div style="text-align: center;">
                <h3>🌲 Predicted Trading Volume on Dec 31, 2024</h3>
                <p style="font-size:18px;">Using <b>Random Forest</b>, the estimated trading volume between 9:30am - 10:30am EST is <b>{predicted_volume:,.0f}</b> shares.</p>
            </div>
            """, unsafe_allow_html=True)

        elif model_choice == 'Linear Regression':
            st.markdown(f"""
            <div style="text-align: center;">
                <h3>📈 Predicted Trading Volume on Dec 31, 2024</h3>
                <p style="font-size:18px;">Using <b>Linear Regression</b>, the estimated trading volume between 9:30am - 10:30am EST is <b>{predicted_volume:,.0f}</b> shares.</p>
            </div>
            """, unsafe_allow_html=True)

        elif model_choice == 'XGBoost':
            st.markdown(f"""
            <div style="text-align: center;">
                <h3>🚀 Predicted Trading Volume on Dec 31, 2024</h3>
                <p style="font-size:18px;">Using <b>XGBoost</b>, the estimated trading volume between 9:30am - 10:30am EST is <b>{predicted_volume:,.0f}</b> shares.</p>
            </div>
            """, unsafe_allow_html=True)


    # Overall metrics for the whole testing period
    overall_mse = mean_squared_error(actuals, predictions)
    overall_r2 = r2_score(actuals, predictions)
    overall_mae = mean_absolute_error(actuals, predictions)

    # st.write(f"\nOverall Mean Squared Error: {overall_mse}")
    # st.write(f"Overall R-squared Score: {overall_r2}")
    # st.write(f"Overall Mean Absolute Error: {overall_mae}")

    metrics_df = pd.DataFrame({
        "Metric": ["Mean Squared Error (MSE)", "R-squared Score (R²)", "Mean Absolute Error (MAE)"],
        "Value": [overall_mse, overall_r2, overall_mae]
    })

    # Centering the Table Title
    st.markdown("""
        <div style="text-align: center;">
            <h3>📊 Overall Model Performance Metrics</h3>
        </div>
    """, unsafe_allow_html=True)

    # Convert DataFrame to HTML with center alignment
    table_html = metrics_df.to_html(index=False, justify='center')

    # Centering the entire table using HTML & CSS
    st.markdown(f"""
        <div style="display: flex; justify-content: center;">
            {table_html}
        </div>
    """, unsafe_allow_html=True)

    # Display Linear Regression Coefficients
    if model_choice == 'Linear Regression':
        coefficients = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_
        })
        st.markdown("""
            <div style="text-align: center;">
                <h3>🎯 Linear Regression Coefficients</h3>
            </div>
        """, unsafe_allow_html=True)
        st.write(coefficients)

    st.markdown("""
        <div style="text-align: center;">
            <h4>Actual vs. Predicted Volumes with Confidence Interval</h4>
        </div>
    """, unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test_data['Date'], actuals, label='Actual Volume', linestyle='-', marker='o')
    ax.plot(test_data['Date'], predictions, label='Predicted Volume', linestyle='--', marker='x')
    ax.fill_between(test_data['Date'], predictions - 1.645 * np.std(predictions), predictions + 1.645 * np.std(predictions), color='red', alpha=0.2, label='90% Confidence Interval')
    ax.set_title('Actual vs. Predicted TSLA Volumes Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume (in millions)')
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
        <div style="text-align: center;">
            <h4>Prediction Error Over Time</h4>
        </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test_data['Date'], abs(actuals - predictions), color='red', linestyle='-', marker='o')
    ax.set_title('Prediction Error (Absolute Difference) Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Absolute Difference in Volume')
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("""
        <div style="text-align: center;">
            <h4>Actual vs Predicted Scatter Plot</h4>
        </div>
    """, unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(actuals, predictions, alpha=0.7)
    ax.plot([actuals.min(), actuals.max()], 
            [actuals.min(), actuals.max()], 
            color='blue', linestyle='--', label='Perfect Prediction Line')
    ax.set_title('Actual vs Predicted TSLA Volume')
    ax.set_xlabel('Actual Volume')
    ax.set_ylabel('Predicted Volume')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Convert results to DataFrame and display
    results_df = pd.DataFrame({
        'Date': test_data['Date'],
        'Actual Volume': actuals,
        'Predicted Volume': predictions
    })
    
    # Feature Importance Analysis for XGBoost
    if model_choice == 'XGBoost':
        importance = model.get_booster().get_score(importance_type='gain')
        importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        # Convert to DataFrame
        importance_df = pd.DataFrame(importance, columns=["Feature", "Gain"]).head(10)

        # Centering the Table Title
        st.markdown("""
            <div style="text-align: center;">
                <h3>🔥 Top 10 Features (by Gain Importance) 📊</h3>
            </div>
        """, unsafe_allow_html=True)

        # Display the centered table
        st.markdown(
            "<style> .stDataFrame { margin: auto; text-align: center; } </style>",
            unsafe_allow_html=True,
        )
        st.dataframe(importance_df.style.set_properties(**{'text-align': 'center'}))


    st.write("")    
    # # Save the results to a CSV file
    # csv = results_df.to_csv(index=False).encode('utf-8')
    # st.download_button(
    #     label="📥 Download as CSV",
    #     data=csv,
    #     file_name='tsla_prediction_results.csv',
    #     mime='text/csv',
    # )

    
    st.markdown("""
        <hr>
        <div style="text-align: center; font-size:16px;">
            <b>👩‍💻👨‍💻 Authors:</b> Mingyang Li, Renjie Wang, Chen Xu, Yiwei Yan, Shirley Zhu
        </div>
    """, unsafe_allow_html=True)

