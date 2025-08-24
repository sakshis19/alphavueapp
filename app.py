import os
# Forcing a redeploy to fix theme
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import scipy.optimize as sco
import plotly.express as px
import mysql.connector
import bcrypt
from email_validator import validate_email, EmailNotValidError
import re
import yfinance

import streamlit as st

st.set_page_config(
    page_title="AlphaVue",
    page_icon="static/logo4.png",  # Path to your favicon file
    layout="wide"
)



# --- ENHANCED VISUALS & CUSTOM THEME ---
st.markdown("""
<style>
/* 1. Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

/* 2. Apply Font and Base Theme */
html, [class*="st-"], [class*="css-"] {
    font-family: 'Poppins', sans-serif;
}

/* ---
   3. CONDITIONAL BACKGROUND LOGIC (THE FIX)
   --- */

/* Default background for the login page */
body {
    background-image: url("https://i.postimg.cc/mDg9ZP1M/background.png");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Make containers transparent on the login page by default */
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"] {
    background: none !important;
}
[data-testid="stAppViewContainer"] > .main {
    background: transparent !important;
}

/* --- AGGRESSIVE White Background Fix (for logged-in state) --- */
body:has([data-testid="stSidebar"]),
body:has([data-testid="stSidebar"]) [data-testid="stAppViewContainer"],
body:has([data-testid="stSidebar"]) [data-testid="stAppViewContainer"] > .main {
    background: #f9f7f1 !important; /* Force white background and remove image */
}

/* Set color for the st.header in the sidebar to white */
[data-testid="stSidebar"] h2 {
    color: white;
}
/* ---
   4. STANDARD APP STYLING
   --- */
[data-testid="stSidebar"] {
    background-color: #126262;
}
[data-testid="stSidebar"] * {
    color: #000000; /* Changed to white for better contrast */
}
h1, h2, h4, h5, h6 {
    color: #000000;
}

/* Ensure various text elements are black */
.main p, 
[data-testid="stMetricValue"], 
[data-testid="stMetricLabel"],
label,
summary {
     color: #000000 !important;
}
            
.black-text p {
    color: #000000 !important;
}
            
.card {
    background-color: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(10px);
    border-radius: 10px;
    padding: 25px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin: 1rem 0;
    border: 1px solid rgba(255, 255, 255, 0.18);
}
            
/* --- NEW: Style for default/inactive sidebar buttons --- */
[data-testid="stSidebar"] button {
    background-color: transparent;
    border: 1px solid #FFFFFF;
    transition: all 0.2s ease-in-out; /* Smooth hover effect */
}
[data-testid="stSidebar"] button:hover {
    background-color: #006060; /* A slightly darker teal on hover */
    border-color: #FFFFFF;
}
[data-testid="stSidebar"] button p { /* Target the text inside the button */
     color: #FFFFFF;
}
            
/* --- MODIFIED: Style for the ACTIVE sidebar button --- */
.active-button > button {
    background-color: #FFD700 !important; /* Gold for active state */
    border: 2px solid #FFFFFF !important;
}
.active-button > button p {
    color: #2F4F4F !important; /* Dark text on a light gold background for contrast */
}
[data-testid="stSidebar"] button > div {
    display: flex;
    justify-content: flex-start;
    align-items: center;
    gap: 12px;
}
.login-header h1 {
    margin-bottom: 0 !important; 
    padding-bottom: 0;
    color: #F5F5DC; /* White for the main title */
    text-shadow: 0px 2px 4px rgba(0,0,0,0.5); /* Add shadow for readability */
}
/* Style for the first paragraph (tagline) */
.login-header p:first-of-type {
    font-size: 1.2em;
    color: #F5F5DC;
    margin-top: 0;
}

/* Style for the second paragraph (description) */
.login-header p:last-of-type {
    font-size: 1rem;
    color: #F5F5DC; /* Applies the correct light color */
    margin-top: 1.5rem;
}
.steps-container {
    display: flex;
    flex-wrap: wrap;
    align-items: stretch;
    justify-content: space-between;
    gap: 15px;
    margin-top: 2rem;
    width: 100%;
}
.step-card {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 12px;
    padding: 1.5rem 1rem;
    text-align: center;
    flex: 1 1 calc(25% - 15px);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    cursor: pointer;
}
.step-card:hover {
    transform: translateY(-8px) scale(1.03);
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
}
.step-card h5 {
    font-size: 1rem;
    margin: 0 0 8px 0;
    color: #008080;
    font-weight: 600;
}
.step-card p {
    font-size: 0.85rem !important;
    color: #555 !important;
    margin: 0 !important;
    line-height: 1.4;
}
            
/* --- USP CARDS STYLING --- */
.usp-section {
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid #e0e0e0; /* Light separator line */
}

.usp-container {
    display: flex;
    justify-content: space-between;
    gap: 15px; /* Spacing between cards */
}

.usp-card {
    display: flex;
    flex-direction: column; /* Icon above text */
    align-items: center;
    justify-content: center;
    text-align: center;
    background-color: #ffffff;
    border-radius: 10px;
    padding: 1rem 0.5rem;
    flex: 1; /* Each card takes equal space */
    border: 1px solid #E0E0E0;
    transition: all 0.25s ease;
}

.usp-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
}

.usp-icon {
    font-size: 2rem; /* Size of the icon */
    margin-bottom: 0.5rem;
}

.usp-text {
    font-size: 0.9rem;
    font-weight: 600;
    color: #FFFFFF;
    line-height: 1.3;
}
            
.sidebar-logo {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 80%;
    margin-bottom: 20px;
}
/* Add this to your main <style> block */

.market-section {
    background: transparent !important;
    border: none !important;
    backdrop-filter: none !important;
    box-shadow: none !important;
    /* We keep the margin to maintain spacing */
    margin-top: 2rem;
}

.market-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #000080 !important; /* Changed to white */
    text-align: center;
    margin-bottom: 1.5rem;
    text-shadow: 0px 2px 4px rgba(0,0,0,0.5); /* Added for readability */
}

.index-cards-container, .stock-cards-container {
    display: flex;
    justify-content: space-around;
    gap: 1rem;
    flex-wrap: wrap;
}

.index-card {
    flex-grow: 1;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    color: #FFFFFF;
}

.stock-card {
    flex-basis: calc(30% - 1rem); /* Make cards slightly narrower */
    padding: 0.5rem; /* Reduced padding */
    border-radius: 8px;
    color: #FFFFFF;
    text-align: center;
    flex-grow: 1; /* Allow cards to grow to fill space */
}

.gainer {
    background: linear-gradient(135deg, #2E8B57, #3CB371); /* Sea Green gradient */
}

.loser {
    background: linear-gradient(135deg, #A52A2A, #CD5C5C); /* Brown to Indian Red gradient */
}

.card-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.25rem;
}

.card-value {
    font-size: 1.5rem;
    font-weight: 700;
}

.card-change {
    font-size: 1rem;
    font-weight: 500;
}

.stock-card .card-title { font-size: 0.8rem; font-weight: 600; }
.stock-card .card-value { font-size: 1.0rem; font-weight: 700; }
.stock-card .card-change { font-size: 0.8rem; }
            
/* Add this to your main <style> block */

.stock-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #000080 !important; /* Changed to white */
    text-align: center;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    text-shadow: 0px 2px 4px rgba(0,0,0,0.5); /* Added for readability */
}
/* Add these new rules to your main <style> block */

/* --- STYLING FOR THE "GET STARTED" BUTTON --- */
.get-started-button {
    background: linear-gradient(45deg, #008080, #00A3A3);
    color: white;
    font-size: 1.2rem;
    font-weight: 600;
    padding: 1rem 2rem;
    border-radius: 50px;
    border: 2px solid white;
    cursor: pointer;
    transition: all 0.3s ease;
}
.get-started-button:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 25px rgba(0, 128, 128, 0.5);
}
/* Target the popover's wrapper to control its size */

/* Style the button inside the popover */
[data-testid="stPopover"] button {
    background: linear-gradient(45deg, #F0E68C, #FFF8DC); /* Soft Gold/Champagne gradient */
    color: #000080; /* Navy Blue text */
    font-size: 1.2rem;
    font-weight: 600;
    padding: 0.8rem 2rem; 
    border-radius: 50px;
    border: 2px solid #DAA520; /* A darker gold border for definition */
    transition: all 0.3s ease;
}
[data-testid="stPopover"] button:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 25px rgba(218, 165, 32, 0.4); /* Gold shadow */
}
/* --- STYLING FOR THE POPOVER --- */

/* Apply glassmorphism to the popover's content box */
[data-testid="stPopoverContent"] {
    background: rgba(255, 255, 255, 0.35) !important; /* Semi-transparent white */
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important; /* For Safari */
    border-radius: 15px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1) !important;
}

/* Ensure text inside the popover is dark and readable */
[data-testid="stPopoverContent"] h3,
[data-testid="stPopoverContent"] label,
[data-testid="stPopoverContent"] span {
    color: #FFFFFF !important;
}
/* Add this to your main <style> block */

.footer-disclaimer {
    text-align: center;
    padding: 2rem 1rem 1rem 1rem;
    font-size: 0.8rem;
    color: #000000; /* Muted gray color */
    max-width: 800px;
    margin: 0 auto;
}
            
/* Style for expanders on the admin page */
[data-testid="stExpander"] {
    border: 1px solid #000000 !important;           /* Add a solid black outline */
    border-radius: 10px;                             /* Add rounded corners */
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.15);     /* Add a subtle shadow */
    transition: all 0.2s ease-in-out;                /* Smooth transition for hover */
}

/* Make the shadow more pronounced on hover */
[data-testid="stExpander"]:hover {
    box-shadow: 4px 4px 8px rgba(0, 0, 0, 0.25);     /* Enhance shadow on mouse-over */
}
            
/* Force st.subheader (like "Admin Actions") to be black */
[data-testid="stSubheader"] {
    color: #000000 !important;
}

/* Set text color for asset allocation cards to black */
.asset-card span {
    color: black;
}
            
/* --- Custom Style for st.tabs --- */

/* Set default tab text color to black */
button[data-baseweb="tab"] {
    color: black;
}

/* Set selected tab text and underline color to orange */
button[data-baseweb="tab"][aria-selected="true"] {
    color: orange !important;
    border-bottom-color: orange !important;
}
</style>
""", unsafe_allow_html=True)

# --- DATABASE CONNECTION ---

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=st.secrets["connections"]["mysql"]["host"],
            user=st.secrets["connections"]["mysql"]["user"],
            password=st.secrets["connections"]["mysql"]["password"],
            database=st.secrets["connections"]["mysql"]["database"]
        )
        if connection.is_connected():
            return connection
    except mysql.connector.Error as err:
        st.error(f"Database Error: {err}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}. Please ensure your secrets.toml file is correctly configured.")
        return None

# --- DATABASE FUNCTIONS ---
def create_user(username, email, password, admin_code_attempt=None):
    # Determine the role based on the admin code
    role = 'user' # Default role
    
    # Directly read the admin code from environment variables
    correct_admin_code = st.secrets["app_secrets"]["admin_code"]    
    # If an admin code was provided and it matches the one in our settings, grant admin role
    if admin_code_attempt and correct_admin_code and admin_code_attempt == correct_admin_code:
        role = 'admin'

    # The rest of the function is the same, but now uses the 'role' variable
    conn = get_db_connection()
    if not conn: return False
    
    cursor = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        # Include the determined role in the INSERT statement
        cursor.execute("INSERT INTO users (username, email, password_hash, role) VALUES (%s, %s, %s, %s)", 
                       (username, email, hashed_password, role))
        conn.commit()
        return True
    except mysql.connector.Error:
        return False
    finally:
        cursor.close()
        conn.close()

def check_user(username, password):
    conn = get_db_connection()
    if not conn: return None
    cursor = conn.cursor(dictionary=True)
    # Add 'role' to the SELECT statement
    cursor.execute("SELECT user_id, username, password_hash, role FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
        return user # The returned dictionary will now include the role
    return None

def save_portfolio(user_id, inputs, risk_profile, stock_df, mf_df, summary_df):
    conn = get_db_connection()
    if not conn: return False
    cursor = conn.cursor()
    try:
        portfolio_sql = """
        INSERT INTO portfolios (user_id, age, experience, primary_goal, market_reaction, 
                                  horizon, stock_investment_amount, mf_investment_amount, mf_investment_mode, risk_appetite)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        portfolio_data = (
            user_id, inputs['age'], inputs['experience'], inputs['primary_goal'], inputs['market_reaction'],
            inputs['investment_horizon'], inputs['stock_investment_amount'], inputs['mf_investment_amount'],
            inputs['mf_investment_style'], risk_profile
        )
        cursor.execute(portfolio_sql, portfolio_data)
        portfolio_id = cursor.lastrowid

        if not stock_df.empty:
            stock_sql = "INSERT INTO portfolio_stocks (portfolio_id, ticker, invested_amount, expected_return_amount, weight) VALUES (%s, %s, %s, %s, %s)"
            for _, row in stock_df.iterrows():
                # Use the clean column names and divide percentage by 100 for raw weight
                raw_weight = row['Allocation Percent'] / 100.0
                stock_data = (portfolio_id, row['Stock Name'], row['Investment Amount'], row['Projected Return'], raw_weight)
                cursor.execute(stock_sql, stock_data)

        if not mf_df.empty:
            mf_sql = "INSERT INTO portfolio_mutual_funds (portfolio_id, fund_name, invested_amount, expected_return_amount, total_investment_sip, weight) VALUES (%s, %s, %s, %s, %s, %s)"
            for _, row in mf_df.iterrows():
                total_sip = row.get('Total Contribution (₹)', None)
                mf_data = (portfolio_id, row['Fund Name'], row['Lumpsum Investment (₹)'] if 'Lumpsum Investment (₹)' in row else row['Monthly Investment (₹)'], row['Projected Return (₹)'], total_sip, row['Allocation (%)'])
                cursor.execute(mf_sql, mf_data)
        
        if not summary_df.empty:
            summary_sql = "INSERT INTO portfolio_summary (portfolio_id, investment_period, total_investment, estimated_value, profit_earned, return_rate) VALUES (%s, %s, %s, %s, %s, %s)"
            for _, row in summary_df.iterrows():
                # Clean the return rate string before saving
                return_rate_cleaned = row['Return Rate (%)'].replace('%', '')
                summary_data = (portfolio_id, row['Investment Period'], row['Total Investment'], row['Estimated Value'], row['Profit Earned'], return_rate_cleaned)
                cursor.execute(summary_sql, summary_data)

        conn.commit()
        return True
    except mysql.connector.Error as err:
        st.error(f"Database save error: {err}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

def get_user_portfolio(user_id):
    conn = get_db_connection()
    if not conn: return None, None, None, None
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM portfolios WHERE user_id = %s ORDER BY saved_at DESC LIMIT 1", (user_id,))
        portfolio = cursor.fetchone()
        if not portfolio:
            return None, None, None, None
        
        portfolio_id = portfolio['portfolio_id']
        cursor.execute("SELECT * FROM portfolio_stocks WHERE portfolio_id = %s", (portfolio_id,))
        stocks = pd.DataFrame(cursor.fetchall())
        
        cursor.execute("SELECT * FROM portfolio_mutual_funds WHERE portfolio_id = %s", (portfolio_id,))
        mfs = pd.DataFrame(cursor.fetchall())
        
        cursor.execute("SELECT * FROM portfolio_summary WHERE portfolio_id = %s", (portfolio_id,))
        summary_df = pd.DataFrame(cursor.fetchall())
        
        return portfolio, stocks, mfs, summary_df
    finally:
        cursor.close()
        conn.close()

def delete_portfolio(portfolio_id):
    conn = get_db_connection()
    if not conn: return False
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM portfolios WHERE portfolio_id = %s", (portfolio_id,))
        conn.commit()
        return True
    except mysql.connector.Error as err:
        st.error(f"Delete Error: {err}")
        return False
    finally:
        cursor.close()
        conn.close()

def update_portfolio(portfolio_id, user_id, inputs, risk_profile, stock_df, mf_df, summary_df):
    conn = get_db_connection()
    if not conn: return False
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM portfolio_stocks WHERE portfolio_id = %s", (portfolio_id,))
        cursor.execute("DELETE FROM portfolio_mutual_funds WHERE portfolio_id = %s", (portfolio_id,))
        cursor.execute("DELETE FROM portfolio_summary WHERE portfolio_id = %s", (portfolio_id,))

        update_sql = """
        UPDATE portfolios SET age=%s, experience=%s, primary_goal=%s, market_reaction=%s, horizon=%s,
        stock_investment_amount=%s, mf_investment_amount=%s, mf_investment_mode=%s, risk_appetite=%s
        WHERE portfolio_id = %s AND user_id = %s
        """
        update_data = (
            inputs['age'], inputs['experience'], inputs['primary_goal'], inputs['market_reaction'],
            inputs['investment_horizon'], inputs['stock_investment_amount'], inputs['mf_investment_amount'],
            inputs['mf_investment_style'], risk_profile, portfolio_id, user_id
        )
        cursor.execute(update_sql, update_data)

        if not stock_df.empty:
            stock_sql = "INSERT INTO portfolio_stocks (portfolio_id, ticker, invested_amount, expected_return_amount, weight) VALUES (%s, %s, %s, %s, %s)"
            for _, row in stock_df.iterrows():
                # In update_portfolio function
                raw_weight = row['Allocation Percent'] / 100.0
                stock_data = (portfolio_id, row['Stock Name'], row['Investment Amount'], row['Projected Return'], raw_weight)
                cursor.execute(stock_sql, stock_data)

        if not mf_df.empty:
            mf_sql = "INSERT INTO portfolio_mutual_funds (portfolio_id, fund_name, invested_amount, expected_return_amount, total_investment_sip, weight) VALUES (%s, %s, %s, %s, %s, %s)"
            for _, row in mf_df.iterrows():
                total_sip = row.get('Total Contribution (₹)', None)
                mf_data = (portfolio_id, row['Fund Name'], row['Lumpsum Investment (₹)'] if 'Lumpsum Investment (₹)' in row else row['Monthly Investment (₹)'], row['Projected Return (₹)'], total_sip, row['Allocation (%)'])
                cursor.execute(mf_sql, mf_data)

        if not summary_df.empty:
            summary_sql = "INSERT INTO portfolio_summary (portfolio_id, investment_period, total_investment, estimated_value, profit_earned, return_rate) VALUES (%s, %s, %s, %s, %s, %s)"
            for _, row in summary_df.iterrows():
                # Clean the return rate string before saving
                return_rate_cleaned = row['Return Rate (%)'].replace('%', '')
                summary_data = (portfolio_id, row['Investment Period'], row['Total Investment'], row['Estimated Value'], row['Profit Earned'], return_rate_cleaned)
                cursor.execute(summary_sql, summary_data)

        conn.commit()
        return True
    except mysql.connector.Error as err:
        st.error(f"Update Error: {err}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

# --- HELPER FUNCTIONS ---
def calculate_sip_future_value(monthly_investment, annual_rate, investment_years):
    if annual_rate == 0: return monthly_investment * investment_years * 12, monthly_investment * investment_years * 12
    monthly_rate = annual_rate / 12 / 100
    months = investment_years * 12
    future_value = monthly_investment * ((((1 + monthly_rate)**months) - 1) / monthly_rate) * (1 + monthly_rate)
    total_investment = monthly_investment * months
    return future_value, total_investment

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_std, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    if p_std == 0: return float('inf')
    return -(p_ret - risk_free_rate) / p_std

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for asset in range(num_assets))
    initial_weights = num_assets * [1. / num_assets,]
    result = sco.minimize(neg_sharpe_ratio, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# --- HELPER FUNCTIONS ---

@st.cache_data(ttl=75) # Cache for 5 minutes
def fetch_market_data(tickers):
    data = {}
    try:
        for ticker_symbol in tickers:
            ticker_info = yfinance.Ticker(ticker_symbol)
            hist = ticker_info.history(period="2d") # Get last 2 days to calculate change
            if not hist.empty and len(hist) > 1:
                prev_close = hist['Close'].iloc[0]
                current_price = hist['Close'].iloc[1]
                change = current_price - prev_close
                percent_change = (change / prev_close) * 100
                data[ticker_symbol] = {
                    "name": ticker_info.info.get('shortName', ticker_symbol).replace(' S&P BSE', ''),
                    "value": f"{current_price:,.2f}",
                    "change": f"{change:,.2f}",
                    "percent_change": f"{percent_change:.2f}%"
                }
    except Exception as e:
        st.toast(f"Could not fetch market data: {e}", icon="⚠️")
        return {} # Return empty on error
    return data

# --- DATA LOADING ---
@st.cache_resource
def load_models_and_data():
    models = {"risk_profiler": joblib.load('data/risk_profiler_model.joblib'), "allocator": joblib.load('data/allocator_model.joblib'), "scaler": joblib.load('data/allocator_scaler.joblib'), "risk_columns": joblib.load('data/risk_model_columns.joblib'), "alloc_map": joblib.load('data/allocation_map.joblib')}
    return models

@st.cache_data
def load_csv_files():
    data_files = {"stocks_fc": pd.read_csv('data/stock_forecast_results.csv'), "stocks_risk": pd.read_csv('data/Stock_Risk_Categories.csv'), "mf_fc": pd.read_csv('data/mutual_fund_forecast_metrics_filtered.csv'), "mf_meta": pd.read_csv('data/Mutual_Fund_Metadata.csv').dropna(subset=['Scheme_Name']), "all_prices": pd.read_csv('data/all_stocks_close_prices.csv', index_col='Date', parse_dates=True), "mf_navs": pd.read_csv('data/combined_mutual_fund_navs.csv', index_col='date', parse_dates=True), "stock_name_map": pd.read_excel('data/New_Stocklist.xlsx')}
    return data_files

models = load_models_and_data()
data_files = load_csv_files()
risk_profiler_model = models["risk_profiler"]
risk_model_columns = models["risk_columns"]
allocator_model = models["allocator"]
allocator_scaler = models["scaler"]
allocation_map = models["alloc_map"]
stocks_fc = data_files["stocks_fc"]
stocks_risk = data_files["stocks_risk"]
mf_fc = data_files["mf_fc"]
mf_meta = data_files["mf_meta"]
all_prices = data_files["all_prices"]
mf_navs = data_files["mf_navs"]
stock_name_map = data_files["stock_name_map"]

# --- PAGE DEFINITIONS ---

def display_disclaimer():
    st.markdown("""
        <div class="footer-disclaimer">
            <p>
                <b>Disclaimer:</b> This is not investment advice. The information and recommendations provided by AlphaVue are for educational and informational purposes only. All investments involve risk, and the past performance of a security or a financial product does not guarantee future results or returns. It is important to conduct your own research and consult with a qualified financial advisor before making any investment decisions.
            </p>
        </div>
    """, unsafe_allow_html=True)

def set_bg(image_url):
    st.markdown(f"""
        <style>
        html, body, [data-testid="stAppViewContainer"] {{
            background: url("{image_url}") no-repeat center center fixed !important;
            background-size: cover !important;
        }}
        </style>
    """, unsafe_allow_html=True)

# --- PAGE DEFINITIONS ---

def admin_page():
    st.markdown('<h1 style="color: black;">🔑 Admin Panel</h1>', unsafe_allow_html=True)

    if st.session_state.get('role') != 'admin':
        st.error("You do not have permission to view this page.")
        return

    st.markdown('<h2 style="color: black;">User Management</h2>', unsafe_allow_html=True)

    with st.form("search_form"):
        search_term = st.text_input("Search by User ID or Username:", help="Enter a User ID (e.g., 5) or a username (e.g., Rohan).")
        submit_search = st.form_submit_button("Search")

    conn = get_db_connection()
    if not conn:
        st.error("Could not connect to the database.")
        return

    try:
        base_query = "SELECT user_id, username, email, role, created_at FROM users"
        params = None
        
        # --- MODIFIED: Dynamic SQL Query based on input type ---
        if submit_search and search_term:
            if search_term.isdigit():
                # If input is a number, search by user_id
                query = f"{base_query} WHERE user_id = %s"
                params = (int(search_term),)
            else:
                # If input is text, search by username
                query = f"{base_query} WHERE username LIKE %s"
                params = (f"%{search_term}%",)
            
            st.write(f"Showing results for '{search_term}':")
            users_df = pd.read_sql(query, conn, params=params)
        else:
            # Fetch all users if no search is performed
            users_df = pd.read_sql(base_query, conn)

        st.dataframe(users_df, use_container_width=True)
        
        if not users_df.empty:
            st.markdown('<h3 style="color: black;">Admin Actions</h3>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                with st.expander("✏️ Change User Role"):
                    user_to_change = st.selectbox("Select User by ID", users_df['user_id'].tolist(), key="role_user")
                    new_role = st.selectbox("Select New Role", ['user', 'admin'], key="new_role")
                    if st.button("Update Role"):
                        cursor = conn.cursor()
                        cursor.execute("UPDATE users SET role = %s WHERE user_id = %s", (new_role, user_to_change))
                        conn.commit()
                        st.success(f"User {user_to_change}'s role updated to {new_role}.")
                        st.rerun()

            with col2:
                with st.expander("🗑️ Delete User"):
                    deletable_users = users_df[users_df['user_id'] != st.session_state['user_id']]
                    if not deletable_users.empty:
                        user_to_delete = st.selectbox("Select User by ID", deletable_users['user_id'].tolist(), key="delete_user")
                        if st.button("Delete User Permanently", type="primary"):
                            cursor = conn.cursor()
                            cursor.execute("DELETE FROM users WHERE user_id = %s", (user_to_delete,))
                            conn.commit()
                            st.warning(f"User {user_to_delete} has been deleted.")
                            st.rerun()
                    else:
                        st.info("No other users to delete.")
        else:
            if submit_search and search_term:
                message = "No users found matching your search term."

                st.markdown(f'''
                <div style="
                    background-color: #B5C7EB; 
                    border-left: 5px solid #0041C2; 
                    padding: 1rem; 
                    border-radius: 0.25rem; 
                    margin-bottom: 1rem;
                    ">
                    <span style="color: black;"> ⚠️ {message}
                    </span>
                </div>
                ''', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        if conn and conn.is_connected():
            conn.close()

def login_signup_page():
    # Use it at the top of login_signup_page
    set_bg("https://i.postimg.cc/cHjq5FKr/Subtle-white-and-light-gray-abstract-background-with-a-soft-texture.jpg")

    # --- PAGE CONTENT ---
    # 1. Welcome Header
    st.markdown("""
        <div style="text-align: center; padding-top: 2rem;">
            <h1 style="color:#000080; font-size: 3.8rem; text-shadow: 0px 2px 5px rgba(0,0,0,0.5);">Welcome to AlphaVue  </h1>
            <p style="color:#000080; font-size: 1.0rem;">For when "<i>trust me, bro</i>" isn't a financial plan.</p>
        </div>
    """, unsafe_allow_html=True)

    # 2. USP Cards
    st.markdown("""
        <div style="padding: 2rem 5rem;"> <div class="usp-container">
                <div class="usp-card" style="background-color: rgba(255, 255, 255, 0.4);">
                    <span class="usp-icon">📊</span>
                    <span class="usp-text" style="color:#000080;">Data-Backed Recommendations</span>
                </div>
                <div class="usp-card" style="background-color: rgba(255, 255, 255, 0.4);">
                    <span class="usp-icon">⚡</span>
                    <span class="usp-text" style="color:#000080;">Real-Time Insights</span>
                </div>
                <div class="usp-card" style="background-color: rgba(255, 255, 255, 0.4);">
                    <span class="usp-icon">🤖</span>
                    <span class="usp-text" style="color:#000080;">AI-Powered Analysis</span>
                </div>
                <div class="usp-card" style="background-color: rgba(255, 255, 255, 0.4);">
                    <span class="usp-icon">🔒</span>
                    <span class="usp-text" style="color:#000080;">Secure & Private</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 3. "Get Started" Popover Button
    _, center_col, _ = st.columns([2, 1, 2]) # Create 3 columns, we'll only use the middle one

    with center_col: # Place the popover in the central, narrower column
        with st.popover("Get Started", use_container_width=True):
            # ... all your form logic goes here, no changes needed inside ...
            choice = st.radio("Choose an option:", ["Login", "Sign Up"], horizontal=True, key="popover_choice", label_visibility="collapsed")
            
            if choice == "Login":
                st.markdown("<h3>Login to your account</h3>", unsafe_allow_html=True)
                with st.form("login_form_popover"):
                    username = st.text_input("Username", placeholder="Enter your username", label_visibility="collapsed")
                    password = st.text_input("Password", type="password", placeholder="Enter your password", label_visibility="collapsed")
                    submitted = st.form_submit_button("Login")
                    if submitted:
                        user = check_user(username, password)
                        if user:
                            st.session_state['logged_in'] = True
                            st.session_state['user_id'] = user['user_id']
                            st.session_state['username'] = user['username']
                            st.session_state['role'] = user['role']
                            
                            # --- THE FIX: Conditional redirection based on role ---
                            if st.session_state['role'] == 'admin':
                                st.session_state['page'] = 'Admin Panel'
                            else:
                                st.session_state['page'] = 'Home'
                            
                            st.rerun()
                        else:
                            st.error("Incorrect username or password")
            
            elif choice == "Sign Up":
                st.markdown("<h3>Create a new account</h3>", unsafe_allow_html=True)
                with st.form("signup_form_popover"):
                    username = st.text_input("Username", placeholder="Choose a username", label_visibility="collapsed")
                    email = st.text_input("Email", placeholder="Enter your email", label_visibility="collapsed")
                    password = st.text_input("Password", type="password", placeholder="Create a password", label_visibility="collapsed")
                    confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", label_visibility="collapsed")
                    admin_code = st.text_input("Admin Code (Optional)", placeholder="Enter admin code for admin access", label_visibility="collapsed")
                    submitted = st.form_submit_button("Sign Up")
                    if submitted:
                        if password != confirm_password:
                            st.error("Passwords do not match!")
                        elif not re.match(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$", password):
                            st.error("Password must be at least 8 characters long and include uppercase, lowercase, a number, and a special character.")
                        else:
                            try:
                                validate_email(email)
                                if create_user(username, email, password, admin_code_attempt=admin_code):
                                    st.success("Account created successfully! Please login.")
                                else:
                                    st.error("Username or Email already exists.")
                            except EmailNotValidError as e:
                                st.error(str(e))
                            
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr style='margin-top: 2rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)
    
    # --- MARKET DATA SECTION ---
    st.markdown('<div class="market-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="market-header">Live Market Snapshot</h3>', unsafe_allow_html=True)

    # Fetch Index Data
    indices_data = fetch_market_data(['^NSEI', '^BSESN'])

    if indices_data:
        # Use st.columns for robust side-by-side layout
        cols = st.columns(len(indices_data))
        for i, (symbol, data) in enumerate(indices_data.items()):
            with cols[i]:
                change_val = float(data['change'])
                arrow = "🔺" if change_val >= 0 else "🔻"
                color_class = "gainer" if change_val >= 0 else "loser"
                # Render each card inside its own column
                st.markdown(f"""
                    <div class="index-card {color_class}">
                        <div class="card-title">{data['name']}</div>
                        <div class="card-value">{data['value']}</div>
                        <div class="card-change">{arrow} {data['change']} ({data['percent_change']})</div>
                    </div>
                """, unsafe_allow_html=True)

    # Placeholder for Gainers/Losers
    st.markdown('<h4 class="stock-header">📈 Top Gainers & Losers (Nifty 50)</h4>', unsafe_allow_html=True)
    
    gainers = [
        {'name': 'TATA MOTORS', 'value': '1,012.50', 'change': '+3.5%'},
        {'name': 'ADANI PORTS', 'value': '1,450.10', 'change': '+2.8%'},
        {'name': 'HINDALCO', 'value': '690.75', 'change': '+2.5%'}
    ]
    losers = [
        {'name': 'INFOSYS', 'value': '1,550.80', 'change': '-2.1%'},
        {'name': 'HDFC BANK', 'value': '1,501.20', 'change': '-1.8%'},
        {'name': 'TCS', 'value': '3,890.45', 'change': '-1.5%'}
    ]

    gainer_cards_html = "".join([
        f"""<div class="stock-card gainer">
               <div class="card-title">{stock['name']}</div>
               <div class="card-value">₹{stock['value']}</div>
               <div class="card-change">🔺 {stock['change']}</div>
           </div>"""
        for stock in gainers
    ])
    st.markdown(f'<div class="stock-cards-container">{gainer_cards_html}</div>', unsafe_allow_html=True)
    
    loser_cards_html = "".join([
        f"""<div class="stock-card loser">
               <div class="card-title">{stock['name']}</div>
               <div class="card-value">₹{stock['value']}</div>
               <div class="card-change">🔻 {stock['change']}</div>
           </div>"""
        for stock in losers
    ])
    st.markdown(f'<div class="stock-cards-container" style="margin-top:0.5rem;">{loser_cards_html}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True) # Close .market-section div

    display_disclaimer()

def home_page():
    # --- ADMIN VIEW ---
    if st.session_state.get('role') == 'admin':
        st.markdown('<h1 style="color: black;">Admin Dashboard 👋</h1>', unsafe_allow_html=True)
        welcome_message = f"Welcome back, Admin {st.session_state['username']}. Here's a summary of the application's activity."
        st.markdown(f'<div class="black-text"><p>{welcome_message}</p></div>', unsafe_allow_html=True)
        st.divider()

        # Fetch stats from the database
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM users")
                user_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM portfolios")
                portfolio_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT user_id) FROM portfolios")
                users_with_portfolios = cursor.fetchone()[0]
                
                conn.close()

                # Display stats in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Registered Users", user_count)
                with col2:
                    st.metric("Total Portfolios Created", portfolio_count)
                with col3:
                    st.metric("Users with a Portfolio", users_with_portfolios)

            except Exception as e:
                st.error(f"Failed to fetch admin statistics: {e}")
        
        st.divider()

        # --- NEW: Admin Action Buttons in Columns ---
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to User Management 🔑"):
                st.session_state.page = "Admin Panel"
                # Hide dashboard when navigating away
                st.session_state.show_admin_dashboard = False
                st.rerun()
        
        with col2:
            # Button text changes based on whether the dashboard is shown or hidden
            button_text = "Hide Analytics Dashboard" if st.session_state.get('show_admin_dashboard') else "Show Analytics Dashboard"
            if st.button(f"{button_text} 📊"):
                # Toggle the state
                st.session_state.show_admin_dashboard = not st.session_state.get('show_admin_dashboard', False)
                st.rerun()

        # --- NEW: Conditionally display the Power BI dashboard ---
        if st.session_state.get('show_admin_dashboard'):
            st.header("App Analytics Dashboard")
            
            # ❗️ IMPORTANT: Replace this with your actual public Power BI URL
            power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiZmFhYjc4YTctMTIxNi00ZWJmLTllMmEtNjcxMmI3ZDQ4NjhjIiwidCI6IjE0ZjljNmYzLTIyMGUtNDA4Ni1iYzc5LTFlNjUxZTQwZDZhYiJ9&pageName=17f0075e1577ea32cb7d" 
            
            st.markdown(f'<iframe title="AlphaVue Admin Analytics" width="100%" height="600" src="{power_bi_url}" frameborder="0" allowFullScreen="true"></iframe>', unsafe_allow_html=True)
    
    # --- REGULAR USER VIEW ---
    else:
        # This single function call now fetches everything, including the pre-calculated summary
        portfolio, stocks, mfs, summary_df = get_user_portfolio(st.session_state['user_id'])
        
        # SCENARIO 1: USER HAS A SAVED PORTFOLIO
        if portfolio:
            st.markdown(f"<h1 style='color: black;'>Welcome Back, {st.session_state['username']}! 👋</h1>", unsafe_allow_html=True)

            st.markdown("<h2 style='color: black;'>Your Portfolio At a Glance</h2>", unsafe_allow_html=True)

            # --- Calculate Key Metrics FROM FETCHED DATA (No Recalculation) ---
            total_stock_investment = float(portfolio.get('stock_investment_amount', 0))
            total_mf_investment = 0
            if not mfs.empty and portfolio['mf_investment_mode'] != 'Lumpsum (One-Time)':
                total_mf_investment = mfs['total_investment_sip'].astype(float).sum()
            else:
                total_mf_investment = float(portfolio.get('mf_investment_amount', 0))
            
            total_investment = total_stock_investment + total_mf_investment

            # --- THE FIX: Get the final projected value directly from the saved summary table ---
            total_projected_value = 0
            if not summary_df.empty:
                # Get the last row of the summary table for the final projection
                final_projection_str = summary_df.iloc[-1]['estimated_value'].replace('₹', '').replace(',', '')
                total_projected_value = float(final_projection_str)
                
            overall_return_pct = ((total_projected_value - total_investment) / total_investment) * 100 if total_investment > 0 else 0

            # --- Display Key Metrics in Columns ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="💰 Total Investment", value=f"₹{total_investment:,.2f}")
            with col2:
                st.metric(label="🚀 Projected Value", value=f"₹{total_projected_value:,.2f}", help=f"Projected over {portfolio['horizon']} years")
            with col3:
                st.metric(label="📈 Overall Return", value=f"{overall_return_pct:.2f}%", delta=f"{overall_return_pct:.2f}%")
            
            st.divider()

            # --- VISUAL ASSET ALLOCATION ---
            st.markdown("<h3 style='color: black;'>Visual Asset Allocation</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            fig_stocks, fig_mfs = None, None

            with col1:
                if not stocks.empty:
                    stock_data = stocks[['ticker', 'invested_amount']].copy()
                    stock_data['invested_amount'] = stock_data['invested_amount'].astype(float)
                    stock_data['parent'] = 'Stocks'
                    fig_stocks = px.sunburst(stock_data, 
                                            path=['parent', 'ticker'], 
                                            values='invested_amount', 
                                            color_discrete_sequence=px.colors.sequential.Teal)
                    
                    # --- MODIFIED THIS LINE ---
                    fig_stocks.update_traces(textinfo='label+percent parent', textfont_color='black')
                    
                    # --- MODIFIED THIS BLOCK ---
                    fig_stocks.update_layout(
                        title_text='<b>Stock Breakup</b>',   # Set and bold the title
                        title_font_color="black",        # Set title color to black
                        title_x=0.5,                     # Center the title
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig_stocks, use_container_width=True)
                    
                else:
                    st.info("No stock investments to display.")

            with col2:
                if not mfs.empty:
                    mf_data = mfs[['fund_name', 'invested_amount']].copy()
                    if portfolio['mf_investment_mode'] != 'Lumpsum (One-Time)':
                        mf_data['invested_amount'] = mfs['total_investment_sip']
                    mf_data['invested_amount'] = mf_data['invested_amount'].astype(float)
                    mf_data['parent'] = 'Mutual Funds'
                    fig_mfs = px.sunburst(mf_data, 
                                            path=['parent', 'fund_name'], 
                                            values='invested_amount', 
                                            color_discrete_sequence=px.colors.sequential.Aggrnyl)

                    # --- MODIFIED THIS LINE ---
                    fig_mfs.update_traces(textinfo='label+percent parent', textfont_color='black')

                    # --- MODIFIED THIS BLOCK ---
                    fig_mfs.update_layout(
                        title_text='<b>Mutual Fund Breakup</b>', # Set and bold the title
                        title_font_color="black",            # Set title color to black
                        title_x=0.5,                         # Center the title
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )

                    st.plotly_chart(fig_mfs, use_container_width=True)
                    
                else:
                    st.info("No mutual fund investments to display.")
            
            st.divider()

            # --- DETAILED PORTFOLIO VIEW ---
            st.markdown("<h3 style='color: black;'>Detailed Portfolio View</h3>", unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["📈 Stocks", "💰 Mutual Funds", "✨ Summary"])

            with tab1:
                stocks_display = stocks.drop(columns=['portfolio_id', 'stock_record_id'], errors='ignore').rename(columns={'ticker': 'Stock Name', 'invested_amount': 'Invested Amount (₹)', 'expected_return_amount': 'Projected Return (₹)', 'weight': 'Allocation (%)'})
                st.dataframe(stocks_display, use_container_width=True)
            with tab2:
                mfs_display = mfs.drop(columns=['portfolio_id', 'mf_record_id'], errors='ignore').rename(columns={'fund_name': 'Fund Name', 'invested_amount': 'Invested Amount (₹)', 'expected_return_amount': 'Projected Return (₹)', 'total_investment_sip': 'Total SIP Investment (₹)', 'weight': 'Allocation (%)'})
                st.dataframe(mfs_display, use_container_width=True)
            with tab3:
                st.markdown("<h3 style='color: black;'>Your Portfolio's Projected Growth</h3>", unsafe_allow_html=True)

                if not summary_df.empty:
                    # --- NO RECALCULATION: Simply display the fetched data ---
                    summary_display = summary_df.drop(columns=['summary_id', 'portfolio_id'], errors='ignore').rename(columns={'investment_period': 'Investment Period', 'total_investment': 'Total Investment (₹)', 'estimated_value': 'Estimated Value (₹)', 'profit_earned': 'Profit Earned (₹)', 'return_rate': 'Return Rate (%)'})
                    st.dataframe(summary_display, use_container_width=True)
                else:
                    st.info("Summary data could not be loaded.")

            # --- ACTION BUTTONS ---
            st.write("")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Edit Recommendations"):
                    st.session_state['edit_mode'] = True
                    st.session_state['portfolio_to_edit'] = portfolio
                    st.session_state['page'] = "Get Recommendations"
                    st.rerun()
            with col2:
                if st.button("Delete Recommendations"):
                    if delete_portfolio(portfolio['portfolio_id']):
                        success_message = "Recommendations deleted successfully!"
                        st.markdown(f'''
                        <div style="
                            background-color: #e6ffed; 
                            border-left: 5px solid #28a745; 
                            padding: 1rem; 
                            border-radius: 0.25rem; 
                            margin-bottom: 1rem;
                            ">
                            <p style="color: black;"> {success_message}
                            </p>
                        </div>
                        ''', unsafe_allow_html=True)
                        st.rerun()
                    else:
                        st.error("Could not delete Recommendations.")
        
        # SCENARIO 2: NEW USER WITH NO PORTFOLIO
        else:
            st.markdown(f"<h1 style='color: black;'>Welcome to AlphaVue, {st.session_state['username']}! 👋</h1>", unsafe_allow_html=True)
            st.markdown("""
            <div class='card' style='text-align: center;'>
                <h3 style='color: black;'>You haven't created a portfolio yet.</h3>
                <p style='color: black;'>Click on the "Get Recommendations" button to get your personalized investment recommendations and start your journey towards financial growth.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
                        <div class="steps-container">
                            <div class="step-card">
                                <h5>Step 1</h5>
                                <p>Answer a few questions about your goals</p>
                            </div>
                            <div class="step-card">
                                <h5>Step 2</h5>
                                <p>Get your AI-powered portfolio plan</p>
                            </div>
                            <div class="step-card">
                                <h5>Step 3</h5>
                                <p>Invest, track, and grow your wealth</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
        display_disclaimer()
       

def recommendation_page():
    edit_data = st.session_state.get('portfolio_to_edit', {}) if st.session_state.get('edit_mode') else {}
    st.markdown("<h1 style='color: black;'>Tell Us About Yourself:</h1>", unsafe_allow_html=True)

    
    with st.form(key='user_profile_form'):
        st.markdown("<h3 style='color: black;'>About You:</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1: 
            age = st.number_input("What is your age?", 18, 70, int(edit_data.get('age', 35)), 1)
        with col2: 
            experience = st.selectbox("What is your investment experience?", ('Beginner', 'Intermediate', 'Advanced'), index=['Beginner', 'Intermediate', 'Advanced'].index(edit_data.get('experience', 'Beginner')))
        
        st.markdown("<h3 style='color: black;'>Your Investment Goals:</h3>", unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3: 
            primary_goal = st.selectbox("What is your primary investment goal?", ('Steady Growth', 'Capital Protection', 'Aggressive Wealth Creation'), index=['Steady Growth', 'Capital Protection', 'Aggressive Wealth Creation'].index(edit_data.get('primary_goal', 'Steady Growth')))
        with col4: 
            market_reaction = st.selectbox("How would you react to a 20% market drop?", ('Do nothing', 'Buy more', 'Sell some', 'Sell all'), index=['Do nothing', 'Buy more', 'Sell some', 'Sell all'].index(edit_data.get('market_reaction', 'Do nothing')))
        
        st.markdown("<h3 style='color: black;'>Your Investment Plan:</h3>", unsafe_allow_html=True)
        col5, col6 = st.columns(2)
        with col5:
            investment_horizon = st.number_input("How many years do you plan to invest for?", 1, 40, int(edit_data.get('horizon', 10)), 1)
            stock_investment_amount = st.number_input("How much do you want to invest in stocks (₹)?", 10000, value=int(edit_data.get('stock_investment_amount', 50000)), step=5000)
        with col6:
            # This is the changed line:
            mf_investment_style = st.selectbox("How do you want to invest in mutual funds?", ('Lumpsum (One-Time)', 'Monthly SIP'), index=['Lumpsum (One-Time)', 'Monthly SIP'].index(edit_data.get('mf_investment_mode', 'Lumpsum (One-Time)')))
            mf_investment_amount = st.number_input("How much do you want to invest in mutual funds (₹)?", 500, value=int(edit_data.get('mf_investment_amount', 10000)), step=500)
        
        submit_label = "Update Profile" if st.session_state.get('edit_mode') else "Generate My Profile"
        submit_button = st.form_submit_button(label=submit_label)
        
        if submit_button:
            with st.spinner('Analyzing your profile...'):
                user_data = pd.DataFrame({'Age': [age], 'Primary_Goal': [primary_goal], 'Market_Drop_Reaction': [market_reaction], 'Investment_Experience': [experience]})
                user_data_encoded = pd.get_dummies(user_data)
                user_data_aligned = user_data_encoded.reindex(columns=risk_model_columns, fill_value=0)
                predicted_risk = risk_profiler_model.predict(user_data_aligned)[0]
                st.session_state['generated_recommendations'] = True
                st.session_state['risk_profile'] = predicted_risk
                st.session_state['user_inputs'] = {'age': age, 'experience': experience, 'primary_goal': primary_goal, 'market_reaction': market_reaction, 'investment_horizon': investment_horizon, 'stock_investment_amount': stock_investment_amount, 'mf_investment_amount': mf_investment_amount, 'mf_investment_style': mf_investment_style}

    if st.session_state.get('generated_recommendations', False):
        st.markdown(f"""
            <div class="card" style="padding: 15px; text-align: center; color: #126262;">
                Your predicted risk profile is: &nbsp; <span class="badge">{st.session_state['risk_profile']}</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
        user_profile_data = pd.DataFrame({'Age': [st.session_state['user_inputs']['age']],'Investment_Horizon_Yrs': [st.session_state['user_inputs']['investment_horizon']],'RiskProfile_Encoded': [risk_map[st.session_state['risk_profile']]]})
        scaled_data = allocator_scaler.transform(user_profile_data)
        cluster = allocator_model.predict(scaled_data)[0]
        allocation = allocation_map[cluster]

        st.markdown("<h1 style='color: black;'>Your Recommended Asset Allocation:</h1>", unsafe_allow_html=True)
        st.markdown('<p style="color: black;">Based on your profile, we recommend allocating your investment capital as follows:</p>', unsafe_allow_html=True)

        asset_icons = {"Equity": "📈", "Debt": "📄", "Gold": "🪙"}
        progress_bar_color = "#008080"
        for asset_class, percentage in allocation.items():
            icon = asset_icons.get(asset_class, '💰')
            st.markdown(f"""
            <div class="asset-card">
                <div class="asset-card-header">
                    <div class="asset-card-title">
                        <span class="asset-card-icon">{icon}</span>
                        <span>{asset_class}</span>
                    </div>
                    <span class="asset-card-value">{percentage}%</span>
                </div>
                <div style="background-color: #e9ecef; border-radius: 5px; height: 8px;">
                    <div style="background-color: {progress_bar_color}; width: {percentage}%; height: 100%; border-radius: 5px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("<h1 style='color: black;'>Your Personalized Portfolio:</h1>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='color: black;'>📈 Optimized Stock Portfolio</h3>", unsafe_allow_html=True)
        disclaimer_text = "Disclaimer: Projections are based on historical data and are for illustrative purposes only. Past performance is not indicative of future results."

        st.markdown(f'''
        <div style="
            background-color: #e6f3ff; 
            border-left: 5px solid #1c83e1; 
            padding: 1rem; 
            border-radius: 0.25rem; 
            margin-bottom: 1rem;
            ">
            <p style="color: black;"> ⚠️ {disclaimer_text}
            </p>
        </div>
        ''', unsafe_allow_html=True)

        num_stocks_to_consider = st.number_input(
            "How many Stocks should we recommend?", 
            min_value=2, max_value=20, value=5, step=1, 
            help="We will select the best stocks from this pool to build an optimized portfolio."
        )

        if 'risk_profile' in st.session_state:
            stock_risk_map = {'Low': ['Low Risk', 'Medium Risk'],'Medium': ['Medium Risk'],'High': ['Medium Risk', 'High Risk']}
            eligible_stock_categories = stock_risk_map[st.session_state['risk_profile']]
            eligible_stocks = stocks_risk[stocks_risk['Risk Category'].isin(eligible_stock_categories)]
            
            # This merge should now work. If it fails, the printout will tell us why.
            recommended_stocks = pd.merge(eligible_stocks, stocks_fc, on='StockName')
            
            candidate_stocks = recommended_stocks.sort_values(by='Historical_5Y_CAGR', ascending=False).head(num_stocks_to_consider)
            
            candidate_tickers = candidate_stocks['StockName'].tolist()
            prices_df = all_prices[candidate_tickers].copy()
            
            # ... (The rest of your code for this section remains the same) ...
            
            prices_df.ffill(inplace=True)
            prices_df.dropna(inplace=True) 

            if prices_df.empty or len(prices_df.columns) < 2:
                st.error("Insufficient historical data for the selected stocks to build a portfolio.")
            else:
                with st.spinner("Optimizing portfolio..."):
                    returns = prices_df.pct_change()
                    mean_returns = returns.mean()
                    cov_matrix = returns.cov()
                    risk_free_rate = 0.02
                    optimal_portfolio = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
                    optimal_weights = optimal_portfolio['x']
                    std, ret = portfolio_annualised_performance(optimal_weights, mean_returns, cov_matrix)
                    sharpe = (ret - risk_free_rate) / std if std != 0 else 0
                    success_message = "✅ Portfolio Optimized!"

                    st.markdown(f'''
                    <div style="
                        background-color: #e6ffed; 
                        border-left: 5px solid #28a745; 
                        padding: 1rem; 
                        border-radius: 0.25rem; 
                        margin-bottom: 1rem;
                        ">
                        <p style="color: black;"> {success_message}
                        </p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([2,1])
                    with col1:
                        st.markdown("<h2 style='color: black;'>📊 Recommended Allocation</h2>", unsafe_allow_html=True)
                        investment_amounts = [w * st.session_state['user_inputs']['stock_investment_amount'] for w in optimal_weights]
                        annual_returns = mean_returns * 252
                        expected_returns_amt = [(amt * (1 + r)**st.session_state['user_inputs']['investment_horizon']) for amt, r in zip(investment_amounts, annual_returns)]
                        
                        # In recommendation_page, inside the "with col1:" block
                        allocation_df = pd.DataFrame({
                            'Stock Name': prices_df.columns,
                            'Investment Amount': investment_amounts,       # Cleaned column name
                            'Projected Return': expected_returns_amt,     # Cleaned column name
                            'Allocation Percent': [w * 100 for w in optimal_weights] # Use percentage here
                        })
                        st.session_state['stock_df_to_save'] = allocation_df.copy()

                        # Create a separate DataFrame for display with pretty names and formatting
                        display_df = allocation_df.copy()
                        display_df = display_df.rename(columns={
                            'Stock Name': 'Stock Name',
                            'Investment Amount': 'Investment Amount (₹)',
                            'Projected Return': f"Projected Return (₹)"
                        })
                        display_df['Investment Amount (₹)'] = display_df['Investment Amount (₹)'].map('{:,.2f}'.format)
                        display_df[f"Projected Return (₹)"] = display_df[f"Projected Return (₹)"].map('{:,.2f}'.format)
                        st.dataframe(display_df[['Stock Name', 'Investment Amount (₹)', f"Projected Return (₹)"]], use_container_width=True)
                    
                    with col2:
                        st.subheader("📋 Overall Performance")
                        st.metric("Expected Annual Return", f"{ret*100:.2f}%")
                        st.metric("Annual Volatility (Risk)", f"{std*100:.2f}%")
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")

        st.markdown("---")
        st.markdown("<h3 style='color: black;'>💰 Recommended Mutual Fund</h3>", unsafe_allow_html=True)
        num_mf = st.number_input("How many Mutual Funds should we recommend?", 1, 10, 5, 1)
        risk_meter_map = {'Low': ['Low to Moderate Risk'],'Medium': ['Moderate Risk', 'Moderately High Risk'],'High': ['High Risk', 'Very High Risk']}
        eligible_mf_categories = risk_meter_map[st.session_state['risk_profile']]
        eligible_mf_meta = mf_meta[mf_meta['Riskometer'].isin(eligible_mf_categories)]
        recommended_mf = pd.merge(eligible_mf_meta, mf_fc, left_on='Scheme_Name', right_on='Fund Name')
        top_mf = recommended_mf.sort_values(by='Historical_5Y_CAGR (%)', ascending=False).head(num_mf)
        st.markdown(f'<p style="color: black;">Top {num_mf} recommended funds for you based on their historical performance:</p>', unsafe_allow_html=True)
        disclaimer_text = "Disclaimer: Past performance is not indicative of future results. Projections are based on historical data and are for illustrative purposes only."

        st.markdown(f'''
        <div style="
            background-color: #e6f3ff; 
            border-left: 5px solid #1c83e1; 
            padding: 1rem; 
            border-radius: 0.25rem; 
            margin-bottom: 1rem;
            ">
            <p style="color: black;"> ⚠️ {disclaimer_text}
            </p>
        </div>
        ''', unsafe_allow_html=True)
        if not top_mf.empty:
            investment_style = st.session_state['user_inputs']['mf_investment_style']
            total_investment_amount = st.session_state['user_inputs']['mf_investment_amount']
            investment_horizon = st.session_state['user_inputs']['investment_horizon']
            cagr_sum = top_mf['Historical_5Y_CAGR (%)'].sum()
            top_mf['weight'] = top_mf['Historical_5Y_CAGR (%)'] / cagr_sum if cagr_sum > 0 else 1 / len(top_mf)
            fund_names, investment_amounts, expected_returns, total_investments_sip = [], [], [], []
            for index, row in top_mf.iterrows():
                fund_names.append(row['Fund Name'])
                investment_per_fund = total_investment_amount * row['weight']
                if investment_style == 'Lumpsum (One-Time)':
                    investment_amounts.append(investment_per_fund)
                    future_value = investment_per_fund * (1 + (row['Historical_5Y_CAGR (%)'] / 100)) ** investment_horizon
                    expected_returns.append(future_value)
                else: # SIP
                    investment_amounts.append(investment_per_fund)
                    fv, total_inv = calculate_sip_future_value(investment_per_fund, row['Historical_5Y_CAGR (%)'], investment_horizon)
                    expected_returns.append(fv)
                    total_investments_sip.append(total_inv)
            mf_df_data = {'Fund Name': fund_names, 'Lumpsum Investment (₹)' if investment_style == 'Lumpsum (One-Time)' else 'Monthly Investment (₹)': investment_amounts}
            if investment_style == 'Monthly SIP': mf_df_data['Total Contribution (₹)'] = total_investments_sip
            mf_df_data['Projected Return (₹)'] = expected_returns
            mf_allocation_df = pd.DataFrame(mf_df_data)
            top_mf = top_mf.rename(columns={'weight': 'Allocation (%)'})
            st.session_state['mf_df_to_save'] = pd.concat([mf_allocation_df, top_mf['Allocation (%)'].reset_index(drop=True)], axis=1)
            display_mf_df = mf_allocation_df.copy()
            display_mf_df.iloc[:, 1] = display_mf_df.iloc[:, 1].map('{:,.2f}'.format)
            if investment_style == 'Monthly SIP':
                display_mf_df.iloc[:, 2] = display_mf_df.iloc[:, 2].map('{:,.2f}'.format)
                display_mf_df.iloc[:, 3] = display_mf_df.iloc[:, 3].map('{:,.2f}'.format)
            else:
                display_mf_df.iloc[:, 2] = display_mf_df.iloc[:, 2].map('{:,.2f}'.format)
            st.dataframe(display_mf_df, use_container_width=True)
        else:
            st.warning("No mutual funds found in this category.")
        
        st.markdown("---")
        st.markdown("<h2 style='color: black;'>✨ Your Final Portfolio Summary</h2>", unsafe_allow_html=True)
        time_periods = sorted(list(set([1, 3, 5, st.session_state['user_inputs']['investment_horizon']])))
        summary_data = []
        for t in time_periods:
            stock_lumpsum = st.session_state['user_inputs']['stock_investment_amount']
            projected_stock_value = 0
            if 'optimal_weights' in locals() and 'returns' in locals():
                stock_portfolio_return = np.sum((returns.mean() * 252) * optimal_weights)
                projected_stock_value = stock_lumpsum * (1 + stock_portfolio_return) ** t
            elif not candidate_stocks.empty:
                single_stock_cagr = candidate_stocks.iloc[0]['Historical_5Y_CAGR'] / 100
                projected_stock_value = stock_lumpsum * (1 + single_stock_cagr) ** t
            
            projected_mf_value, total_mf_investment = 0, 0
            if not top_mf.empty:
                if st.session_state['user_inputs']['mf_investment_style'] == 'Lumpsum (One-Time)':
                    total_mf_investment = st.session_state['user_inputs']['mf_investment_amount']
                    for _, row in top_mf.iterrows():
                        investment_per_fund = st.session_state['user_inputs']['mf_investment_amount'] * row['Allocation (%)']
                        cagr = row['Historical_5Y_CAGR (%)'] / 100
                        projected_mf_value += investment_per_fund * (1 + cagr) ** t
                else: # SIP
                    for _, row in top_mf.iterrows():
                        sip_per_fund = st.session_state['user_inputs']['mf_investment_amount'] * row['Allocation (%)']
                        cagr = row['Historical_5Y_CAGR (%)']
                        fv, total_inv_per_fund = calculate_sip_future_value(sip_per_fund, cagr, t)
                        projected_mf_value += fv
                        total_mf_investment += total_inv_per_fund
            
            total_invested = stock_lumpsum + total_mf_investment
            total_projected_value = projected_stock_value + projected_mf_value
            total_profit = total_projected_value - total_invested
            percent_gain = (total_profit / total_invested) * 100 if total_invested > 0 else 0
            summary_data.append((f"{t} Years", f"₹{total_invested:,.2f}", f"₹{total_projected_value:,.2f}", f"₹{total_profit:,.2f}", f"{percent_gain:.2f}%"))
        
        summary_df = pd.DataFrame(summary_data, columns=["Investment Period", "Total Investment", "Estimated Value", "Profit Earned", "Return Rate (%)"])
        st.dataframe(summary_df, use_container_width=True)

        save_button_label = "Update Recommendations" if st.session_state.get('edit_mode') else "Save Recommendations"
        if st.button(save_button_label):
            stock_df = st.session_state.get('stock_df_to_save', pd.DataFrame())
            mf_df = st.session_state.get('mf_df_to_save', pd.DataFrame())
            
            # 'summary_df' is already calculated and available here. We now pass it.
            if st.session_state.get('edit_mode'):
                if update_portfolio(edit_data['portfolio_id'], st.session_state['user_id'], st.session_state['user_inputs'], st.session_state['risk_profile'], stock_df, mf_df, summary_df):
                    success_message = "Portfolio updated successfully!"
                    st.markdown(f'''
                    <div style="
                        background-color: #e6ffed; 
                        border-left: 5px solid #28a745; 
                        padding: 1rem; 
                        border-radius: 0.25rem; 
                        margin-bottom: 1rem;
                        ">
                        <p style="color: black;"> {success_message}
                        </p>
                    </div>
                    ''', unsafe_allow_html=True)
                    st.session_state['edit_mode'] = False
                    st.session_state['page'] = "Home"
                    st.rerun()
            else:
                if save_portfolio(st.session_state['user_id'], st.session_state['user_inputs'], st.session_state['risk_profile'], stock_df, mf_df, summary_df):
                    success_message = "Portfolio saved successfully!"
                    st.markdown(f'''
                    <div style="
                        background-color: #e6ffed; 
                        border-left: 5px solid #28a745; 
                        padding: 1rem; 
                        border-radius: 0.25rem; 
                        margin-bottom: 1rem;
                        ">
                        <p style="color: black;"> {success_message}
                        </p>
                    </div>
                    ''', unsafe_allow_html=True)
                    st.session_state['generated_recommendations'] = False
                    st.session_state['page'] = "Home"
                    st.rerun()

def dashboard_page(dashboard_type):
    st.markdown(f"<h2 style='color: black;'>{dashboard_type} Dashboard</h2>", unsafe_allow_html=True)

    portfolio, stocks, mfs, summary_df = get_user_portfolio(st.session_state['user_id'])

    # --- ADD THIS CHECK ---
    # If the user has no portfolio, display a message and stop.
    if portfolio is None:
        warning_message = "You have not created any recommendations yet. Please go to the 'Get Recommendations' page to build your portfolio first."

        st.markdown(f'''
        <div style="
            background-color: #fff8e1; 
            border-left: 5px solid #ffc107; 
            padding: 1rem; 
            border-radius: 0.25rem; 
            margin-bottom: 1rem;
            ">
            <p style="color: black;"> ⚠️ {warning_message}
            </p>
        </div>
        ''', unsafe_allow_html=True)
        return # This stops the function from running further

    # --- THE REST OF YOUR CODE REMAINS THE SAME ---
    if dashboard_type == 'Stock':
        stocks_display = stocks.drop(columns=['portfolio_id', 'stock_record_id'], errors='ignore').rename(columns={'ticker': 'Stock Name', 'invested_amount': 'Invested Amount (₹)', 'expected_return_amount': 'Projected Return (₹)', 'weight': 'Allocation (%)'})
        st.dataframe(stocks_display, use_container_width=True)
    else:
        mfs_display = mfs.drop(columns=['portfolio_id', 'mf_record_id'], errors='ignore').rename(columns={'fund_name': 'Fund Name', 'invested_amount': 'Invested Amount (₹)', 'expected_return_amount': 'Projected Return (₹)', 'total_investment_sip': 'Total SIP Investment (₹)', 'weight': 'Allocation (%)'})
        st.dataframe(mfs_display, use_container_width=True)
    
    urls = {
        "Stock": r"https://app.powerbi.com/view?r=eyJrIjoiNDBkMWQzMzUtYzAwYS00NGE3LTk5NDEtZTI1NzM0MjE4Yjc1IiwidCI6IjE0ZjljNmYzLTIyMGUtNDA4Ni1iYzc5LTFlNjUxZTQwZDZhYiJ9&pageName=8c6c3d2a8437e7888986",
        "MF": r"https://app.fabric.microsoft.com/view?r=eyJrIjoiNDBkMWQzMzUtYzAwYS00NGE3LTk5NDEtZTI1NzM0MjE4Yjc1IiwidCI6IjE0ZjljNmYzLTIyMGUtNDA4Ni1iYzc5LTFlNjUxZTQwZDZhYiJ9" 
    }
    st.markdown(f'<iframe title="{dashboard_type} Dashboard" width="100%" height="600" src="{urls[dashboard_type]}" frameborder="0" allowFullScreen="true"></iframe>', unsafe_allow_html=True)

# --- MAIN APP ROUTER ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login_signup_page()
else:
    # --- UPDATED SIDEBAR LOGIC ---
    with st.sidebar:
        # --- ADD LOGO AT THE TOP ---
        image_url = "https://i.postimg.cc/52hdMrjW/logo4.png" 
        st.markdown(
            f'<img src="{image_url}" class="sidebar-logo">',
            unsafe_allow_html=True,
        )
        st.markdown(f"<h2 style='color: white;'>Welcome, {st.session_state['username']}</h2>", unsafe_allow_html=True)
        st.divider()

        # Helper function to create a styled button and handle page navigation
        def sidebar_button(page_name, label, key):
            is_active = (st.session_state.get('page', 'Home') == page_name)
            button_class = "active-button" if is_active else ""
            
            st.markdown(f'<div class="{button_class}">', unsafe_allow_html=True)
            if st.button(label, use_container_width=True, key=key):
                st.session_state.page = page_name
                st.session_state.edit_mode = False
                st.session_state.generated_recommendations = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # --- NEW: Conditional Button Rendering ---
        if st.session_state.get('role') == 'admin':
            # --- ADMIN VIEW ---
            sidebar_button("Home", "🏠 Admin Home", "admin_home_btn")
            sidebar_button("Admin Panel", "🔑 User Management", "admin_panel_btn")
        else:
            # --- REGULAR USER VIEW ---
            sidebar_button("Home", "🏠 Home", "home_btn")
            sidebar_button("Get Recommendations", "✍️ Get Recommendations", "rec_btn")
            sidebar_button("Stock Dashboard", "📊 Stock Dashboard", "stock_db_btn")
            sidebar_button("MF Dashboard", "📈 MF Dashboard", "mf_db_btn")

        st.divider()
        if st.button("Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # --- PAGE RENDERING LOGIC ---
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Get Recommendations":
        recommendation_page()
    elif st.session_state.page == "Stock Dashboard":
        dashboard_page("Stock")
    elif st.session_state.page == "MF Dashboard":
        dashboard_page("MF")
    elif st.session_state.page == "Admin Panel":
        admin_page()
