import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import mysql.connector
from mysql.connector import Error
import hashlib
import re  # For extracting numeric values from formatted strings

# --- Database Connection and Operations ---

@st.cache_resource
def get_db_connection():
    """Establishes and returns a MySQL database connection."""
    try:
        conn = mysql.connector.connect(
            host=st.secrets["DB_HOST"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            database=st.secrets["DB_NAME"]
        )
        if conn.is_connected():
            return conn
    except Error as e:
        st.error(f"Error connecting to MySQL database: {e}")
        st.info("Please ensure your MySQL server is running and `secrets.toml` is configured correctly.")
        return None

# --- User Authentication Functions ---
def hash_password(password):
    """Hashes a password using SHA256 (for demonstration, use bcrypt/argon2 in production)."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password_hash, provided_password):
    """Verifies a provided password against a stored hash."""
    return stored_password_hash == hashlib.sha256(provided_password.encode()).hexdigest()

def add_user(username, email, password):
    """Adds a new user to the database."""
    conn = get_db_connection()
    if conn is None:
        return False, "Database connection error."

    cursor = conn.cursor()
    try:
        password_hash = hash_password(password)
        insert_user_sql = "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)"
        cursor.execute(insert_user_sql, (username, email, password_hash))
        conn.commit()
        return True, "User registered successfully!"
    except Error as e:
        conn.rollback()
        if "Duplicate entry" in str(e) and "username" in str(e):
            return False, "Username already exists. Please choose a different one."
        elif "Duplicate entry" in str(e) and "email" in str(e):
            return False, "Email already registered. Please use a different one."
        return False, f"Error registering user: {e}"
    finally:
        cursor.close()

def authenticate_user(username, password):
    """Authenticates a user and returns user_id if successful."""
    conn = get_db_connection()
    if conn is None:
        return None, "Database connection error."

    cursor = conn.cursor(dictionary=True)
    try:
        select_user_sql = "SELECT user_id, password_hash FROM users WHERE username = %s"
        cursor.execute(select_user_sql, (username,))
        user_record = cursor.fetchone()

        if user_record:
            if verify_password(user_record['password_hash'], password):
                return user_record['user_id'], "Login successful!"
            else:
                return None, "Invalid username or password."
        else:
            return None, "Invalid username or password."
    except Error as e:
        return None, f"Error during authentication: {e}"
    finally:
        cursor.close()

def delete_recommendation(portfolio_id):
    """Deletes a portfolio recommendation from the database."""
    conn = get_db_connection()
    if conn is None:
        st.error("Database connection error. Cannot delete recommendation.")
        return False
    cursor = conn.cursor()
    try:
        delete_sql = "DELETE FROM portfolios WHERE portfolio_id = %s"
        cursor.execute(delete_sql, (portfolio_id,))
        conn.commit()
        st.success(f"Portfolio ID {portfolio_id} successfully deleted.")
        return True
    except Error as e:
        conn.rollback()
        st.error(f"Error deleting recommendation: {e}")
        return False
    finally:
        cursor.close()

def save_recommendation_to_db(recommendation_data, user_inputs, user_id, portfolio_id=None):
    """Saves or updates the current portfolio recommendation to the database."""
    conn = get_db_connection()
    if conn is None:
        st.error("Failed to get database connection. Cannot save recommendation.")
        return False

    cursor = conn.cursor()
    try:
        if portfolio_id:
            st.info(f"Updating portfolio ID {portfolio_id}...")
            update_portfolio_sql = """
            UPDATE portfolios SET
                investment_amount = %s,
                horizon = %s,
                risk_appetite = %s,
                mf_investment_mode = %s,
                overall_total_invested = %s,
                overall_total_after_horizon = %s,
                overall_total_gain = %s,
                saved_at = CURRENT_TIMESTAMP
            WHERE portfolio_id = %s AND user_id = %s
            """
            cursor.execute(update_portfolio_sql, (
                user_inputs['investment_amount'],
                user_inputs['horizon'],
                user_inputs['risk_appetite'],
                user_inputs['mf_investment_mode'],
                recommendation_data['overall_invested'],
                recommendation_data['overall_after_horizon'],
                recommendation_data['overall_gain'],
                portfolio_id,
                user_id
            ))

            cursor.execute("DELETE FROM portfolio_stocks WHERE portfolio_id = %s", (portfolio_id,))
            cursor.execute("DELETE FROM portfolio_mutual_funds WHERE portfolio_id = %s", (portfolio_id,))
            st.info("Old stock and mutual fund records removed for update.")

        else:
            st.info("Saving new recommendation...")
            insert_portfolio_sql = """
            INSERT INTO portfolios (
                user_id, investment_amount, horizon, risk_appetite, mf_investment_mode,
                overall_total_invested, overall_total_after_horizon, overall_total_gain
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_portfolio_sql, (
                user_id,
                user_inputs['investment_amount'],
                user_inputs['horizon'],
                user_inputs['risk_appetite'],
                user_inputs['mf_investment_mode'],
                recommendation_data['overall_invested'],
                recommendation_data['overall_after_horizon'],
                recommendation_data['overall_gain']
            ))
            portfolio_id = cursor.lastrowid
            st.info(f"New recommendation inserted with Portfolio ID: {portfolio_id}.")

        # Insert stocks
        if not recommendation_data['stocks'].empty:
            insert_stocks_sql = """
            INSERT INTO portfolio_stocks (
                portfolio_id, stock_name, risk_category, cagr, invested_amount, expected_amount_after_horizon
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """
            stock_records = []
            for _, row in recommendation_data['stocks'].iterrows():
                stock_records.append((
                    portfolio_id,
                    row['Name of Stock'],
                    row['Risk Category'],
                    row['CAGR'],
                    row['Invested Amount'],
                    row['Expected Amount After Horizon']
                ))
            cursor.executemany(insert_stocks_sql, stock_records)
            st.info(f"Inserted {cursor.rowcount} stock records.")
        else:
            st.info("No stock recommendations to save.")
        
        # Insert mutual funds
        if not recommendation_data['mutual_funds'].empty:
            insert_mfs_sql = """
            INSERT INTO portfolio_mutual_funds (
                portfolio_id, fund_name, risk_category, cagr_type, cagr_value, invested_amount, expected_amount_after_horizon
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            mf_records = []
            for _, row in recommendation_data['mutual_funds'].iterrows():
                cagr_type_col = 'CAGR' if user_inputs['mf_investment_mode'].lower() == 'lumpsum' else 'SIP CAGR'
                cagr_value = row.get('CAGR') if user_inputs['mf_investment_mode'].lower() == 'lumpsum' else row.get('SIP CAGR')
                if cagr_value is None:
                    st.warning(f"CAGR value not found for mutual fund '{row.get('Name of Fund', 'Unknown Fund')}'. Skipping.")
                    continue
                # Ensure invested_amount is numeric (strip formatting if present)
                invested_amount = row['Invested Amount']
                if isinstance(invested_amount, str):
                    # Extract numeric value from strings like "Monthly ₹14,000.00" or "Lumpsum ₹70,000.00"
                    match = re.search(r'[\d,.]+', invested_amount)
                    if match:
                        invested_amount = float(match.group().replace(',', ''))
                    else:
                        st.warning(f"Invalid invested amount format for fund '{row.get('Name of Fund', 'Unknown Fund')}'. Skipping.")
                        continue
                mf_records.append((
                    portfolio_id,
                    row['Name of Fund'],
                    row['Risk Category'],
                    cagr_type_col,
                    cagr_value,
                    invested_amount,
                    row['Expected Amount After Horizon']
                ))
            if mf_records:
                cursor.executemany(insert_mfs_sql, mf_records)
                st.info(f"Inserted {cursor.rowcount} mutual fund records.")
            else:
                st.info("No mutual fund recommendations to save after filtering.")
        else:
            st.info("No mutual fund recommendations to save.")
        
        conn.commit()
        st.success("Recommendation successfully saved to database!")
        return True
    except Error as e:
        conn.rollback()
        st.error(f"Error saving recommendation to database: {e}")
        return False
    finally:
        cursor.close()

def load_saved_recommendations_from_db(user_id):
    """Loads all saved recommendations for a specific user from the database."""
    conn = get_db_connection()
    if conn is None:
        return []

    cursor = conn.cursor(dictionary=True)
    saved_portfolios = []

    try:
        cursor.execute("SELECT portfolio_id, user_id, investment_amount, horizon, risk_appetite, mf_investment_mode, overall_total_invested, overall_total_after_horizon, overall_total_gain, saved_at FROM portfolios WHERE user_id = %s ORDER BY saved_at DESC", (user_id,))
        portfolios = cursor.fetchall()

        for portfolio in portfolios:
            portfolio_id = portfolio['portfolio_id']
            
            cursor.execute("SELECT stock_name, risk_category, cagr, invested_amount, expected_amount_after_horizon FROM portfolio_stocks WHERE portfolio_id = %s", (portfolio_id,))
            stocks_data = pd.DataFrame(cursor.fetchall())
            if not stocks_data.empty:
                stocks_data = stocks_data.rename(columns={
                    'stock_name': 'Name of Stock',
                    'risk_category': 'Risk Category',
                    'cagr': 'CAGR',
                    'invested_amount': 'Invested Amount',
                    'expected_amount_after_horizon': 'Expected Amount After Horizon'
                })

            cursor.execute("SELECT fund_name, risk_category, cagr_type, cagr_value, invested_amount, expected_amount_after_horizon FROM portfolio_mutual_funds WHERE portfolio_id = %s", (portfolio_id,))
            mfs_data = pd.DataFrame(cursor.fetchall())
            if not mfs_data.empty:
                mfs_data = mfs_data.rename(columns={
                    'fund_name': 'Name of Fund',
                    'risk_category': 'Risk Category',
                    'cagr_type': 'CAGR Type',
                    'cagr_value': 'CAGR',
                    'invested_amount': 'Invested Amount',
                    'expected_amount_after_horizon': 'Expected Amount After Horizon'
                })
                if 'CAGR Type' in mfs_data.columns:
                    # Convert Invested Amount to numeric, handling strings
                    def parse_invested_amount(x):
                        if isinstance(x, (int, float)):
                            return float(x)
                        if isinstance(x, str):
                            match = re.search(r'[\d,.]+', x)
                            if match:
                                return float(match.group().replace(',', ''))
                        return 0.0  # Default if parsing fails
                    mfs_data['Invested Amount'] = mfs_data['Invested Amount'].apply(parse_invested_amount)
                    # Apply formatting only if not already formatted
                    mfs_data.loc[mfs_data['CAGR Type'] == 'SIP CAGR', 'Invested Amount'] = mfs_data.loc[mfs_data['CAGR Type'] == 'SIP CAGR', 'Invested Amount'].apply(lambda x: f"Monthly ₹{x:,.2f}")
                    mfs_data.loc[mfs_data['CAGR Type'] == 'CAGR', 'Invested Amount'] = mfs_data.loc[mfs_data['CAGR Type'] == 'CAGR', 'Invested Amount'].apply(lambda x: f"Lumpsum ₹{x:,.2f}")

            saved_portfolios.append({
                'id': portfolio_id,
                'user_inputs': {
                    'investment_amount': portfolio['investment_amount'],
                    'horizon': portfolio['horizon'],
                    'risk_appetite': portfolio['risk_appetite'],
                    'mf_investment_mode': portfolio['mf_investment_mode']
                },
                'stocks': stocks_data,
                'mutual_funds': mfs_data,
                'overall_invested': portfolio['overall_total_invested'],
                'overall_after_horizon': portfolio['overall_total_after_horizon'],
                'overall_gain': portfolio['overall_total_gain'],
                'saved_at': portfolio['saved_at']
            })
    except Error as e:
        st.error(f"Error loading saved recommendations: {e}")
    finally:
        cursor.close()
    return saved_portfolios

# --- Data Loading Functions ---
@st.cache_data
def load_prediction_data(stock_path=r"final stock output.csv", mutual_fund_path=r"final mutual fund output.csv"):
    """Loads stock and mutual fund prediction data from CSVs."""
    try:
        stocks_df = pd.read_csv(stock_path)
        mutual_funds_df = pd.read_csv(mutual_fund_path)

        stock_column_rename_map = {
            'Symbol': 'Symbol',
            'Company Name': 'Company Name',
            'Prediction Horizon (Years)': 'Prediction Horizon (Years)',
            'As Of Date': 'As Of Date',
            'Predicted Total Return (%)': 'Predicted Total Return (%)',
            'Predicted CAGR (%)': 'Predicted CAGR (%)',
            'Annualized Volatility (%)': 'Annualized Volatility (%)',
            'Latest P/E Ratio': 'Latest P/E Ratio',
            'Latest Beta': 'Latest Beta',
            'Composite Risk Score': 'Composite Risk Score',
            'Risk Band (Composite)': 'Risk Band',
            'Risk Band (Volatility Only)': 'Risk Band (Volatility Only)',
            'Conservative Band': 'Conservative Band',
            'Expected Band': 'Expected Band',
            'Optimistic Band': 'Optimistic Band'
        }
        stocks_df.rename(columns=stock_column_rename_map, inplace=True)

        mutual_fund_column_rename_map = {
            'file': 'file',
            'isin': 'isin',
            'fund_name': 'fund_name',
            'amc': 'amc',
            'category': 'category',
            'fund_category': 'fund_category',
            'years_of_prediction': 'years_of_prediction',
            'rmse': 'rmse',
            'rmse_percent': 'rmse_percent',
            'cagr': 'cagr',
            'sip_cagr': 'sip_cagr',
            'annualized_volatility': 'annualized_volatility',
            'risk_band': 'risk_band'
        }
        mutual_funds_df.rename(columns=mutual_fund_column_rename_map, inplace=True)

        required_stock_cols = ['Company Name', 'Predicted CAGR (%)', 'Risk Band']
        for col in required_stock_cols:
            if col not in stocks_df.columns:
                st.error(f"Critical column '{col}' missing in stock prediction data.")
                return pd.DataFrame(), pd.DataFrame()

        return stocks_df, mutual_funds_df
    except FileNotFoundError as e:
        st.error(f"Error: Could not load prediction data file: {e}.")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading prediction data: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(show_spinner="Loading historical stock data...")
def load_historical_stock_data(historical_stocks_folder_path, combined_parquet_path):
    """Loads historical stock data, prioritizing a combined Parquet file."""
    if os.path.exists(combined_parquet_path):
        try:
            combined_df = pd.read_parquet(combined_parquet_path)
            if 'Date' in combined_df.columns:
                combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')
                combined_df.dropna(subset=['Date'], inplace=True)
            return combined_df
        except Exception as e:
            st.warning(f"Error loading Parquet file ({e}). Falling back to CSVs.")

    all_stocks_data = []
    if not os.path.isdir(historical_stocks_folder_path):
        st.error(f"Error: Historical stock data folder not found at '{historical_stocks_folder_path}'.")
        return pd.DataFrame()

    csv_files = [f for f in os.listdir(historical_stocks_folder_path) if f.endswith(".csv")]
    if not csv_files:
        st.warning(f"No CSV files found in '{historical_stocks_folder_path}'.")
        return pd.DataFrame()

    file_count = 0
    for filename in csv_files:
        file_path = os.path.join(historical_stocks_folder_path, filename)
        try:
            df = pd.read_csv(file_path, usecols=['Date', 'Close', 'Company Name'])
            required_columns = ['Date', 'Close', 'Company Name']
            if not all(col in df.columns for col in required_columns):
                st.warning(f"Skipping file '{filename}': Missing required columns.")
                continue

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df.dropna(subset=['Date', 'Close'], inplace=True)

            all_stocks_data.append(df)
            file_count += 1
        except Exception as e:
            st.warning(f"Skipping file '{filename}' due to error: {e}")
            continue
    
    if not all_stocks_data:
        st.warning(f"No valid CSV files found in '{historical_stocks_folder_path}'.")
        return pd.DataFrame()

    combined_df = pd.concat(all_stocks_data, ignore_index=True)
    
    try:
        combined_df.to_parquet(combined_parquet_path, index=False)
        st.info(f"Saved combined CSVs to Parquet: {combined_parquet_path}")
    except Exception as e:
        st.warning(f"Could not save combined data to Parquet: {e}")

    return combined_df

@st.cache_data(show_spinner="Loading historical mutual fund data...")
def load_historical_mf_data(combined_mf_parquet_path):
    """Loads historical mutual fund data from a combined Parquet file."""
    if os.path.exists(combined_mf_parquet_path):
        try:
            combined_df = pd.read_parquet(combined_mf_parquet_path)
            if 'date' in combined_df.columns and 'nav' in combined_df.columns and 'fund_name' in combined_df.columns:
                combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
                combined_df['nav'] = pd.to_numeric(combined_df['nav'], errors='coerce')
                combined_df.dropna(subset=['date', 'nav', 'fund_name'], inplace=True)
                return combined_df
            else:
                st.warning("Parquet file missing required columns: 'date', 'nav', 'fund_name'.")
                return pd.DataFrame()
        except Exception as e:
            st.warning(f"Error loading mutual fund Parquet file ({e}).")
            return pd.DataFrame()
    else:
        st.error(f"Mutual fund Parquet file not found at '{combined_mf_parquet_path}'.")
        return pd.DataFrame()

# --- Core Recommendation Logic Functions ---

def get_risk_allocation(risk_appetite):
    """Determines investment allocation between stocks and mutual funds."""
    if risk_appetite.lower() == 'low':
        return {'stocks': 0.30, 'mutual_funds': 0.70}
    elif risk_appetite.lower() == 'medium':
        return {'stocks': 0.50, 'mutual_funds': 0.50}
    elif risk_appetite.lower() == 'high':
        return {'stocks': 0.70, 'mutual_funds': 0.30}
    else:
        st.warning("Invalid risk appetite. Using medium allocation.")
        return {'stocks': 0.50, 'mutual_funds': 0.50}

def calculate_future_value_lumpsum(principal, cagr, horizon_years):
    """Calculates the future value of a lump sum investment."""
    if cagr is None:
        cagr = 0
    return principal * (1 + cagr / 100) ** horizon_years

def calculate_future_value_sip(monthly_sip, cagr, horizon_years):
    """Calculates the future value of an SIP investment."""
    if cagr is None:
        cagr = 0
    annual_rate = cagr / 100
    total_months = horizon_years * 12

    if annual_rate == 0:
        return monthly_sip * total_months

    monthly_rate = (1 + annual_rate)**(1/12) - 1
    fv = monthly_sip * (((1 + monthly_rate)**total_months - 1) / monthly_rate) * (1 + monthly_rate)
    return fv

def recommend_portfolio(
    total_investment_amount,
    horizon,
    risk_appetite,
    mutual_fund_mode,
    num_stocks_to_invest,
    num_mutual_funds_to_invest,
    stocks_df,
    mutual_funds_df
):
    """Generates stock and mutual fund recommendations based on user input."""
    recommended_stocks = pd.DataFrame()
    total_invested_stocks = 0.0
    total_after_horizon_stocks = 0.0
    total_gain_stocks = 0.0

    recommended_mutual_funds = pd.DataFrame()
    total_invested_mutual_funds = 0.0
    total_after_horizon_mutual_funds = 0.0
    total_gain_mutual_funds = 0.0

    overall_invested = 0.0
    overall_after_horizon = 0.0
    overall_gain = 0.0

    allocation = get_risk_allocation(risk_appetite)
    investment_for_stocks = total_investment_amount * allocation['stocks']
    investment_for_mutual_funds = total_investment_amount * allocation['mutual_funds']

    risk_band = risk_appetite.capitalize()

    # --- Stock Recommendations (always lump sum) ---
    if num_stocks_to_invest > 0 and stocks_df is not None and not stocks_df.empty:
        filtered_stocks = stocks_df[stocks_df['Risk Band'] == risk_band].copy()
        if filtered_stocks.empty:
            st.warning(f"No stocks found for '{risk_band}' risk category.")
        else:
            filtered_stocks = filtered_stocks.sort_values(by='Predicted CAGR (%)', ascending=False)
            selected_stocks = filtered_stocks.head(num_stocks_to_invest).copy()

            if not selected_stocks.empty:
                cagr_sum = selected_stocks['Predicted CAGR (%)'].sum()
                if cagr_sum > 0:
                    selected_stocks['Weight'] = selected_stocks['Predicted CAGR (%)'] / cagr_sum
                else:
                    selected_stocks['Weight'] = 1 / len(selected_stocks)
                    st.warning("All selected stock CAGRs are zero. Distributing investment equally.")

                selected_stocks['Invested Amount'] = selected_stocks['Weight'] * investment_for_stocks
                selected_stocks['Expected Amount After Horizon'] = selected_stocks.apply(
                    lambda row: calculate_future_value_lumpsum(row['Invested Amount'], row['Predicted CAGR (%)'], horizon),
                    axis=1
                )
                recommended_stocks = selected_stocks[[
                    'Company Name', 'Risk Band', 'Predicted CAGR (%)', 'Invested Amount', 'Expected Amount After Horizon'
                ]].rename(columns={'Predicted CAGR (%)': 'CAGR', 'Risk Band': 'Risk Category', 'Company Name': 'Name of Stock'})
                
                total_invested_stocks = recommended_stocks['Invested Amount'].sum()
                total_after_horizon_stocks = recommended_stocks['Expected Amount After Horizon'].sum()
                total_gain_stocks = total_after_horizon_stocks - total_invested_stocks
            else:
                st.warning(f"Not enough stocks for '{risk_band}' to recommend {num_stocks_to_invest} stocks.")

    # --- Mutual Fund Recommendations ---
    if num_mutual_funds_to_invest > 0 and mutual_funds_df is not None and not mutual_funds_df.empty:
        filtered_mutual_funds = mutual_funds_df[mutual_funds_df['risk_band'] == risk_band].copy()
        if filtered_mutual_funds.empty:
            st.warning(f"No mutual funds found for '{risk_band}' risk category.")
        else:
            cagr_column = 'sip_cagr' if mutual_fund_mode.lower() == 'monthly sip' else 'cagr'
            if cagr_column not in filtered_mutual_funds.columns:
                st.error(f"Error: '{cagr_column}' not found in mutual fund data.")
                if cagr_column == 'sip_cagr' and 'cagr' in filtered_mutual_funds.columns:
                    cagr_column = 'cagr'
                    st.info("Falling back to 'cagr'.")
                elif cagr_column == 'cagr' and 'sip_cagr' in filtered_mutual_funds.columns:
                    cagr_column = 'sip_cagr'
                    st.info("Falling back to 'sip_cagr'.")
                else:
                    st.error(f"Neither 'cagr' nor 'sip_cagr' found.")
                    return (pd.DataFrame(), 0.0, 0.0, 0.0, pd.DataFrame(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

            filtered_mutual_funds = filtered_mutual_funds.sort_values(by=cagr_column, ascending=False)
            selected_mutual_funds = filtered_mutual_funds.head(num_mutual_funds_to_invest).copy()

            if not selected_mutual_funds.empty:
                cagr_sum_mf = selected_mutual_funds[cagr_column].sum()
                if cagr_sum_mf > 0:
                    selected_mutual_funds['Weight'] = selected_mutual_funds[cagr_column] / cagr_sum_mf
                else:
                    selected_mutual_funds['Weight'] = 1 / len(selected_mutual_funds)
                    st.warning("All selected mutual fund CAGRs are zero. Distributing investment equally.")

                if mutual_fund_mode.lower() == 'monthly sip':
                    monthly_sip_total = investment_for_mutual_funds
                    selected_mutual_funds['Invested Amount'] = selected_mutual_funds['Weight'] * monthly_sip_total
                    selected_mutual_funds['Expected Amount After Horizon'] = selected_mutual_funds.apply(
                        lambda row: calculate_future_value_sip(row['Invested Amount'], row[cagr_column], horizon),
                        axis=1
                    )
                    recommended_mutual_funds = selected_mutual_funds[[
                        'fund_name', 'risk_band', cagr_column, 'Invested Amount', 'Expected Amount After Horizon'
                    ]].rename(columns={cagr_column: 'SIP CAGR', 'risk_band': 'Risk Category', 'fund_name': 'Name of Fund'})
                    total_invested_mutual_funds = recommended_mutual_funds['Invested Amount'].sum() * 12 * horizon
                else:
                    selected_mutual_funds['Invested Amount'] = selected_mutual_funds['Weight'] * investment_for_mutual_funds
                    selected_mutual_funds['Expected Amount After Horizon'] = selected_mutual_funds.apply(
                        lambda row: calculate_future_value_lumpsum(row['Invested Amount'], row[cagr_column], horizon),
                        axis=1
                    )
                    recommended_mutual_funds = selected_mutual_funds[[
                        'fund_name', 'risk_band', cagr_column, 'Invested Amount', 'Expected Amount After Horizon'
                    ]].rename(columns={cagr_column: 'CAGR', 'risk_band': 'Risk Category', 'fund_name': 'Name of Fund'})
                    total_invested_mutual_funds = recommended_mutual_funds['Invested Amount'].sum()

                total_after_horizon_mutual_funds = recommended_mutual_funds['Expected Amount After Horizon'].sum()
                total_gain_mutual_funds = total_after_horizon_mutual_funds - total_invested_mutual_funds
            else:
                st.warning(f"Not enough mutual funds for '{risk_band}' to recommend {num_mutual_funds_to_invest} funds.")

    overall_invested = total_invested_stocks + total_invested_mutual_funds
    overall_after_horizon = total_after_horizon_stocks + total_after_horizon_mutual_funds
    overall_gain = overall_after_horizon - overall_invested

    return (
        recommended_stocks, total_invested_stocks, total_after_horizon_stocks, total_gain_stocks,
        recommended_mutual_funds, total_invested_mutual_funds, total_after_horizon_mutual_funds, total_gain_mutual_funds,
        overall_invested, overall_after_horizon, overall_gain
    )

# --- Streamlit Pages ---

def login_signup_page():
    """Displays the login/signup page."""
    st.title("Welcome to AlphaVue!")
    st.markdown("**Your Investment Journey Starts Here.**", unsafe_allow_html=True)
    st.info("Please log in or sign up to access the application.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Login")
        username_login = st.text_input("Username", key="username_login")
        password_login = st.text_input("Password", type="password", key="password_login")
        if st.button("Login", key="login_button"):
            if username_login and password_login:
                user_id, message = authenticate_user(username_login, password_login)
                if user_id:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id
                    st.session_state.page = 'Home'
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Please enter both username and password.")

    with col2:
        st.subheader("Sign Up")
        username_signup = st.text_input("Username", key="username_signup")
        email_signup = st.text_input("Email", key="email_signup")
        password_signup = st.text_input("Password", type="password", key="password_signup")
        confirm_password_signup = st.text_input("Confirm Password", type="password", key="confirm_password_signup")
        if st.button("Sign Up", key="signup_button"):
            if not username_signup or not email_signup or not password_signup or not confirm_password_signup:
                st.error("All fields are required.")
            elif password_signup != confirm_password_signup:
                st.error("Passwords do not match.")
            else:
                success, message = add_user(username_signup, email_signup, password_signup)
                if success:
                    st.success(message + " You can now log in.")
                else:
                    st.error(message)

def home_page():
    """Displays the homepage including saved recommendations and market details."""
    st.header("🏠 Home Page")
    st.markdown("Welcome to AlphaVue!")

    st.markdown("---")
    st.subheader("Today's Key Market Indices")

    st.metric(label="Nifty 50", value=f"₹{random.uniform(18000, 22000):,.2f}", delta=f"{random.uniform(-100, 100):,.2f}")
    st.metric(label="Sensex", value=f"₹{random.uniform(60000, 75000):,.2f}", delta=f"{random.uniform(-300, 300):,.2f}")

    st.markdown("---")
    st.subheader("Your Saved Recommendations")

    if 'editing_portfolio' not in st.session_state:
        st.session_state.editing_portfolio = None
    if 'deleting_portfolio_id' not in st.session_state:
        st.session_state.deleting_portfolio_id = None

    if st.session_state.deleting_portfolio_id:
        portfolio_id_to_delete = st.session_state.deleting_portfolio_id
        st.warning(f"Are you sure you want to delete Portfolio ID: **{portfolio_id_to_delete}**?")
        col_confirm_del, col_cancel_del = st.columns(2)
        with col_confirm_del:
            if st.button("Confirm Delete", key=f"confirm_delete_{portfolio_id_to_delete}"):
                if delete_recommendation(portfolio_id_to_delete):
                    st.session_state.deleting_portfolio_id = None
                    st.rerun()
        with col_cancel_del:
            if st.button("Cancel Delete", key=f"cancel_delete_{portfolio_id_to_delete}"):
                st.session_state.deleting_portfolio_id = None
                st.rerun()

    if 'user_id' in st.session_state and st.session_state.user_id:
        saved_recommendations = load_saved_recommendations_from_db(st.session_state.user_id)
        if saved_recommendations:
            for i, rec in enumerate(saved_recommendations):
                st.write(f"**Portfolio {rec['id']} (Saved on: {rec['saved_at'].strftime('%Y-%m-%d %H:%M:%S')})**")
                st.info(f"Initial Investment Input: ₹{rec['user_inputs']['investment_amount']:,.2f} | Horizon: {rec['user_inputs']['horizon']} years | Risk: {rec['user_inputs']['risk_appetite']} | MF Mode: {rec['user_inputs']['mf_investment_mode']}")
                
                if not rec['stocks'].empty:
                    st.write("Stocks:")
                    st.dataframe(rec['stocks'], hide_index=True)
                if not rec['mutual_funds'].empty:
                    st.write("Mutual Funds:")
                    st.dataframe(rec['mutual_funds'], hide_index=True)
                
                st.success(f"**Overall Total Invested (Cumulative):** ₹{rec['overall_invested']:,.2f}")
                st.success(f"**Overall Total Expected After Horizon:** ₹{rec['overall_after_horizon']:,.2f}")
                st.success(f"**Overall Total Gain:** ₹{rec['overall_gain']:,.2f}")
                
                col_edit, col_delete = st.columns(2)
                with col_edit:
                    if st.button("Edit", key=f"edit_rec_{rec['id']}"):
                        st.session_state.editing_portfolio = rec
                        st.session_state.page = 'Recommendation'
                        st.rerun()
                with col_delete:
                    if st.button("Delete", key=f"delete_rec_{rec['id']}"):
                        st.session_state.deleting_portfolio_id = rec['id']
                        st.rerun()
                st.markdown("---")
        else:
            st.info("No recommendations saved yet. Generate and save portfolios on the 'Recommendation' page.")
    else:
        st.warning("Please log in to view your saved recommendations.")

def recommendation_page(stocks_df, mutual_funds_df):
    """Displays portfolio recommendation inputs and outputs."""
    st.header("📈 Portfolio Recommendations")
    st.markdown("Enter your investment preferences and get personalized portfolio recommendations.")

    recommended_stocks_df = pd.DataFrame()
    stock_total_invested = 0.0
    stock_total_after_horizon = 0.0
    stock_total_gain = 0.0

    recommended_mf_df = pd.DataFrame()
    mf_total_invested = 0.0
    mf_total_after_horizon = 0.0
    mf_total_gain = 0.0

    is_editing = st.session_state.editing_portfolio is not None
    portfolio_to_edit = st.session_state.editing_portfolio

    initial_investment_amount = 100000.0
    initial_horizon = 5
    initial_risk_appetite = 'Medium'
    initial_mf_investment_mode = 'lumpsum'
    initial_num_stocks = 5
    initial_num_mutual_funds = 5

    if is_editing and portfolio_to_edit:
        st.subheader(f"Editing Portfolio ID: {portfolio_to_edit['id']}")
        initial_investment_amount = float(portfolio_to_edit['user_inputs']['investment_amount'])
        initial_horizon = int(portfolio_to_edit['user_inputs']['horizon'])
        initial_risk_appetite = portfolio_to_edit['user_inputs']['risk_appetite']
        initial_mf_investment_mode = portfolio_to_edit['user_inputs']['mf_investment_mode']
        initial_num_stocks = len(portfolio_to_edit['stocks']) if not portfolio_to_edit['stocks'].empty else 1
        initial_num_mutual_funds = len(portfolio_to_edit['mutual_funds']) if not portfolio_to_edit['mutual_funds'].empty else 1
        st.warning("Adjust the values below and click 'Update Recommendation' to save changes.")
        if st.button("Cancel Edit", key="cancel_edit_button"):
            st.session_state.editing_portfolio = None
            st.rerun()

    st.subheader("Your Investment Preferences")
    col1, col2 = st.columns(2)

    with col1:
        investment_amount = st.number_input(
            "Total Investment Amount (in currency)",
            min_value=1000.0,
            value=initial_investment_amount,
            step=1000.0,
            key="investment_amount_input"
        )
        horizon = st.number_input(
            "Investment Horizon (Years)",
            min_value=1,
            value=initial_horizon,
            step=1,
            key="horizon_input"
        )
        risk_appetite = st.selectbox(
            "Risk Appetite",
            options=['Low', 'Medium', 'High'],
            index=['Low', 'Medium', 'High'].index(initial_risk_appetite),
            key="risk_select"
        )

    with col2:
        mf_investment_mode = st.selectbox(
            "Mutual Fund Investment Mode",
            options=['lumpsum', 'monthly sip'],
            index=['lumpsum', 'monthly sip'].index(initial_mf_investment_mode),
            key="mf_mode_select"
        )
        max_stocks = len(stocks_df) if stocks_df is not None and not stocks_df.empty else 100
        max_mfs = len(mutual_funds_df) if mutual_funds_df is not None and not mutual_funds_df.empty else 100

        num_stocks = st.number_input(
            "Number of Stocks to Recommend",
            min_value=1,
            max_value=max_stocks,
            value=min(initial_num_stocks, max_stocks),
            step=1,
            key="num_stocks_input"
        )
        num_mutual_funds = st.number_input(
            "Number of Mutual Funds to Recommend",
            min_value=1,
            max_value=max_mfs,
            value=min(initial_num_mutual_funds, max_mfs),
            step=1,
            key="num_mfs_input"
        )

    # Calculate allocation for display
    allocation = get_risk_allocation(risk_appetite)
    investment_for_stocks = investment_amount * allocation['stocks']
    investment_for_mutual_funds = investment_amount * allocation['mutual_funds']

    button_label = "Update Recommendation" if is_editing else "Get Recommendations"
    if st.button(button_label):
        if investment_amount <= 0 or horizon <= 0 or num_stocks <= 0 or num_mutual_funds <= 0:
            st.error("Please ensure all input values are positive and valid.")
        elif stocks_df.empty or mutual_funds_df.empty:
            st.error("Cannot generate recommendations. Stock or Mutual Fund prediction data is empty.")
        else:
            st.header("Generating Your Personalized Portfolio...")

            (
                recommended_stocks_df, stock_total_invested, stock_total_after_horizon, stock_total_gain,
                recommended_mf_df, mf_total_invested, mf_total_after_horizon, mf_total_gain,
                overall_total_invested, overall_total_after_horizon, overall_total_gain
            ) = recommend_portfolio(
                total_investment_amount=investment_amount,
                horizon=horizon,
                risk_appetite=risk_appetite,
                mutual_fund_mode=mf_investment_mode,
                num_stocks_to_invest=num_stocks,
                num_mutual_funds_to_invest=num_mutual_funds,
                stocks_df=stocks_df,
                mutual_funds_df=mutual_funds_df
            )

            st.session_state.recommended_stock_names = recommended_stocks_df['Name of Stock'].tolist() if not recommended_stocks_df.empty else []
            st.session_state.recommended_mf_names = recommended_mf_df['Name of Fund'].tolist() if not recommended_mf_df.empty else []

            st.session_state['current_recommendation_for_save'] = {
                'stocks': recommended_stocks_df,
                'mutual_funds': recommended_mf_df,
                'overall_invested': overall_total_invested,
                'overall_after_horizon': overall_total_after_horizon,
                'overall_gain': overall_total_gain,
            }
            st.session_state['current_user_inputs_for_save'] = {
                'investment_amount': investment_amount,
                'horizon': horizon,
                'risk_appetite': risk_appetite,
                'mf_investment_mode': mf_investment_mode
            }

            st.markdown("---")
            st.write(f"### 📊 Investment Allocation")
            stock_allocation_percent = (investment_for_stocks / investment_amount) * 100 if investment_amount > 0 else 0
            mf_allocation_percent = (investment_for_mutual_funds / investment_amount) * 100 if investment_amount > 0 else 0
            allocation_note = f"Note: Mutual fund allocation is monthly for SIP mode, leading to a cumulative investment of ₹{mf_total_invested:,.2f} over {horizon} years." if mf_investment_mode.lower() == 'monthly sip' else ""
            st.info(f"**Allocated to Stocks (Lump Sum):** ₹{investment_for_stocks:,.2f} ({stock_allocation_percent:.1f}%)  "
                    f"|  **Allocated to Mutual Funds ({'Monthly SIP' if mf_investment_mode.lower() == 'monthly sip' else 'Lump Sum'}):** ₹{investment_for_mutual_funds:,.2f} ({mf_allocation_percent:.1f}%)  {allocation_note}")
            st.markdown("---")

            st.header("📈 Stock Recommendations")
            if not recommended_stocks_df.empty:
                st.dataframe(recommended_stocks_df, hide_index=True)
                st.markdown(f"**Total Invested in Stocks (Lump Sum):** ₹{stock_total_invested:,.2f}")
                st.markdown(f"**Total Expected After Horizon (Stocks):** ₹{stock_total_after_horizon:,.2f}")
                st.markdown(f"**Total Gain (Stocks):** ₹{stock_total_gain:,.2f}")
            else:
                st.warning("No stock recommendations to display. Adjust inputs or check prediction data.")

            st.markdown("---")

            st.header("📊 Mutual Fund Recommendations")
            if not recommended_mf_df.empty:
                display_mf_df = recommended_mf_df.copy()
                if mf_investment_mode.lower() == 'monthly sip':
                    display_mf_df['Invested Amount'] = display_mf_df['Invested Amount'].apply(lambda x: f"Monthly ₹{x:,.2f}")
                else:
                    display_mf_df['Invested Amount'] = display_mf_df['Invested Amount'].apply(lambda x: f"₹{x:,.2f}")
                st.dataframe(display_mf_df, hide_index=True)
                st.markdown(f"**Total Invested in Mutual Funds ({'Cumulative' if mf_investment_mode.lower() == 'monthly sip' else 'Lump Sum'}):** ₹{mf_total_invested:,.2f}")
                st.markdown(f"**Total Expected After Horizon (Mutual Funds):** ₹{mf_total_after_horizon:,.2f}")
                st.markdown(f"**Total Gain (Mutual Funds):** ₹{mf_total_gain:,.2f}")
            else:
                st.warning("No mutual fund recommendations to display. Adjust inputs or check prediction data.")

            st.markdown("---")

            st.header("💰 Overall Portfolio Summary")
            st.markdown(f"**Overall Total Invested (Cumulative):** ₹{overall_total_invested:,.2f}")
            st.markdown(f"**Overall Total Expected After Horizon:** ₹{overall_total_after_horizon:,.2f}")
            st.markdown(f"**Total Gain:** ₹{overall_total_gain:,.2f}")
            st.markdown("---")

    if st.button("Save/Update Recommendation to Database"):
        if 'current_recommendation_for_save' not in st.session_state:
            st.warning("Please click 'Get Recommendations' first to generate a portfolio to save.")
        else:
            is_stocks_empty = st.session_state['current_recommendation_for_save']['stocks'].empty
            is_mfs_empty = st.session_state['current_recommendation_for_save']['mutual_funds'].empty
            if not is_stocks_empty or not is_mfs_empty:
                if 'user_id' in st.session_state:
                    portfolio_id = portfolio_to_edit['id'] if is_editing else None
                    if save_recommendation_to_db(
                        st.session_state['current_recommendation_for_save'],
                        st.session_state['current_user_inputs_for_save'],
                        st.session_state.user_id,
                        portfolio_id=portfolio_id
                    ):
                        st.session_state.editing_portfolio = None
                        st.session_state.page = 'Home'
                        st.rerun()
                else:
                    st.error("You must be logged in to save/update recommendations.")
            else:
                st.warning("No valid recommendation to save. Ensure results are generated.")

def stock_analysis_page(stocks_historical_df):
    """Displays line charts of historical close prices for stocks."""
    st.header("📉 Stock Analysis")
    st.markdown("Here you can analyze the historical price trends of various stocks.")

    if stocks_historical_df is None or stocks_historical_df.empty:
        st.warning("No historical data available. Check folder path and ensure files have 'Date', 'Close', 'Company Name' columns.")
        return

    if 'recommended_stock_names' in st.session_state and st.session_state.recommended_stock_names:
        display_companies = [
            company for company in st.session_state.recommended_stock_names
            if company in stocks_historical_df['Company Name'].values
        ]
        if not display_companies:
            st.info("No historical data found for recommended stocks. Ensure company names match.")
            st.markdown("---")
            st.info("Generate a recommendation on the 'Recommendation' page to see charts.")
            return

        st.info("Showing historical data for your recommended stocks.")
        selected_companies = st.multiselect(
            "Select Recommended Stocks for Analysis",
            options=display_companies,
            default=display_companies
        )

        if selected_companies:
            for company_name in selected_companies:
                st.subheader(f"{company_name} - Close Price Trend")
                df_to_plot = stocks_historical_df[stocks_historical_df['Company Name'] == company_name].set_index('Date')
                if 'Close' in df_to_plot.columns and not df_to_plot.empty:
                    st.line_chart(df_to_plot['Close'])
                else:
                    st.warning(f"No 'Close' price data for {company_name}.")
                st.markdown("---")
        else:
            st.info("No recommended stocks selected.")
    else:
        st.info("Generate a portfolio recommendation first on the 'Recommendation' page.")
        st.markdown("---")

def mf_analysis_page(mf_historical_df):
    """Displays line charts of historical NAV for mutual funds."""
    st.header("📊 Mutual Fund Analysis")
    st.markdown("Here you can analyze the historical NAV trends of your recommended mutual funds.")

    if mf_historical_df is None or mf_historical_df.empty:
        st.warning("No historical mutual fund data available. Check Parquet file path and ensure it has 'date', 'nav', 'fund_name' columns.")
        return

    if 'recommended_mf_names' in st.session_state and st.session_state.recommended_mf_names:
        display_funds = [
            fund for fund in st.session_state.recommended_mf_names
            if fund in mf_historical_df['fund_name'].values
        ]
        if not display_funds:
            st.info("No historical data found for recommended mutual funds. Ensure fund names match.")
            st.markdown("---")
            st.info("Generate a recommendation on the 'Recommendation' page to see charts.")
            return

        st.info("Showing historical NAV data for your recommended mutual funds.")
        selected_funds = st.multiselect(
            "Select Recommended Mutual Funds for Analysis",
            options=display_funds,
            default=display_funds
        )

        if selected_funds:
            for fund_name in selected_funds:
                st.subheader(f"{fund_name} - NAV Trend")
                df_to_plot = mf_historical_df[mf_historical_df['fund_name'] == fund_name].set_index('date')
                if 'nav' in df_to_plot.columns and not df_to_plot.empty:
                    st.line_chart(df_to_plot['nav'])
                else:
                    st.warning(f"No 'nav' data for {fund_name}.")
                st.markdown("---")
        else:
            st.info("No recommended mutual funds selected.")
    else:
        st.info("Generate a portfolio recommendation first on the 'Recommendation' page.")
        st.markdown("---")

# --- Streamlit Application Main Execution ---
st.set_page_config(layout="wide", page_title="Portfolio Recommender")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'page' not in st.session_state:
    st.session_state.page = 'Login'
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'recommended_stock_names' not in st.session_state:
    st.session_state.recommended_stock_names = []
if 'recommended_mf_names' not in st.session_state:
    st.session_state.recommended_mf_names = []

COMBINED_STOCK_PARQUET_FILE = r"combined_historical_stocks.parquet"
COMBINED_MF_PARQUET_FILE = r"all_mutual_funds.parquet"
stocks_prediction_df, mutual_funds_prediction_df = load_prediction_data()
stocks_historical_folder = r"nse_stock_data"
stocks_historical_df = load_historical_stock_data(
    historical_stocks_folder_path=stocks_historical_folder,
    combined_parquet_path=COMBINED_STOCK_PARQUET_FILE
)
mf_historical_df = load_historical_mf_data(COMBINED_MF_PARQUET_FILE)

if not st.session_state.logged_in:
    login_signup_page()
else:
    with st.sidebar:
        st.image(r"AlphaVue.png")
        st.title("Navigation")
        if st.button("🏠 Home"):
            st.session_state.page = 'Home'
        if st.button("📈 Recommendation"):
            st.session_state.page = 'Recommendation'
        if st.button("📉 Stock Analysis"):
            st.session_state.page = 'Stock Analysis'
        if st.button("📊 MF Analysis"):
            st.session_state.page = 'MF Analysis'
        st.markdown(
            """
            <a href="https://app.powerbi.com/view?r=eyJrIjoiNmE2ZWFlZGUtMjc2NC00M2E0LTgzMzAtODE5OTMxYjlkNDEzIiwidCI6IjE0ZjljNmYzLTIyMGUtNDA4Ni1iYzc5LTFlNjUxZTQwZDZhYiJ9" target="_blank" style="display: inline-block; padding: 8px 16px; background-color: #4CAF50; color: white; text-align: center; text-decoration: none; border-radius: 4px;">📊 MF Dashboard</a>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <a href="https://app.powerbi.com/view?r=eyJrIjoiOTU2MjY5ZjgtNmVmZS00NDQ3LTk1OGUtNTY0OGRjN2UyODA0IiwidCI6IjE0ZjljNmYzLTIyMGUtNDA4Ni1iYzc5LTFlNjUxZTQwZDZhYiJ9" target="_blank" style="display: inline-block; padding: 8px 16px; background-color: #2196F3; color: white; text-align: center; text-decoration: none; border-radius: 4px;">📈 Stock Dashboard</a>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.page = 'Login'
            st.session_state.recommended_stock_names = []
            st.session_state.recommended_mf_names = []
            st.success("Successfully logged out!")
            st.rerun()

    if st.session_state.page == 'Home':
        home_page()
    elif st.session_state.page == 'Recommendation':
        if stocks_prediction_df.empty or mutual_funds_prediction_df.empty:
            st.warning("Prediction data not loaded. Ensure CSV paths are correct.")
        else:
            recommendation_page(stocks_prediction_df, mutual_funds_prediction_df)
    elif st.session_state.page == 'Stock Analysis':
        stock_analysis_page(stocks_historical_df)
    elif st.session_state.page == 'MF Analysis':
        mf_analysis_page(mf_historical_df)
