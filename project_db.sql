-- Connect to your portfolio_recommender database first
USE portfolio_project1;

-- Drop existing tables to facilitate schema change (THIS WILL DELETE ALL YOUR DATA!)
DROP TABLE IF EXISTS portfolio_stocks;
DROP TABLE IF EXISTS portfolio_mutual_funds;
DROP TABLE IF EXISTS portfolios;
DROP TABLE users; -- Drop users table too, to ensure a clean start if needed

-- Create the users table
CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL, -- Storing hashed passwords
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Recreate the portfolios table with a user_id foreign key
CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL, -- New column to link to users table
    investment_amount DECIMAL(15, 2) NOT NULL,
    horizon INT NOT NULL,
    risk_appetite VARCHAR(50) NOT NULL,
    mf_investment_mode VARCHAR(50) NOT NULL,
    overall_total_invested DECIMAL(15, 2),
    overall_total_after_horizon DECIMAL(15, 2),
    overall_total_gain DECIMAL(15, 2),
    saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE -- Link to users table
);

-- Recreate portfolio_stocks table with cascade delete
CREATE TABLE IF NOT EXISTS portfolio_stocks (
    stock_record_id INT AUTO_INCREMENT PRIMARY KEY,
    portfolio_id INT NOT NULL,
    stock_name VARCHAR(255) NOT NULL,
    risk_category VARCHAR(50),
    cagr DECIMAL(10, 2),
    invested_amount DECIMAL(15, 2),
    expected_amount_after_horizon DECIMAL(15, 2),
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id) ON DELETE CASCADE
);

-- Recreate portfolio_mutual_funds table with cascade delete
CREATE TABLE IF NOT EXISTS portfolio_mutual_funds (
    mf_record_id INT AUTO_INCREMENT PRIMARY KEY,
    portfolio_id INT NOT NULL,
    fund_name VARCHAR(255) NOT NULL,
    risk_category VARCHAR(50),
    cagr_type VARCHAR(50), -- To store 'CAGR' or 'SIP CAGR'
    cagr_value DECIMAL(10, 2),
    invested_amount DECIMAL(15, 2),
    expected_amount_after_horizon DECIMAL(15, 2),
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id) ON DELETE CASCADE
);


select * from users;
select * from portfolios;
select * from portfolio_stocks;
select * from portfolio_stocks;