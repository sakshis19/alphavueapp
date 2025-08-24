CREATE DATABASE IF NOT EXISTS alphavue;
USE alphavue;

DROP TABLE IF EXISTS portfolio_summary;
DROP TABLE IF EXISTS portfolio_stocks;
DROP TABLE IF EXISTS portfolio_mutual_funds;
DROP TABLE IF EXISTS portfolios;
DROP TABLE IF EXISTS users;

CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL, 
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    age INT NOT NULL,
    experience VARCHAR(50) NOT NULL,
    primary_goal VARCHAR(100) NOT NULL,
    market_reaction VARCHAR(100) NOT NULL,
    stock_investment_amount DECIMAL(15, 2) NOT NULL,
    mf_investment_amount DECIMAL(15, 2) NOT NULL,
    horizon INT NOT NULL,
    risk_appetite VARCHAR(50) NOT NULL,
    mf_investment_mode VARCHAR(50) NOT NULL,
    saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE 
);

CREATE TABLE IF NOT EXISTS portfolio_stocks (
    stock_record_id INT AUTO_INCREMENT PRIMARY KEY,
    portfolio_id INT NOT NULL,
    ticker VARCHAR(50) NOT NULL,
    invested_amount DECIMAL(15, 2),
    expected_return_amount DECIMAL(15, 2),
    weight DECIMAL(6, 2),
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS portfolio_mutual_funds (
    mf_record_id INT AUTO_INCREMENT PRIMARY KEY,
    portfolio_id INT NOT NULL,
    fund_name VARCHAR(255) NOT NULL,
    invested_amount DECIMAL(15, 2),
    expected_return_amount DECIMAL(15, 2),
    total_investment_sip DECIMAL(15, 2), 
    weight DECIMAL(6, 2),
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id) ON DELETE CASCADE
);

CREATE TABLE portfolio_summary (
    summary_id INT AUTO_INCREMENT PRIMARY KEY,
    portfolio_id INT,
    investment_period VARCHAR(255),
    total_investment VARCHAR(255),
    estimated_value VARCHAR(255),
    profit_earned VARCHAR(255),
    return_rate VARCHAR(255),
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id) ON DELETE CASCADE
);

select * from users;
select * from portfolios;
select * from portfolio_stocks;
select * from portfolio_mutual_funds;
select * from portfolio_summary;

