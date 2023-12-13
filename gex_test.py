import streamlit as st
import yfinance as yf
import requests
from datetime import datetime, timedelta 
from polygon import RESTClient
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from scipy.stats import norm
import pandas_market_calendars as mcal
from scipy.optimize import brentq
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.implied_volatility import implied_volatility
import math
import pytz

client = RESTClient(api_key=st.secrets["API_KEY"])

def get_sofr_rate():
    # Define the URL for the latest rates in JSON format
    url = 'https://markets.newyorkfed.org/api/rates/all/latest.json'
    
    # Make a GET request to fetch the rates
    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise an HTTPError if the response was an error
        
        # Parse the JSON response
        data = response.json()
        
        # Look for the SOFR rate in the response data
        for rate in data.get("refRates", []):
            if rate.get("type") == "SOFR":
                return rate.get("percentRate")
        
        print("SOFR rate not found.")
        return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # Python 3.6
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Unexpected error occurred: {req_err}")

    return None

###################################
###################################

def is_market_open():
    # Define NYSE market hours (9:30 AM to 4:00 PM ET)
    market_open = datetime.now(pytz.timezone('US/Eastern')).replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = market_open.replace(hour=16, minute=0, second=0, microsecond=0)

    # Get current time in ET
    current_time = datetime.now(pytz.timezone('US/Eastern'))

    # Check if current time is within market hours
    return market_open <= current_time <= market_close

###################################
###################################

def days_till_expiration(row, todayDate, nyse):
    # Convert 'Exp Date' from string to datetime
    expiration_date = pd.to_datetime(row['Exp Date'])

    # Check for same-day expiration
    if expiration_date.date() == todayDate.date():
        return 1/262
    else:
        # Get all the business days
        bus_days = nyse.valid_days(start_date=todayDate, end_date=expiration_date)
        return len(bus_days) / 262

######################################################################################################
######################################################################################################

# Function to calculate Black-Scholes option price and Greeks
def black_scholes_greeks(S, K, T, r, sigma, flag='c'):
    gamma = None
    delta = None

    # Ensure that S, K, and sigma are non-zero and T is non-negative
    if S <= 0 or K <= 0 or sigma <= 0 or T < 0:
        return 0, 0  # Return zero for both gamma and delta
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if flag == 'c':
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    elif flag == 'p':
        delta = -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
     # If flag is not 'call' or 'put', raise an error
    else:
        raise ValueError("Invalid option type. Use 'c' or 'p'.")
    
    # Check if gamma was set, if not, this is an unexpected case
    if gamma is None or delta is None:
        raise ValueError(f"Gamma or Delta not calculated. Flag was {flag}")

    
    return gamma, delta

def calculate_implied_volatility(contract_price, S, K, T, r, option_type):
    """
    Calculate implied volatility.
    
    :param contract_price: Market price of the option
    :param S: Current price of the underlying
    :param K: Strike price
    :param T: Time to expiration in years
    :param r: Risk-free rate
    :param option_type: 'c' for call, 'p' for put
    :return: Implied volatility
    """

    # Check if the market price is very close to zero
    if np.isclose(contract_price, 0.0):
        return 0.0

    # Brent method to find the root of the Black-Scholes price - market price
    try:
        implied_vol = brentq(lambda x: black_scholes(option_type, S, K, T, r, x) - contract_price, 1e-6, 1)
    except ValueError:
        # Return a NaN if the calculation fails
        implied_vol = np.nan

    return implied_vol

###################################
###################################

# Function to calculate the Black-Scholes value of a call or put option
def bs_option_price(S, K, T, r, sigma, option_type='c'):
    """
    Calculate the Black-Scholes option price for a call or put option.

    Parameters:
    S (float): Underlying asset price.
    K (float): Option strike price.
    T (float): Time to expiration in years.
    r (float): Risk-free interest rate (annual rate).
    sigma (float): Volatility of the underlying asset (annual rate).
    option_type (str): 'call' for call option, 'p' for put option.

    Returns:
    float: The Black-Scholes option price.
    """
    # Calculate d1 and d2 parameters
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate the call option price
    if option_type == 'c':
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    # Calculate the put option price
    elif option_type == 'p':
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price
    else:
        raise ValueError("Invalid option type. Use 'c' or 'p'.")

###################################
###################################

def fetch_stock_price(symbol):
    # Check if the symbol is empty
    if not symbol.strip():
        st.error("No symbol provided. Please enter a ticker symbol.")
        return None, None
    
    try:
        url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
        params = {"apiKey": st.secrets["API_KEY"]}
        response = requests.get(url, params=params)
        response.raise_for_status()  # This will raise an HTTPError for non-200 status
        data = response.json()

       # Extracting last close and previous day's close
        last_close = data['ticker']['day']['c'] if ('ticker' in data and 'day' in data['ticker'] and 'c' in data['ticker']['day']) else None
        prev_day_close = data['ticker']['prevDay']['c'] if ('ticker' in data and 'prevDay' in data['ticker'] and 'c' in data['ticker']['prevDay']) else None

        return last_close
    
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
        return None
    except requests.exceptions.ConnectionError as conn_err:
        st.error(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        st.error(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"Unexpected error occurred: {req_err}")

    return None

###################################
###################################

def get_option_expirations(symbol):
    ticker = yf.Ticker(symbol)
    return ticker.options  # This returns a tuple of expiration dates

def get_option_chain(symbol, from_date, to_date):
    params = {}
    if from_date:
        params["expiration_date.gte"] = from_date
    if to_date:
        params["expiration_date.lte"] = to_date

    options_chain_data = client.list_snapshot_options_chain(symbol, params=params)

    options_chain = [
        option for option in options_chain_data if option.underlying_asset.ticker == symbol
    ]
    return options_chain

###################################
###################################

def fetch_option_data(symbol, from_date, to_date):
    # Fetching and processing option chain data
    option_chain_data = get_option_chain(symbol, from_date, to_date)
    if not option_chain_data:  # Check if the option chain data is empty
        st.warning(f"No options data available for {symbol}. Please enter a different symbol.")
        return None
    return option_chain_data

###################################
###################################

def convert_options_chain_to_df(options_chain, stock_price, todayDate, nyse):
    # Initialize the list to hold processed option data
    processed_data = []
    market_is_open = is_market_open()

    # Calculate the range of strike prices within +/- 15% of the stock_price
    fromStrike = 0.85 * stock_price
    toStrike = 1.15 * stock_price

    # Process each option in the options chain
    for option in options_chain:
        # Filter out options that don't fall within the specified strike range
        if fromStrike <= option.details.strike_price <= toStrike:
            contract_price = option.day.open if market_is_open else option.day.close
    
            # Extract the required fields from each option
            processed_data.append({
                'Exp Date': option.details.expiration_date,
                'Type': 'Call' if option.details.contract_type == 'call' else 'Put',
                'Strike': round(option.details.strike_price, 2),
                'Open Interest': option.open_interest if option.open_interest is not None else 0,
                'IV': round(option.implied_volatility, 2) if option.implied_volatility is not None else 0,
                'Contract Price': contract_price
            })

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(processed_data)

    # Ensure 'Exp Date' is the correct datetime type
    df['Exp Date'] = pd.to_datetime(df['Exp Date'])
    df['days2Exp'] = df.apply(lambda row: days_till_expiration(row, todayDate, nyse), axis=1)

    # Split the DataFrame into two separate DataFrames for calls and puts
    calls_df = df[df['Type'] == 'Call']
    puts_df = df[df['Type'] == 'Put']

    return calls_df, puts_df

###################################
###################################

def setup_sidebar():
    # Sidebar setup with input field
    symbol = st.sidebar.text_input("Ticker Symbol", placeholder="SPY", value="", max_chars=5, key="sym")
    get_data_button = st.sidebar.button('Get Data')

    return symbol, get_data_button

###################################
###################################

def main():
    # Create a market calendar
    nyse = mcal.get_calendar('NYSE')
    todayDate = datetime.now()

    symbol, is_data_btn_clicked = setup_sidebar()
    if is_data_btn_clicked and symbol:
        stock_price = fetch_stock_price(symbol.upper())
        st.write(stock_price)
        stock_price = math.floor(stock_price)

        # Fetch expirations
        expirations = get_option_expirations(symbol)

        if expirations:
            from_expiration = expirations[0]
            to_expiration = expirations[-1]

            option_chain_data = fetch_option_data(symbol, from_expiration, to_expiration)
            calls, puts = convert_options_chain_to_df(option_chain_data, stock_price, todayDate, nyse)
            
            # Create range of underlying prices -- this is for SPY. For SPX need to use +50/-50
            min_price = stock_price - 50
            max_price = stock_price + 50
            strike_range = np.arange(min_price, max_price + 1, 1)

            # get daily interest rate
            rate = get_sofr_rate()

            # Initialize a DataFrame to store call gammas
            call_gamma_df = pd.DataFrame()

            # Calculate gamma exposure (GEX) for calls
            for index, call in calls.iterrows():
                # Skip options with missing 'greeks' or 'IV' data
                if 'IV' not in call or np.isnan(call['IV']):
                    continue

                S = stock_price  # Current stock price
                K = call['Strike']  # Strike price
                T = call['days2Exp'] / 365.25  # Time to expiration in years
                option_type = 'c' if call['Type'] == 'Call' else 'p'

                if call['IV'] not in [0, 0.0, np.nan]:
                    thisVol = call['IV']
                else:
                    thisVol = 0.1  # Placeholder for implied volatility

                contract_price = call['Contract Price']
                if np.isnan(contract_price):
                    contract_price = bs_option_price(
                        S=S, 
                        K=K, 
                        T=T, 
                        r=rate, 
                        sigma=thisVol, 
                        option_type=option_type
                    )

                # If IV was zero, now calculate thisVol using the obtained contract_price
                if call['IV'] in [0, 0.0]:
                    thisVol = calculate_implied_volatility(contract_price, S, K, T, rate, option_type)

                thisRate = rate * call['days2Exp']
                
                # Using the calculated or existing IV in the Black-Scholes Greeks function
                gammas, deltas = black_scholes_greeks(S, K, T, thisRate, thisVol, flag=option_type)

                OI = call['Open Interest']

                temp_df = pd.DataFrame({
                    'Gamma': gammas,
                    'OI': OI,
                    'underlyingClose': strike_range,
                    'Strike': call['Strike'],
                    'direction': 1
                })

                call_gamma_df = pd.concat([call_gamma_df, temp_df]) 
        
            # Calculate the GEX for calls aggregated by underlying price
            call_gamma_df['calls'] = call_gamma_df['Gamma'] * 100 * call_gamma_df['OI'] * call_gamma_df['underlyingClose']**2 * 0.01 * call_gamma_df['direction']
            #call_gamma_df.to_csv("call_gamma_df2.csv")
            call_gex = call_gamma_df.groupby('underlyingClose')['calls'].sum().reset_index()

###################################
###################################

if __name__ == "__main__":
    main()


        