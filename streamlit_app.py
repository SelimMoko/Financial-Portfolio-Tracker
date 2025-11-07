import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')
import yfinance as yf
import mplfinance as mpf
from scipy.stats import norm
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
elts=["Open","Close","High","Low"]


# Page Configuration
st.set_page_config(
    page_title="Get hostorical data",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded")

def add_sharp_ratio(df,rolling):
    df['SMA'] = df['Close'].rolling(window=rolling).mean()
    df['STD'] = df['Close'].rolling(window=rolling).std()
    df['Return'] = df["Close"][0]-df['SMA']
    df["sharp ratio"]=df['Return']/(df['STD'])

def get_global_return(df):
    return " (return: " + str(np.round(100*(df["Close"][-1]-df["Close"][0])/df["Close"][-1],2)) +"%)"

#return the matplotlib plot of the specified data_frame
def plot_candle_matplotlib(df_input,rolling,Name,lines):
    #create the dataframe first
    df = pd.DataFrame({})
    for parameter in elts:
        df[parameter]=df_input[parameter]    
        additionals=[]
    if lines:
        df['SMA'] = df['Close'].rolling(window=rolling).mean()
        df['STD'] = df['Close'].rolling(window=rolling).std()
        k = 1
        df['UpperBand'] = df['SMA'] + (k * df['STD'])
        df['LowerBand'] = df['SMA'] - (k * df['STD'])
        additionals=[mpf.make_addplot(df['SMA'], color='red'),
        mpf.make_addplot(df['UpperBand'], color='red'),
        mpf.make_addplot(df['LowerBand'], color='red')]
    if add_sharp:
        add_sharp_ratio(df,rolling)
        additionals.append(mpf.make_addplot(df["sharp ratio"], color="red"))

    fig, axes = mpf.plot(
        df,
        #volume=True,
        type='candle',
        style='charles',
        addplot=additionals,
        title=Name + get_global_return(df), 
        ylabel='Price',
        returnfig=True 
    )
    return fig

#Calculate the coef. to apply to a specified index depending on the initial amount invested
def get_coef(price,name_data,name):
    return (float(price)/name_data["Close",name][0])

#get stock list and its corresponding amount in a dictionnary
def stocks_amount(txt):
    dico={}
    for k  in txt.split(" "):
        name=k.split("-")[0]
        amount=k.split("-")[1]
        dico[name]=amount
    return dico

#Return a dataframe containing a weighted portfolio historical data (Yahoo format)
def build_portfolio_df(tickers_prices , start_date, end_date):    
    tickers=""
    general_data_raw=pd.DataFrame({})
    for k in elts:
        general_data_raw[k]=[]
    wallet=stocks_amount(tickers_prices)
    for stock_name in wallet:
        amount=wallet[stock_name]
        stock_data=yf.download(stock_name, start_date, end_date)
        coefficient=get_coef(amount,stock_data,stock_name)
        for k in elts:
            general_data_raw[k]=(coefficient*stock_data[k][stock_name]).add(general_data_raw[k], fill_value=0)
    return general_data_raw

# Sidebar for User Inputs
with st.sidebar:
    st.title('📈 Portfolio tracker')
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/selim-mokobodzki-27b738230/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">Selim Mokobodzki</a>', unsafe_allow_html=True)
    tickers_prices = st.text_input('Enter "stock tickers-amount invested at the beginning" separated by space', 'AAPL-25 GOOG-80 SPY-25')
    start_date = st.date_input('Start date', value=pd.to_datetime('2025-01-01'))
    end_date = st.date_input('End date', value=pd.to_datetime('today'))
    rolling_window = st.slider('Rolling window in days', min_value=1, max_value=80, value=20)
    add_lines=st.checkbox("add moving average and volatility")
    add_sharp=st.checkbox("add sharp ratio")
    plot_all = st.button('Plot all components of your portfolio')

def plot_all_stocks(tickers_prices,start_date,end_date,rolling_window):
    wallet=stocks_amount(tickers_prices)
    
    portfolio_data=build_portfolio_df(tickers_prices,start_date,end_date)
    graph=plot_candle_matplotlib(portfolio_data,rolling_window,"Portfolio history" ,add_lines)
    # col= st.columns(1)
    # with col:
    st.pyplot(graph)
    #get the list of name of stocks
    L=[]
    for k in wallet:
        L.append(k)

    #create a grid to display a graph for each component of the wallet
    # Define number of columns per row
    cols_per_row = 2
    
    for stock_index in range(0,len(L),cols_per_row):
        cols= st.columns(cols_per_row)
        for j,col in enumerate(cols):
            if stock_index+j<len(L):
                df=yf.download(L[j+stock_index],start_date, end_date)
                graph=plot_candle_matplotlib(df,rolling_window,L[j+stock_index],add_lines)
                with col:
                    st.pyplot(graph)

if plot_all:
    plot_all_stocks(tickers_prices,start_date,end_date,rolling_window)