#Stock Analysis Software
import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sklearn.linear_model as lm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score, classification_report
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"  

def getWeeklyData ():
    stockName = input("What is the name of the companies stock ")
    key = input("What is your api key for alphavantage")
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol='  + stockName + '&apikey=' + key
    r = requests.get(url)
    data = r.json()
    data = data["Weekly Adjusted Time Series"]
    return data

def getDailyData ():
    stockName = input("What is the name of the companies stock ")
    key = input("What is your api key for alphavantage")
    url = url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol='  + stockName + '&apikey=' + key
    r = requests.get(url)
    data = r.json()
    data = data["Daily Adjusted Time Series"]
    return data

def getMonthlyData ():
    stockName = input("What is the name of the companies stock ")
    key = input("What is your api key for alphavantage")
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol=' + stockName + '&apikey=' + key
    r = requests.get(url)
    data = r.json()
    data = data["Monthly Adjusted Time Series"]
    return data

def formatData(data):
    df = pd.DataFrame.from_dict(data, orient='index')
    column_rename = {
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. adjusted close": "Adjusted Close",
        "6. volume": "Volume",
        "7. dividend amount": "Dividend Amount",
        "8. split coefficient": "Split Coefficient",
    }
    df = df.rename(columns=column_rename)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    for col in df.columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])

    return df

def display(df):
    print("The display")
    display(df)

def describeData(df):
    print("\n The Describe")
    print(df.describe())

def dataInfo(df):
    print("\n The info")
    print(df.info)

def lineChart(df,column):
    df[column].plot(title=column, figsize=(10,5))
    plt.show()

def barChart(df,column):
    df[column].plot(kind="bar", figsize=(12,4), title="Monthly column")
    plt.tight_layout()
    plt.show()

def histogram(df,column):
    (df[column].pct_change() * 100).hist(bins=50)
    plt.tight_layout()
    plt.show()

def boxPlot(df,column):
    plot = df[column].pct_change()*100
    plt.boxplot(plot.dropna())
    plt.show()

def scatter(df,column_a,column_b):
    plt.scatter(df["Volume"], df["Close"].pct_change()*100, alpha=0.5)
    plt.show()

def Ohlc(df):
    fig = go.Figure(data=go.Ohlc(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    ))
    fig.show()

