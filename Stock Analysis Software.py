#Stock Analysis Software
import requests

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
import sklearn.linear_model as lm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def getWeeklyData ():
    stockName = input("What is the name of the companies stock ")
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol='  + stockName + '&apikey=OYKT17X1F6JCW060'
    r = requests.get(url)
    data = r.json()
    data = data["Weekly Adjusted Time Series"]
    return data

def getDailyData ():
    stockName = input("What is the name of the companies stock ")
    url = url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol='  + stockName + '&apikey=OYKT17X1F6JCW060'
    r = requests.get(url)
    data = r.json()
    data = data["Daily Adjusted Time Series"]
    return data

def getMonthlyData ():
    stockName = input("What is the name of the companies stock ")
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol=' + stockName + '&apikey=OYKT17X1F6JCW060'
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
    return df
