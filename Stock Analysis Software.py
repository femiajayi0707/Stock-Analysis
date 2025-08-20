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

data = []

def getMonthlyData ():
    stockName = input("What is the name of the companies stock")
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol=' + stockName + '&apikey=OYKT17X1F6JCW060'
    r = requests.get(url)
    data = r.json()

def getWeeklyData ():
    stockName = input("What is the name of the companies stock")
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol='  + stockName + '&apikey=OYKT17X1F6JCW060'
    r = requests.get(url)
    data = r.json()

def getDailyData ():
    stockName = input("What is the name of the companies stock")
    url = url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol='  + stockName + '&apikey=OYKT17X1F6JCW060'
    r = requests.get(url)
    data = r.json()
