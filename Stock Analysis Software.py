#Stock Analysis Software
import tkinter as tk
from tkinter import ttk
import requests
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sklearn.linear_model as lm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report, mean_squared_error, mean_absolute_error, r2_score 
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"  

def getWeeklyData ():
    stockName = input("What is the name of the companies stock ")
    key = input("What is your api key for alphavantage ")
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol='  + stockName + '&apikey=' + key
    r = requests.get(url)
    data = r.json()
    data = data["Weekly Adjusted Time Series"]
    return formatData(data)

def getDailyData ():
    stockName = input("What is the name of the companies stock ")
    key = input("What is your api key for alphavantage ")
    url = url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol='  + stockName + '&apikey=' + key
    r = requests.get(url)
    data = r.json()
    data = data["Daily Adjusted Time Series"]
    return formatData(data)

def getMonthlyData ():
    stockName = input("What is the name of the companies stock ")
    key = input("What is your api key for alphavantage ")
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol=' + stockName + '&apikey=' + key
    r = requests.get(url)
    data = r.json()
    data = data["Monthly Adjusted Time Series"]
    return formatData(data)

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

def logisticReg(df):
    X = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    y = (df["Close"].shift(-1) > df["Close"]).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.80, test_size=0.2, shuffle=False  
    )
    model = lm.LogisticRegression()
    model = model.fit(X_train, y_train)
    pred = model.predict(X_test)
    eval_for_class_models(pred, y_test)


def decisionTreeClass(df):
    X = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    y = (df["Close"].shift(-1) > df["Close"]).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.80, test_size=0.2, shuffle=False  
    )
    model = DecisionTreeClassifier()
    model = model.fit(X_train,y_train)
    pred = model.predict(X_test)
    eval_for_class_models(pred, y_test)

def decisionTreeClass(df):
    X = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    y = df["Close"].shift(-1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.80, test_size=0.2, shuffle=False  
    )
    model = DecisionTreeRegressor()
    model = model.fit(X_train,y_train)
    pred = model.predict(X_test)
    eval_for_reg_models(pred, y_test)

def linearReg(df):
    X = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    y = df["Close"].shift(-1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.80, test_size=0.2, shuffle=False  
    )
    model = lm.LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    eval_for_reg_models(pred, y_test)

def eval_for_reg_models(pred, test):
    print("MSE:", mean_squared_error(test, pred))
    print("MAE:", mean_absolute_error(test, pred))
    print("R²:", r2_score(test, pred))

def eval_for_class_models(pred, test):
    print("Accuracy:", accuracy_score(test, pred))
    print(classification_report(test, pred))
    print("Confusion Matrix:\n", confusion_matrix(test, pred))

class StockGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stock Analysis App")
        self.geometry("600x400")

        # Container that holds all pages stacked
        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        # Make the container stretch with the window
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Create pages and keep references in a dict
        self.pages = {}
        for Page in (FirstPage, SecondPage, ThirdPage, ModelPage, VisPage, SelectionPage):
            page = Page(parent=container, controller=self)
            self.pages[Page.__name__] = page
            page.grid(row=0, column=0, sticky="nsew")  # stacked

        self.show_page("FirstPage")

    def show_page(self, name: str):
        """Raise the requested page to the top."""
        self.pages[name].tkraise()


class FirstPage(tk.Frame):
    def __init__(self, parent, controller: StockGUI):
        super().__init__(parent)
        tk.Label(self, text="Welcome", font=("Arial", 18)).pack(pady=20)
        tk.Button(self, text="Next",
                  command=lambda: controller.show_page("SecondPage")
                  ).pack(pady=10)


class SecondPage(tk.Frame):
    def __init__(self, parent, controller: StockGUI):
        super().__init__(parent)
        tk.Label(self, text="Second Screen", font=("Arial", 18)).pack(pady=20)
        tk.Button(self, text="Next",
                  command=lambda: controller.show_page("ThirdPage")
                  ).pack(pady=10)
        tk.Button(self, text="Back",
                  command=lambda: controller.show_page("FirstPage")
                  ).pack(pady=10)

class ThirdPage(tk.Frame):
    def __init__(self, parent, controller: StockGUI):
        super().__init__(parent)
        tk.Label(self, text="Third Screen", font=("Arial", 18)).pack(pady=20)
        tk.Button(self, text="Model",
                  command=lambda: controller.show_page("ModelPage")
                  ).pack(pady=10)
        tk.Button(self, text="Visualistaion",
                  command=lambda: controller.show_page("VisPage")
                  ).pack(pady=10)
        tk.Button(self, text="Back",
                  command=lambda: controller.show_page("SecondPage")
                  ).pack(pady=10)

class ModelPage(tk.Frame):
    def __init__(self, parent, controller: StockGUI):
        super().__init__(parent)
        tk.Label(self, text="Model Screen", font=("Arial", 18)).pack(pady=20)
        tk.Button(self, text="Next",
                  command=lambda: controller.show_page("FirstPage")
                  ).pack(pady=10)
        tk.Button(self, text="Back",
                  command=lambda: controller.show_page("ThirdPage")
                  ).pack(pady=10)

class VisPage(tk.Frame):
    def __init__(self, parent, controller: StockGUI):
        super().__init__(parent)
        tk.Label(self, text="Vis Screen", font=("Arial", 18)).pack(pady=20)
        tk.Button(self, text="Next",
                  command=lambda: controller.show_page("SelectionPage")
                  ).pack(pady=10)
        tk.Button(self, text="Back",
                  command=lambda: controller.show_page("ThirdPage")
                  ).pack(pady=10)

class SelectionPage(tk.Frame):
    def __init__(self, parent, controller: StockGUI):
        super().__init__(parent)
        tk.Label(self, text="Selection Screen", font=("Arial", 18)).pack(pady=20)
        tk.Button(self, text="Next",
                  command=lambda: controller.show_page("FirstPage")
                  ).pack(pady=10)
        tk.Button(self, text="Back",
                  command=lambda: controller.show_page("VisPage")
                  ).pack(pady=10)


if __name__ == "__main__":
    app = StockGUI()
    app.mainloop()
