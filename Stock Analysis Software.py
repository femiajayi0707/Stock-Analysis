#Stock Analysis Software
#Importing the necessary modules 
import tkinter as tk
from tkinter import ttk
import requests
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as lm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report, mean_squared_error, mean_absolute_error, r2_score 
from tkhtmlview import HTMLLabel 
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"  
import pygame

#Defining the Parent Class
class StockGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stock Analysis App")
        self.geometry("600x600")

        self.choice = None

        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.pages = {}
        for Page in (FirstPage, SecondPage, ThirdPage, ModelPage, VisPage):
            page = Page(parent=container, controller=self)
            self.pages[Page.__name__] = page
            page.grid(row=0, column=0, sticky="nsew")  # stacked

        self.show_page("FirstPage")
        
    def set_choice(self, text):
        self.choice = text
    def get_choice(self):
        return self.choice

    def show_page(self, name: str):
        self.pages[name].tkraise()

class FirstPage(tk.Frame):
    def __init__(self, parent, controller: StockGUI):
        super().__init__(parent)
        tk.Label(self, text="Welcome to the Stock Analysis and Prediction Tool", font=("Arial", 18)).pack(pady=20)
        change = tk.Frame(self)
        change.pack(padx=6)
        tk.Button(change, text="Next",
                  command=lambda: controller.show_page("SecondPage")
                  ).pack(side='left', padx = 20, pady=15)

class SecondPage(tk.Frame):
    def __init__(self, parent, controller: StockGUI):
        super().__init__(parent)
        self.controller = controller
        tk.Label(self, text="Welcome", font=("Arial", 18)).pack(pady=20)
        questions = tk.Frame(self)
        questions.pack(pady=5)
        tk.Label(questions, text="What is the name of the companies stock ", font=("Arial", 18)).pack(pady=20)
        self.name = tk.Entry(questions)
        self.name.pack()
        tk.Label(questions, text="What is your api key for alphavantage ", font=("Arial", 18)).pack(pady=20)
        self.api = tk.Entry(questions)
        self.api.pack()

        data = tk.Frame(self)
        data.pack(pady=12)
        tk.Button(data, text="Daily Data",
                  command=lambda: self.getDailyData()
                  ).pack(side="left",padx=20)
        tk.Button(data, text="Weekly Data",
                  command=lambda: self.getWeeklyData()
                  ).pack(side="left",padx=6)
        tk.Button(data, text="Monthly Data",
                  command=lambda: self.getMonthlyData()
                  ).pack(side="left",padx=6)
        change = tk.Frame(self)
        change.pack(padx=6)
        tk.Button(change, text="Back",
                  command=lambda: controller.show_page("FirstPage")
                  ).pack(side='right', padx = 20, pady=15)
        tk.Button(change, text="Next",
                  command=lambda: controller.show_page("ThirdPage")
                  ).pack(side='left', padx = 20, pady=15)

    def getWeeklyData(self):
        stockName = self.name.get()
        key = self.api.get()
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol='  + stockName + '&apikey=' + key
        r = requests.get(url)
        data = r.json()
        data = data["Weekly Adjusted Time Series"]
        tk.Label(self, text="Weekly Data has been selected", font=("Arial", 18)).pack(pady=20)
        self.controller.set_choice(self.formatData(data))

    def getDailyData(self):
        stockName = self.name.get()
        key = self.api.get()
        url = url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol='  + stockName + '&apikey=' + key
        r = requests.get(url)
        data = r.json()
        data = data["Daily Adjusted Time Series"]
        tk.Label(self, text="Daily data has been selected", font=("Arial", 18)).pack(pady=20)
        self.controller.set_choice(self.formatData(data))

    def getMonthlyData(self):
        stockName = self.name.get()
        key = self.api.get()
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol=' + stockName + '&apikey=' + key
        r = requests.get(url)
        data = r.json()
        data = data["Monthly Adjusted Time Series"]
        tk.Label(self, text="Monthly data has been selected", font=("Arial", 18)).pack(pady=20)
        self.controller.set_choice(self.formatData(data))

    def formatData(self, data):
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

class ThirdPage(tk.Frame):
    def __init__(self, parent, controller: StockGUI):
        super().__init__(parent)
        tk.Label(self, text="Choose One: Do you wish to Create a Model or do you seek visualisation of the data", font=("Arial", 18)).pack(pady=20)
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
        tk.Button(self, text="Back",
                  command=lambda: controller.show_page("ThirdPage")
                  ).pack(pady=10)
        
        tk.Label(self, text="Classification Models", font=("Arial", 18)).pack(pady=20)
        tk.Button(self, text="Logistic Regression Model",
                  command=lambda: self.logisticReg(controller.get_choice())
                  ).pack(pady=10)
        tk.Button(self, text="Decision Tree Classifier Model",
                  command=lambda: self.decisionTreeClass(controller.get_choice())
                  ).pack(pady=10)
        
        tk.Label(self, text="Regression Models", font=("Arial", 18)).pack(pady=20)
    
        tk.Button(self, text="Decision Tree Regression Model",
                  command=lambda: self.decisionTreeReg(controller.get_choice())
                  ).pack(pady=10)
        tk.Button(self, text="Linear Regression Model",
                  command=lambda: self.linearReg(controller.get_choice())
                  ).pack(pady=10)

    def logisticReg(self, df):
        X = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        y = (df["Close"].shift(-1) > df["Close"]).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.80, test_size=0.2, shuffle=False  
        )
        model = lm.LogisticRegression()
        model = model.fit(X_train, y_train)
        pred = model.predict(X_test)
        self.eval_for_class_models(pred, y_test)

    def decisionTreeClass(self, df):
        X = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        y = (df["Close"].shift(-1) > df["Close"]).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.80, test_size=0.2, shuffle=False  
        )
        model = DecisionTreeClassifier()
        model = model.fit(X_train,y_train)
        pred = model.predict(X_test)
        self.eval_for_class_models(pred, y_test)

    def decisionTreeReg(self, df):
        X = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        y = df["Close"].shift(-1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.80, test_size=0.2, shuffle=False  
        )
        model = DecisionTreeRegressor()
        model = model.fit(X_train,y_train)
        pred = model.predict(X_test)
        self.eval_for_reg_models(pred, y_test)

    def linearReg(self, df):
        X = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        y = df["Close"].shift(-1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.80, test_size=0.2, shuffle=False  
        )
        model = lm.LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        self.eval_for_reg_models(pred, y_test)

    def eval_for_reg_models(self, pred, test):
        tk.Label(self, text=("MSE:", mean_squared_error(test, pred)), font=("Arial", 18)).pack(pady=20)
        tk.Label(self, text=("MAE:", mean_absolute_error(test, pred)), font=("Arial", 18)).pack(pady=20)
        tk.Label(self, text=("R²:", r2_score(test, pred)), font=("Arial", 18)).pack(pady=20)


    def eval_for_class_models(self, pred, test):
        tk.Label(self, text=("Accuracy:", accuracy_score(test, pred)), font=("Arial", 18)).pack(pady=20)
        tk.Label(self, text=(classification_report(test, pred,zero_division=0)), font=("Arial", 18)).pack(pady=20)
        tk.Label(self, text=("Confusion Matrix:\n", confusion_matrix(test, pred)), font=("Arial", 18)).pack(pady=20)

class VisPage(tk.Frame):
    def __init__(self, parent, controller: StockGUI):
        super().__init__(parent)
        tk.Label(self, text="Visualisation Screen", font=("Arial", 18)).pack(pady=20)
        tk.Button(self, text="Back",
                  command=lambda: controller.show_page("ThirdPage")
                  ).pack(pady=10)
        
        tk.Button(self, text="Display",
                  command=lambda: self.display(controller.get_choice())
                  ).pack(pady=10)
        tk.Button(self, text="Describe",
                  command=lambda: self.describeData(controller.get_choice())
                  ).pack(pady=10)
        tk.Button(self, text="Information",
                  command=lambda: self.dataInfo(controller.get_choice())
                  ).pack(pady=10)
        tk.Button(self, text="Line Chart",
                  command=lambda: self.lineChart(controller.get_choice())
                  ).pack(pady=10)
        tk.Button(self, text="Bar Chart",
                  command=lambda: self.barChart(controller.get_choice())
                  ).pack(pady=10)
        tk.Button(self, text="Histogram",
                  command=lambda: self.histogram(controller.get_choice())
                  ).pack(pady=10)
        tk.Button(self, text="Box Plot",
                  command=lambda: self.boxPlot(controller.get_choice())
                  ).pack(pady=10)
        tk.Button(self, text="Scatter",
                  command=lambda: self.scatter(controller.get_choice())
                  ).pack(pady=10)
        tk.Button(self, text="OHLC",
                  command=lambda: self.Ohlc(controller.get_choice())
                  ).pack(pady=10)
        
    def column_choice(self):
        tk.Label(self, text="Choose one to be the data represented in the graph", font=("Arial", 18)).pack(pady=20)
        column = None
        tk.Button(self, text="Open", column = "Open").pack(pady=10)
        tk.Button(self, text="High", column = "High").pack(pady=10)
        tk.Button(self, text="Low", column = "Low").pack(pady=10)
        tk.Button(self, text="Close", column = "Close").pack(pady=10)
        tk.Button(self, text="Adjusted Close", column = "Adjusted Close").pack(pady=10)
        tk.Button(self, text="Dividend Amount", column = "Dividend Amount").pack(pady=10)
        return column

    def display(self, df):
        tk.Label(self, text="The Display", font=("Arial", 18)).pack(pady=20)
        tk.Label(self, text=df, font=("Arial", 18)).pack(pady=20,fill="both", expand=True)

    def describeData(self, df):
        tk.Label(self, text="The Describe", font=("Arial", 18)).pack(pady=20)
        tk.Label(self, text=df.describe(), font=("Arial", 18)).pack(fill="both", padx=10, pady=5)

    def dataInfo(self, df):
        tk.Label(self, text="The Info", font=("Arial", 18)).pack(pady=20)
        tk.Label(self, text=df.info, font=("Arial", 18)).pack(fill="both", padx=10, pady=5)

    def lineChart(self, df):
        column = self.column_choice()
        fig, ax = plt.subplots(figsize=(10,5))
        df[column].plot(title=column, ax=ax)

        canvas = FigureCanvasTkAgg(fig, master=self) 
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def barChart(self, df):
        column = self.column_choice()
        fig, ax = plt.subplots(figsize=(12,4))
        df[column].plot(kind="bar", title="Monthly column", ax=ax)

        plt.tight_layout()   

        canvas = FigureCanvasTkAgg(fig, master=self)  
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def histogram(self, df):
        column = self.column_choice()
        fig, ax = plt.subplots()
        (df[column].pct_change() * 100).hist(bins=50, ax=ax)
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def boxPlot(self, df):
        column = self.column_choice()
        plot = df[column].pct_change() * 100
        fig, ax = plt.subplots()
        ax.boxplot(plot.dropna())
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def scatter(self, df):
        fig, ax = plt.subplots()
        ax.scatter(df["Volume"], df["Close"], alpha=0.5)
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def Ohlc(self, df):
        fig = go.Figure(data=go.Ohlc(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        ))
        html = fig.to_html(include_plotlyjs="cdn")
        HTMLLabel(self, html=html).pack(fill="both", expand=True)

if __name__ == "__main__":
    app = StockGUI()
    app.mainloop()
