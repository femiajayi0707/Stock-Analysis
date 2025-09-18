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

#Defining the Root Window and Main Applications
class StockGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stock Analysis App")
        self.geometry("600x600")
        self.resizable(False,True)

        self.choice = None #stores which data user selected

        #container which holds the pages stacked on each other
        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        # Allow pages to expand and fill space
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Dictionary to store all pages
        self.pages = {}
        for Page in (FirstPage, SecondPage, ThirdPage, ModelPage, VisPage):
            page = Page(parent=container, controller=self)
            self.pages[Page.__name__] = page
        # Dictionary to store all pages
            page.grid(row=0, column=0, sticky="nsew")  # stacked
        
        # Show the first page initially
        self.show_page("FirstPage")

    #Setter and Getter for choice   
    def set_choice(self, text):
        self.choice = text
    def get_choice(self):
        return self.choice

    # Method to bring a given page to the front
    def show_page(self, name: str):
        self.pages[name].tkraise()

class FirstPage(tk.Frame):
    def __init__(self, parent, controller: StockGUI):
        super().__init__(parent)
        #Welcome Text
        tk.Label(self, text="Welcome to the Stock Analysis and Prediction Tool", font=("Arial", 18)).pack(pady=20)
        
        #Frame for the buttons
        change = tk.Frame(self)
        change.pack(padx=6)
        tk.Button(change, text="Next",
                  command=lambda: controller.show_page("SecondPage")
                  ).pack(side='left', padx = 20, pady=15)

class SecondPage(tk.Frame):
    def __init__(self, parent, controller: StockGUI):
        super().__init__(parent)
        self.controller = controller
        tk.Label(self, text="Please input your data and make a selection", font=("Arial", 18)).pack(pady=20)
        
        # Frame for input questions
        questions = tk.Frame(self)
        questions.pack(pady=5)

        #Stock name input
        tk.Label(questions, text="What is the name of the companies stock ", font=("Arial", 18)).pack(pady=20)
        self.name = tk.Entry(questions)
        self.name.pack()

        #Api key input
        tk.Label(questions, text="What is your api key for alphavantage ", font=("Arial", 18)).pack(pady=20)
        self.api = tk.Entry(questions)
        self.api.pack()

        #Frame for daily, weekly, monthly buttons
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
        
        #Status label 
        self.status_label = tk.Label(self, text="", font=("Arial", 14), fg="blue")
        self.status_label.pack(pady=20)

        #Navigation buttons
        change = tk.Frame(self)
        change.pack(padx=6)
        tk.Button(change, text="Back",
                  command=lambda: controller.show_page("FirstPage")
                  ).pack(side='left', padx = 20, pady=15)
        tk.Button(change, text="Next",
                  command=lambda: controller.show_page("ThirdPage")
                  ).pack(side='right', padx = 20, pady=15)

    #Data fetching methods
    def getWeeklyData(self):
        stockName = self.name.get()
        key = self.api.get()
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol='  + stockName + '&apikey=' + key
        r = requests.get(url)
        data = r.json()
        data = data["Weekly Adjusted Time Series"]
        self.status_label.config(text="Weekly Data has been selected")
        self.controller.set_choice(self.formatData(data))

    def getDailyData(self):
        stockName = self.name.get()
        key = self.api.get()
        url = url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol='  + stockName + '&apikey=' + key
        r = requests.get(url)
        data = r.json()
        data = data["Daily Adjusted Time Series"]
        self.status_label.config(text="Daily Data has been selected")
        self.controller.set_choice(self.formatData(data))

    def getMonthlyData(self):
        stockName = self.name.get()
        key = self.api.get()
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol=' + stockName + '&apikey=' + key
        r = requests.get(url)
        data = r.json()
        data = data["Monthly Adjusted Time Series"]
        self.status_label.config(text="Monthly Data has been selected")
        self.controller.set_choice(self.formatData(data))

    #Format data into a data frame
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
        #convert index to datetime
        df = df.rename(columns=column_rename)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        #conver columns to be numeric
        for col in df.columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
        return df

class ThirdPage(tk.Frame):
    def __init__(self, parent, controller: StockGUI):
        super().__init__(parent)
        tk.Label(self, text="""  Choose One: 
         Do you wish to Create a Model or 
        do you seek visualisation of the data""", font=("Arial", 18)).pack(pady=20)
        
        #Navigation buttons
        change = tk.Frame(self)
        change.pack(padx=6)
        tk.Button(change, text="Models",
                  command=lambda: controller.show_page("ModelPage")
                  ).pack(side="left",padx=20)
        tk.Button(change, text="Visualistaion",
                  command=lambda: controller.show_page("VisPage")
                  ).pack(side="left",padx=6)
        tk.Button(change, text="Back",
                  command=lambda: controller.show_page("SecondPage")
                  ).pack(side="left",padx=6)

class ModelPage(tk.Frame):
    def __init__(self, parent, controller: StockGUI):
        super().__init__(parent)

        #Title row
        tk.Label(self, text="Model Screen", font=("Arial", 18)).grid(row=0, column=0, columnspan=2, pady=10)

        #Navigation button
        tk.Button(self, text="Back",
                  command=lambda: controller.show_page("ThirdPage")
                  ).grid(row=1, column=0, sticky="w", padx=10, pady=10)

        #Classification section 
        tk.Label(self, text="Classification Models", font=("Arial", 16)).grid(row=2, column=0, pady=10, sticky="w")

        tk.Button(self, text="Logistic Regression Model",
                  command=lambda: self.logisticReg(controller.get_choice())
                  ).grid(row=3, column=0, sticky="w", padx=20, pady=5)

        tk.Button(self, text="Decision Tree Classifier Model",
                  command=lambda: self.decisionTreeClass(controller.get_choice())
                  ).grid(row=4, column=0, sticky="w", padx=20, pady=5)

        #Regression section
        tk.Label(self, text="Regression Models", font=("Arial", 16)).grid(row=2, column=1, pady=10, sticky="w")

        tk.Button(self, text="Decision Tree Regression Model",
                  command=lambda: self.decisionTreeReg(controller.get_choice())
                  ).grid(row=3, column=1, sticky="w", padx=20, pady=5)

        tk.Button(self, text="Linear Regression Model",
                  command=lambda: self.linearReg(controller.get_choice())
                  ).grid(row=4, column=1, sticky="w", padx=20, pady=5)

        #Results area
        self.results = tk.Text(self, wrap="word", width=62, height=16, font=("Consolas", 12))
        self.results.grid(row=5, column=0, columnspan=2, pady=20, padx=20)
        self.results.insert("end", "Model results will appear here...\n")
    
    #Classification Models code
    def logisticReg(self, df):
        X = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        y = (df["Close"].shift(-1) > df["Close"]).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.80, test_size=0.2, shuffle=False
        )
        model = lm.LogisticRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        self.eval_for_class_models(pred, y_test, "Logistic Regression")

    def decisionTreeClass(self, df):
        X = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        y = (df["Close"].shift(-1) > df["Close"]).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.80, test_size=0.2, shuffle=False
        )
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        self.eval_for_class_models(pred, y_test, "Decision Tree Classifier")

    #Regression Models Code
    def decisionTreeReg(self, df):
        X = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        y = df["Close"].shift(-1).dropna()
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx].values
        y = y.loc[common_idx].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.80, test_size=0.2, shuffle=False
        )
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        self.eval_for_reg_models(pred, y_test, "Decision Tree Regressor")

    def linearReg(self, df):
        X = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        y = df["Close"].shift(-1).dropna()
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx].values
        y = y.loc[common_idx].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.80, test_size=0.2, shuffle=False
        )
        model = lm.LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        self.eval_for_reg_models(pred, y_test, "Linear Regression")

    #Evaluation Functions Code
    def eval_for_reg_models(self, pred, test, model_name):
        self.results.insert("end", f"Results for {model_name}\n\n")
        self.results.insert("end", f"MSE: {mean_squared_error(test, pred):.4f}\n")
        self.results.insert("end", f"MAE: {mean_absolute_error(test, pred):.4f}\n")
        self.results.insert("end", f"R²: {r2_score(test, pred):.4f}\n")

    def eval_for_class_models(self, pred, test, model_name):
        self.results.insert("end", f"Results for {model_name}\n\n")
        self.results.insert("end", f"Accuracy: {accuracy_score(test, pred):.4f}\n\n")
        self.results.insert("end", "Classification Report:\n")
        self.results.insert("end", classification_report(test, pred, zero_division=0))
        self.results.insert("end", "\n\nConfusion Matrix:\n")
        self.results.insert("end", str(confusion_matrix(test, pred)))

class VisPage(tk.Frame):
    def __init__(self, parent, controller: StockGUI):
        super().__init__(parent)
        tk.Label(self, text="Visualisation Screen", font=("Arial", 18)).grid(row=0, column=0, columnspan=2, pady=20)
        tk.Button(self, text="Back",
                  command=lambda: controller.show_page("ThirdPage")
                  ).grid(row=0, column=1, pady=10)

        tk.Button(self, text="Display",
                  command=lambda: self.display(controller.get_choice())
                  ).grid(row=1, column=1, pady=10)
        tk.Button(self, text="Describe",
                  command=lambda: self.describeData(controller.get_choice())
                  ).grid(row=1, column=0, pady=10)
        tk.Button(self, text="OHLC",
                  command=lambda: self.Ohlc(controller.get_choice())
                  ).grid(row=2, column=1, pady=10)
        tk.Button(self, text="Line Chart",
                  command=lambda: self.lineChart(controller.get_choice())
                  ).grid(row=3, column=0, pady=10)
        tk.Button(self, text="Bar Chart",
                  command=lambda: self.barChart(controller.get_choice())
                  ).grid(row=3, column=1, pady=10)
        tk.Button(self, text="Histogram",
                  command=lambda: self.histogram(controller.get_choice())
                  ).grid(row=4, column=0, pady=10)
        tk.Button(self, text="Box Plot",
                  command=lambda: self.boxPlot(controller.get_choice())
                  ).grid(row=4, column=1, pady=10)
        tk.Button(self, text="Scatter",
                  command=lambda: self.scatter(controller.get_choice())
                  ).grid(row=2, column=0, pady=10)

        # Results area
        self.results = tk.Text(self, wrap="word", width=62, height=16)
        self.results.grid(row=6, column=0, columnspan=2, pady=20, padx=20)
        self.results.insert("end", "Model results will appear here...\n")

    def column_choice(self):
        win = tk.Toplevel(self)
        win.title("Pick column")
        win.resizable(False, False)
        win.transient(self)      # stay on top of the parent
        win.grab_set()           # make it modal

        tk.Label(win, text="Choose a column").grid(row=0, column=0, columnspan=2, padx=12, pady=8)

        win.choice = None

        def pick(c):
            win.choice = c
            win.destroy()

        tk.Button(win, text="Open", command=lambda: pick("Open")).grid(row=1, column=0, padx=6, pady=6)
        tk.Button(win, text="High", command=lambda: pick("High")).grid(row=1, column=1, padx=6, pady=6)
        tk.Button(win, text="Low", command=lambda: pick("Low")).grid(row=2, column=0, padx=6, pady=6)
        tk.Button(win, text="Close", command=lambda: pick("Close")).grid(row=2, column=1, padx=6, pady=6)
        tk.Button(win, text="Adjusted Close", command=lambda: pick("Adjusted Close")).grid(row=3, column=0, padx=6, pady=6)
        tk.Button(win, text="Dividend Amount", command=lambda: pick("Dividend Amount")).grid(row=3, column=1, padx=6, pady=6)

        self.wait_window(win)    
        return win.choice     

    def display(self, df):
        self.results.insert("end","\n The Display\n")
        self.results.insert("end", df.head(5).to_string() + "\n")

    def describeData(self, df):
        self.results.insert("end","\n The Description\n")
        self.results.insert("end", df.describe().to_string() + "\n")

    def lineChart(self, df):
        column = self.column_choice()
        fig, ax = plt.subplots(figsize=(10,5))
        df[column].plot(title=column, ax=ax)

        win = tk.Toplevel(self)
        win.title("Line Chart")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def barChart(self, df):
        column = self.column_choice()
        fig, ax = plt.subplots(figsize=(12,4))
        df[column].plot(kind="bar", title="Monthly column", ax=ax)
        plt.tight_layout()   

        win = tk.Toplevel(self)
        win.title("Bar Chart")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def histogram(self, df):
        column = self.column_choice()
        fig, ax = plt.subplots()
        (df[column].pct_change() * 100).hist(bins=50, ax=ax)
        plt.tight_layout()

        win = tk.Toplevel(self)
        win.title("Histogram")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def boxPlot(self, df):
        column = self.column_choice()
        plot = df[column].pct_change() * 100
        fig, ax = plt.subplots()
        ax.boxplot(plot.dropna())
        plt.tight_layout()

        win = tk.Toplevel(self)
        win.title("Box plot")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def scatter(self, df):
        fig, ax = plt.subplots()
        ax.scatter(df["Volume"], df["Close"], alpha=0.5)
        plt.tight_layout()

        win = tk.Toplevel(self)
        win.title("Scatter")
        canvas = FigureCanvasTkAgg(fig, master=win)
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
        fig.show()
if __name__ == "__main__":
    app = StockGUI()
    app.mainloop()