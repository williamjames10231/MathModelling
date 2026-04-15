from typing import Any

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(
            self
    )-> None:
        self.tickers : list[str] = [
            "PSEI.PS",
        ]
        self.start_date : str = "2010-01-01"
        self.end_date : str = "2020-12-31"
        self.data = None                    #raw data
        self.data_treated = None            #log returns
        self.data_treated_scaled = None     #log returns scaled


    def load_data(
            self
    ) -> pd.DataFrame:
        self.data = yf.download(
            tickers=self.tickers,
            start=self.start_date,
            end=self.end_date,
            interval="1d",
        )["Close"]
        self.data = self.data.dropna()
        return self.data

    def compute_returns(
            self
    ) -> tuple[Any, int | Any]:
        self.data_treated = np.log(self.data / self.data.shift(1)).dropna()
        self.data_treated_scaled = self.data_treated * 100
        return self.data_treated, self.data_treated_scaled

    def visualize(
            self
    ) -> None:
        self.data.plot()
        plt.title("raw closing prices over time")
        plt.xlabel("date")
        plt.ylabel("closing prices (PHP)")
        plt.savefig("Data/data_loader/raw_closing.png")

        self.data_treated.plot()
        plt.title("log returns over time")
        plt.xlabel("date")
        plt.ylabel("log returns")
        plt.savefig("Data/data_loader/log_returns_closing.png")

        self.data_treated_scaled.plot()
        plt.title("log returns scaled over time")
        plt.xlabel("date")
        plt.ylabel("log returns")
        plt.savefig("Data/data_loader/log_returns_scaled_closing.png")

    def vitals(
            self
    )-> None:
        vitals = self.data_treated_scaled.describe()
        vitals.to_csv("Data/data_loader/log_returns_scaled_descriptive_statistics.csv")