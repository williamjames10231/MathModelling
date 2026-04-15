from typing import Any

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class DataLoader:
    def __init__(
            self,
            display = True
    )-> None:
        self.ticker : str = "PSEI.PS"
        self.start_date : str = "2010-01-01"
        self.end_date : str = "2020-12-31"
        self.data = None                    #raw data
        self.data_treated = None            #log returns
        self.data_treated_scaled = None     #log returns scaled
        self.isDisplay = display
        self.dir = Path(__file__).resolve().parent


    def load_data(
            self,
    ) -> pd.DataFrame:
        self.data = yf.download(
            tickers=self.ticker,
            start=self.start_date,
            end=self.end_date,
            interval="1d",
        )["Close"]
        self.data = self.data.dropna()

        if(self.isDisplay):
            self.data.plot()
            plt.title("raw closing prices over time")
            plt.xlabel("date")
            plt.ylabel("closing prices (PHP)")
            #plt.savefig(self.dir / "Data/data_loader/raw_closing.png")
            plt.legend()
            plt.show()

        return self.data

    def compute_returns(
            self
    ) -> tuple[Any, int | Any]:
        self.data_treated = np.log(self.data / self.data.shift(1)).dropna()
        self.data_treated_scaled = self.data_treated * 100

        if(self.isDisplay):
            fig, axes = plt.subplots(1,2, figsize=(15,6))

            self.data_treated.plot(
                ax=axes[0],
                title="log returns over time",
                xlabel="date",
                ylabel="log returns"
            )

            self.data_treated_scaled.plot(
                ax=axes[1],
                title="log returns scaled over time",
                xlabel="date",
                ylabel="scaled log returns"
            )
            
            plt.legend()
            plt.tight_layout()
            plt.show()

        return self.data_treated, self.data_treated_scaled


    def vitals(
            self
    )-> Any:
        vitals = (
            self.data_treated_scaled.mean(),
            self.data_treated_scaled.std(), 
            self.data_treated_scaled.skew(), 
            self.data_treated_scaled.kurtosis()
        )

        return vitals