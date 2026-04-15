from typing import Any

import numpy as np
from arch import arch_model
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns

class GARCHEngine:
    def __init__(
            self,
            log_returns : pd.DataFrame,
            display = True
    ) -> None:
        self.log_returns_data = log_returns
        self.garch_results = None
        self.std_residual = pd.DataFrame(index=self.log_returns_data.index)
        self.conditional_volatilises = pd.DataFrame(index=self.log_returns_data.index)
        self.VaR = None
        self.display = display

    def garch_engine(
            self
    ) -> None:
        model = arch_model(
            self.log_returns_data,
            mean="zero",
            vol="GARCH",
            p=1,
            q=1,
            dist="skewt"
        )

        residuals = model.fit(disp="off")
        self.garch_results = residuals
        self.conditional_volatilises = residuals.conditional_volatility
        self.std_residual = residuals.resid / residuals.conditional_volatility

        if(self.display):
            plt.figure(figsize=(12, 6))

            plt.plot(
                self.std_residual.index,
                self.std_residual,
                label='Standardized Residuals',
                alpha=0.7)

            plt.axhline(0, color='gray', linestyle='--', linewidth=1)
            plt.title('Standardized Residuals Over Time')
            plt.xlabel('Date')
            plt.ylabel('Standardized Residuals')
            plt.grid(True)
            plt.legend()

    def compute_VaR(
            self,
            alpha = 0.05
    ) -> pd.DataFrame:
        q_alpha = np.quantile(self.std_residual, alpha)
        self.VaR = -q_alpha * self.conditional_volatilises

        return self.VaR

    def vitals(
            self
    ) -> Any:
       return self.garch_results.summary()

    def visuals_extentsion(
            self
    ) -> None:
        axes = plt.subplot()

        sns.histplot(self.std_residual, kde=True, ax=axes, stat='density', color='lightcoral')
        axes.set_title('Distribution of Standardized Residuals')
        axes.set_xlabel('Standardized Residuals')
        axes.set_ylabel('Density')

        plt.tight_layout()
        plt.show()      

