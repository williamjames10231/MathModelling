from typing import Any

import numpy as np
from arch import arch_model
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

class GARCHEngine:
    def __init__(
            self,
            log_returns : pd.DataFrame
    ) -> None:
        self.log_returns_data = log_returns
        self.garch_results = None
        self.std_residual = pd.DataFrame(index=self.log_returns_data.index)
        self.conditional_volatilises = pd.DataFrame(index=self.log_returns_data.index)
        self.VaR = None

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

    def compute_VaR(
            self,
            alpha = 0.05
    ) -> pd.DataFrame:
        q_alpha = np.quantile(self.std_residual, alpha)
        self.VaR = -q_alpha * self.conditional_volatilises

        return self.VaR

    def visualize(
            self
    ) -> None:
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
        plt.savefig("Data/garch_engine/log_returns_closing.png")

    def vitals(
            self
    ) -> None:
        Path(
            "Data/garch_engine/garch_summary.txt"
        ).write_text(
            self.garch_results.summary().as_text()
        )

