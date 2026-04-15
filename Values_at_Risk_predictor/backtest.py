import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt


class Backtest:
    def __init__(self,returns, VaR, alpha):
        self.returns = returns
        self.VaR = VaR
        self.alpha = alpha

        self.aligned_returns = self.returns.iloc[1:]
        self.aligned_VaR = self.VaR.iloc[1:]

        self.violations = None
        self.actual_loss = None

    
    def backtest(self):
        self.actual_loss = -self.aligned_returns
        self.violations = (self.actual_loss["PSEI.PS"] > self.aligned_VaR).astype(int)
        violations_rate = self.violations.mean()
        print(f"Violation rate: {violations_rate:.4f} (expected {self.alpha})")

    def kupiec_test(self):
        n_viol = self.violations.sum()
        T = len(self.violations)
        p_hat = n_viol / T
        log_likelihood_p_hat = n_viol * np.log(p_hat) + (T - n_viol) * np.log(1 - p_hat)
        log_likelihood_alpha = n_viol * np.log(self.alpha) + (T - n_viol) * np.log(1 - self.alpha)
        LR = 2 * (log_likelihood_p_hat - log_likelihood_alpha)
        p_value = 1 - chi2.cdf(LR, df=1)

        return log_likelihood_p_hat, log_likelihood_alpha,LR,p_value
    
    def visualize_VaR(self):
        plt.figure(figsize=(12,6))
        plt.plot(self.aligned_returns.index, self.actual_loss, label='Actual loss', alpha=0.5)
        plt.plot(self.aligned_returns.index, self.aligned_VaR, label='5% VaR', color='red')
        plt.fill_between(self.aligned_returns.index, 0,  self.aligned_VaR, alpha=0.2, color='red')
        plt.legend()
        plt.title('Philippine Index: Daily 5% VaR (Univariate GARCH with skewed‑t)')
        plt.ylabel('Loss / VaR')
        plt.grid(True)
        plt.show()