import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from Values_at_Risk_predictor.data_loader import DataLoader
from Values_at_Risk_predictor.GARCH_Engine import GARCHEngine

if __name__ == "__main__":
    predictor = DataLoader()

    datas = predictor.load_data()
    log_returns, log_returns_scaled = predictor.compute_returns()
    predictor.visualize()
    predictor.vitals()

    garch = GARCHEngine(log_returns_scaled)
    garch.garch_engine()
    VaR = garch.compute_VaR()
    garch.visualize()
    garch.vitals()
    pass