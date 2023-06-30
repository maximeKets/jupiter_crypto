#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn")


class LongOnlyStrategy:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file, parse_dates=["Date"], index_col="Date")
        self.data = self.data[["Close", "Volume"]].copy()
        self.data["returns"] = np.log(self.data.Close.div(self.data.Close.shift(1)))
        self.data["vol_ch"] = np.log(self.data.Volume.div(self.data.Volume.shift(1)))
        self.data["position"] = 1
        self.data["strategy"] = 0
        self.data["cstrategy"] = 0
        self.data["strategy_net"] = 0
        self.data["cstrategy_net"] = 0

    def analyze_returns(self):
        return self.data.returns.describe()

    def plot_close(self):
        self.data.Close.plot(figsize=(12, 8), title="BTC/USDT", fontsize=12)
        plt.show()

    def plot_volume(self):
        self.data.Volume.plot(figsize=(12, 8), title="BTC/USDT", fontsize=12)
        plt.show()

    def prepare_data(self):
        pass  # Ajoutez la logique de préparation des données ici

    def calculate_positions(self):
        pass  # Ajoutez la logique pour calculer les positions ici

    def calculate_strategy(self):
        pass  # Ajoutez la logique pour calculer la stratégie ici

    def calculate_net_strategy(self):
        pass  # Ajoutez la logique pour calculer la stratégie nette ici

    def plot_strategy(self):
        self.data[["creturns", "cstrategy", "cstrategy_net"]].plot(figsize=(12, 8))
        plt.show()


if __name__ == "__main__":
    strategy = LongOnlyStrategy("bitcoin.csv")
    print(strategy.analyze_returns())
    strategy.plot_close()
    strategy.plot_volume()
    strategy.prepare_data()
    strategy.calculate_positions()
    strategy.calculate_strategy()
    strategy.calculate_net_strategy()
    strategy.plot_strategy()
