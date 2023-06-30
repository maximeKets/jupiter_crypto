import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import warnings


class LongOnlyBacktester:
    def __init__(self, filepath, symbol, start, end, tc):
        self.filepath = filepath
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.data = None
        self.results = None
        self.return_thresh = None
        self.volume_thresh = None
        self.results_overview = None
        self._get_data()

    def _get_data(self):
        self.data = pd.read_csv(self.filepath, parse_dates=["Date"], index_col="Date")
        self.data = self.data.loc[self.start:self.end].copy()
        self.data["returns"] = np.log(self.data.Close / self.data.Close.shift(1))

    def test_strategy(self, percentiles=(90, 5, 20)):
        data = self.data.copy()
        return_thresh = np.percentile(data.returns.dropna(), percentiles[0])
        vol_ch = np.log(data.Volume.div(data.Volume.shift(1)))
        vol_ch.loc[(vol_ch > 3) | (vol_ch < -3)] = np.nan
        volume_thresh = np.percentile(vol_ch.dropna(), [percentiles[1], percentiles[2]])
        cond1 = data.returns >= return_thresh
        cond2 = vol_ch.between(volume_thresh[0], volume_thresh[1])

        data["position"] = 1
        data.loc[cond1 & cond2, "position"] = 0
        data["strategy"] = data.position.shift(1) * data["returns"]
        data["trades"] = data.position.diff().fillna(0).abs()
        data.strategy = data.strategy + data.trades * self.tc
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)

        self.results = data
        self.return_thresh = return_thresh
        self.volume_thresh = volume_thresh

    def plot_results(self):
        if self.results is None:
            print("Aucun résultat à afficher. Exécutez d'abord la méthode .test_strategy()")
        else:
            title = f"{self.symbol} | Retour seuil : {self.return_thresh:.2f} | Seuils de volume : {self.volume_thresh[0]:.2f}, {self.volume_thresh[1]:.2f}"
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
            plt.show()

    def optimize_strategy(self, return_range, vol_low_range, vol_high_range):
        combinations = list(product(return_range, vol_low_range, vol_high_range))
        results = []
        for comb in combinations:
            self.test_strategy(percentiles=comb)
            performance = self.results["cstrategy"][-1]
            results.append(performance)

        results_overview = pd.DataFrame(data=combinations, columns=["returns", "vol_low", "vol_high"])
        results_overview["performance"] = results
        self.results_overview = results_overview
