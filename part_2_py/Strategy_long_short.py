import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.style.use("seaborn")


class Long_Short_Backtester():
    ''' Classe pour le backtesting vectorisé de stratégies simples de trading long-courtes.

    Attributs
    ============
    filepath: str
        chemin local du fichier de données (csv)
    symbol: str
        symbole du ticker (instrument) à tester
    start: str
        date de début pour l'importation des données
    end: str
        date de fin pour l'importation des données
    tc: float
        frais de trading proportionnels par transaction

    Méthodes
    =======
    get_data:
        importe les données.

    test_strategy:
        prépare les données et teste la stratégie de trading, y compris les rapports (wrapper).

    prepare_data:
        prépare les données pour le backtesting.

    run_backtest:
        exécute le backtest de la stratégie.

    plot_results:
        trace la performance cumulée de la stratégie de trading par rapport à l'achat et la détention.

    optimize_strategy:
        teste la stratégie pour différentes valeurs de paramètres, y compris l'optimisation et les rapports (wrapper).

    find_best_strategy:
        trouve la stratégie optimale (maximum global).


    print_performance:
        calcule et imprime diverses mesures de performance.

    '''

    def __init__(self, filepath, symbol, start, end, tc):
        self.filepath = filepath
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None
        self.get_data()
        self.tp_year = (self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25))

    def __repr__(self):
        return "Long_Short_Backtester(symbol = {}, start = {}, end = {})".format(self.symbol, self.start, self.end)

    def get_data(self):
        ''' Importe les données.
        '''
        raw = pd.read_csv(self.filepath, parse_dates=["Date"], index_col="Date")
        raw = raw.loc[self.start:self.end].copy()
        raw["returns"] = np.log(raw.Close / raw.Close.shift(1))
        self.data = raw

    def test_strategy(self, percentiles=None, thresh=None):
        '''
        Prépare les données et teste la stratégie de trading, y compris les rapports (Wrapper).

        Paramètres
        ============
        percentiles: tuple (return_low_perc, return_high_perc, vol_low_perc, vol_high_perc)
            percentiles de rendement et de volume à considérer pour la stratégie.

        thresh: tuple (return_low_thresh, return_high_thresh, vol_low_thresh, vol_high_thesh)
            seuils de rendement et de volume à considérer pour la stratégie.
        '''

        self.prepare_data(percentiles=percentiles, thresh=thresh)
        self.run_backtest()

        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data

        self.print_performance()

    def prepare_data(self, percentiles, thresh):
        ''' Prépare les données pour le backtesting.
        '''
        ########################## Stratégie spécifique #############################

        data = self.data[["Close", "Volume", "returns"]].copy()
        data["vol_ch"] = np.log(data.Volume.div(data.Volume.shift(1)))
        data.loc[data.vol_ch > 3, "vol_ch"] = np.nan
        data.loc[data.vol_ch < -3, "vol_ch"] = np.nan

        if percentiles:
            self.return_thresh = np.percentile(data.returns.dropna(), [percentiles[0], percentiles[1]])
            self.volume_thresh = np.percentile(data.vol_ch.dropna(), [percentiles[2], percentiles[3]])
        elif thresh:
            self.return_thresh = [thresh[0], thresh[1]]
            self.volume_thresh = [thresh[2], thresh[3]]

        cond1 = data.returns <= self.return_thresh[0]
        cond2 = data.vol_ch.between(self.volume_thresh[0], self.volume_thresh[1])
        cond3 = data.returns >= self.return_thresh[1]

        data["position"] = 0
        data.loc[cond1 & cond2, "position"] = 1
        data.loc[cond3 & cond2, "position"] = -1

        ##########################################################################

        self.results = data

    def run_backtest(self):
        ''' Exécute le backtest de la stratégie.
        '''

        data = self.results.copy()
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data["trades"] = data.position.diff().fillna(0).abs()
        data.strategy = data.strategy + data.trades * self.tc

        self.results = data

    def plot_results(self):
        ''' Trace la performance cumulée de la stratégie de trading par rapport à l'achat et la détention.
        '''
        if self.results is None:
            print("Exécutez test_strategy() d'abord.")
        else:
            title = "{} | TC = {}".format(self.symbol, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))

    def optimize_strategy(self, return_low_range, return_high_range, vol_low_range, vol_high_range, metric="Multiple"):
        '''
        Teste la stratégie pour différentes valeurs de paramètres, y compris l'optimisation et les rapports (Wrapper).

        Paramètres
        ============
        return_low_range: tuple
            tuples de la forme (début, fin, taille de pas).

        return_high_range: tuple
            tuples de la forme (début, fin, taille de pas).

        vol_low_range: tuple
            tuples de la forme (début, fin, taille de pas).

        vol_high_range: tuple
            tuples de la forme (début, fin, taille de pas).

        metric: str
            mesure de performance à optimiser (peut être "Multiple" ou "Sharpe")
        '''

        self.metric = metric

        if metric == "Multiple":
            performance_function = self.calculate_multiple
        elif metric == "Sharpe":
            performance_function = self.calculate_sharpe

        return_low_range = range(*return_low_range)
        return_high_range = range(*return_high_range)
        vol_low_range = range(*vol_low_range)
        vol_high_range = range(*vol_high_range)

        combinations = list(product(return_low_range, return_high_range, vol_low_range, vol_high_range))
        self.optimization = pd.DataFrame(columns=["return_low", "return_high", "vol_low", "vol_high", "performance"])

        for comb in combinations:
            return_thresh = [comb[0] / 100, comb[1] / 100]
            volume_thresh = [comb[2] / 100, comb[3] / 100]
            self.prepare_data(thresh=return_thresh + volume_thresh)
            self.run_backtest()
            performance = performance_function()
            self.optimization = self.optimization.append({"return_low": return_thresh[0],
                                                          "return_high": return_thresh[1],
                                                          "vol_low": volume_thresh[0],
                                                          "vol_high": volume_thresh[1],
                                                          "performance": performance},
                                                         ignore_index=True)

        self.optimization = self.optimization.sort_values(by="performance", ascending=False).reset_index(drop=True)

    def calculate_multiple(self):
        '''Calcule le multiple de la stratégie de trading.
        '''
        return np.exp(self.results.cstrategy.iloc[-1]) / np.exp(self.results.creturns.iloc[-1])

    def calculate_sharpe(self):
        '''Calcule le ratio de Sharpe de la stratégie de trading.
        '''
        return np.mean(self.results.strategy) / np.std(self.results.strategy)

    def set_params(self, thresh=None, percentiles=None):
        '''Permet de définir les paramètres pour les seuils de rendement et de volume.
        '''
        if thresh:
            self.return_thresh = [thresh[0], thresh[1]]
            self.volume_thresh = [thresh[2], thresh[3]]
        elif percentiles:
            self.prepare_data(percentiles=percentiles)

    def get_best_params(self, n=1):
        '''Retourne les n meilleurs ensembles de paramètres.
        '''
        if self.optimization is None:
            print("Exécutez optimize_strategy() d'abord.")
        else:
            return self.optimization.head(n)
