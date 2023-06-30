#!/usr/bin/env python
# coding: utf-8 # # Strategy Optimization &  Forward Testing # ## Getting started # In[ ]: 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Strategy_optimisation_class_p1 import Long_Only_Backtester

plt.style.use("seaborn")
# In[ ]: 
data = pd.read_csv("bitcoin.csv", parse_dates=["Date"], index_col="Date")
data
# In[ ]: 
data.info()
# In[ ]: 
data["returns"] = np.log(data.Close / data.Close.shift(1))
data
# In[ ]:   # ## Strategy Optimization (Part 1) # __Three Strategy Parameters:__ # - Return Threshold: All Returns >= __90th__ Percentile labeled "Very High Return"
# - Low and High Volume Change Threshold: All Volume Changes between __5th__ and __20th__ Percentile labeled "Moderate to High Decrease in Volume"  # __-> Strategy Parameters = (90, 5, 20)__ # In[ ]: 
data


# In[ ]: 
def backtest(data, parameters, tc):
    # prepare features
    data = data[["Close", "Volume", "returns"]].copy()
    data["vol_ch"] = np.log(data.Volume.div(data.Volume.shift(1)))
    data.loc[data.vol_ch > 3, "vol_ch"] = np.nan
    data.loc[data.vol_ch < -3, "vol_ch"] = np.nan

    # define trading positions
    return_thresh = np.percentile(data.returns.dropna(), parameters[0])
    cond1 = data.returns >= return_thresh
    volume_thresh = np.percentile(data.vol_ch.dropna(), [parameters[1], parameters[2]])
    cond2 = data.vol_ch.between(volume_thresh[0], volume_thresh[1])

    data["position"] = 1
    data.loc[cond1 & cond2, "position"] = 0

    # backtest
    data["strategy"] = data.position.shift(1) * data["returns"]
    data["trades"] = data.position.diff().fillna(0).abs()
    data.strategy = data.strategy + data.trades * tc
    data["creturns"] = data["returns"].cumsum().apply(np.exp)
    data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)

    # return strategy multiple
    return data.cstrategy[-1]


# In[ ]: 
import warnings

warnings.filterwarnings('ignore')
# In[ ]: 
backtest(data=data, parameters=(90, 5, 20), tc=-0.00085)
# In[ ]: 
return_range = range(85, 98, 1)  # potential values for return_thresh
vol_low_range = range(2, 16, 1)  # potential values for vol_low
vol_high_range = range(16, 35, 1)  # potential values for vol_high
# In[ ]: 
list(return_range)
# __Plan: Run Backtest for all combinations and find the best combination(s)__ # In[ ]: 
from itertools import product

# In[ ]: 
combinations = list(product(return_range, vol_low_range, vol_high_range))
combinations
# In[ ]: 
len(combinations)
# In[ ]: 
13 * 14 * 19
# In[ ]: 
results = []
for comb in combinations:
    results.append(backtest(data=data, parameters=comb, tc=-0.00085))
# In[ ]: 
many_results = pd.DataFrame(data=combinations, columns=["returns", "vol_low", "vol_high"])
many_results["performance"] = results
# In[ ]: 
many_results
# In[ ]:   # ## Strategy Optimization (Part 2) # In[ ]: 
many_results
# In[ ]: 
many_results.nlargest(20, "performance")
# In[ ]: 
many_results.nsmallest(20, "performance")
# In[ ]: 
many_results.groupby("returns").performance.mean().plot()
plt.show()
# In[ ]: 
many_results.groupby("vol_low").performance.mean().plot()
plt.show()
# In[ ]: 
many_results.groupby("vol_high").performance.mean().plot()
plt.show()
# In[ ]: 
backtest(data=data, parameters=(94, 11, 27), tc=-0.00085)
# In[ ]: 
backtest(data=data, parameters=(90, 5, 20), tc=-0.00085)
# In[ ]:   # ## Putting everything together: a Backtester Class # In[ ]: 


warnings.filterwarnings("ignore")
plt.style.use("seaborn")


# __Why using OOP and creating a class?__ # - Organizing/Storing/Linking all Functionalities and the Code in one Place/Class (managing/reducing complexity)
# - Reusability of Code
# - Framework for many other Strategies (only few adjustment required) # Note: You can find a __detailed Tutorial__ on OOP & Classes in the __Appendix__ (at the end of this course). # In[ ]: 


filepath = "bitcoin.csv"
symbol = "BTCUSDT"
start = "2017-08-17"
end = "2021-10-07"
tc = -0.00085
# In[ ]: 
tester = Long_Only_Backtester(filepath=filepath, symbol=symbol,
                              start=start, end=end, tc=tc)
# In[ ]: 
tester
# In[ ]: 
tester.data
# In[ ]: 
tester.test_strategy(percentiles=(90, 5, 20))
# In[ ]: 
tester.plot_results()
# In[ ]: 
tester.results
# In[ ]: 
return_thresh = tester.return_thresh
return_thresh
# In[ ]: 
volume_thresh = tester.volume_thresh
volume_thresh
# In[ ]: 
tester.test_strategy(thresh=(return_thresh, volume_thresh[0], volume_thresh[1]))
# In[ ]: 
tester.optimize_strategy(return_range=(85, 98, 1),
                         vol_low_range=(2, 16, 1),
                         vol_high_range=(16, 35, 1))
# In[ ]: 
tester.results_overview.nlargest(20, "performance")
# In[ ]: 
tester.results
# In[ ]: 
tester.plot_results()
# In[ ]:   # ## Backtesting & Forward Testing (Part 1) # __Great Backtesting Results - Too good to be true?__ # Two major Problems:  # - __Data Snooping / "Over-Optimization"__ -> Will these parameters work with new/fresh data as well? <br>
# - __Look-Ahead-Bias__ -> we know all future price and volume data from day 1 to calculate percentiles/thresholds.  # __Will this strategy outperform Buy-and-Hold in the Future?__ # - wait months/year(s) and analyze then (not an option)
# - split past data into __Backtesting Set__ (optimize Strategy) and __Forward Testing Set__ (test optimized Strategy on fresh data) # __Backtesting & Optimization (until the end of 2020)__ # In[ ]: 
filepath = "bitcoin.csv"
symbol = "BTCUSDT"
start = "2017-08-17"
end = "2020-12-31"
tc = -0.00085
# In[ ]: 
tester = Long_Only_Backtester(filepath=filepath, symbol=symbol, start=start, end=end, tc=tc)
tester
# In[ ]: 
tester.optimize_strategy((85, 98, 1), (2, 16, 1), (16, 35, 1))
# In[ ]: 
many_results = tester.results_overview
many_results
# In[ ]: 
many_results.groupby("returns").performance.mean().plot()
plt.show()
# In[ ]: 
many_results.groupby("vol_low").performance.mean().plot()
plt.show()
# In[ ]: 
many_results.groupby("vol_high").performance.mean().plot()
plt.show()
# In[ ]: 
tester.test_strategy((94, 11, 28))
# In[ ]: 
tester.plot_results()
# In[ ]: 
return_thresh = tester.return_thresh
return_thresh
# In[ ]: 
volume_thresh = tester.volume_thresh
volume_thresh
# In[ ]:   # __Forward Testing (starting at 2021-01-01)__ # we need: thresholds from backtesting # In[ ]: 
return_thresh
# In[ ]: 
volume_thresh
# In[ ]: 
filepath = "bitcoin.csv"
symbol = "BTCUSDT"
start = "2021-01-01"
end = "2021-10-7"
tc = -0.00085
# In[ ]: 
tester = Long_Only_Backtester(filepath=filepath, symbol=symbol, start=start, end=end, tc=tc)
tester
# In[ ]: 
tester.test_strategy(thresh=(return_thresh, volume_thresh[0], volume_thresh[1]))
# In[ ]: 
tester.plot_results()
# In[ ]: 
tester.results.position.value_counts()
# __Reasons for Performance Difference between Backtesting and Forward Testing:__ # - Data Snooping / Over-Optimization (partly)
# - Look-Ahead-Bias (partly) # In[ ]: 
tester.optimize_strategy((85, 98, 1), (5, 15, 1), (15, 35, 1))
# In[ ]: 
tester.plot_results()
# - Overall Regime Change (Patterns can change over time)
# - Strategy not powerful enough # In[ ]:
