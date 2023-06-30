#!/usr/bin/env python
# coding: utf-8

# # Implementation - Futures Testnet API

# __Insert your Futures Testnet Credentials here__:

# In[ ]:


api_key = "insert here"
secret_key = "insert here"


# In[ ]:





# ## Introduction to the Futures Testnet API

# Find more information on the Futures API here:

# https://binance-docs.github.io/apidocs/futures/en/#change-log

# In[ ]:


from binance.client import Client
import pandas as pd


# In[ ]:


client = Client(api_key = api_key, api_secret = secret_key, tld = "com", testnet = True) # Testnet!!!


# __Simple Rule:__ <br>
# - Use your __Testnet Credentials with testnet = True__ and your __"Real Account" Credentials with testnet = False__ (default)

# In[ ]:


client


# __USD-margined Account__

# In[ ]:


client.futures_account() # Account details


# In[ ]:


pd.DataFrame(client.futures_account()["assets"])


# In[ ]:


client.futures_account_balance() # Asset Balance details


# In[ ]:


client.futures_position_information() # current Open Positions


# In[ ]:


client.futures_position_information(symbol = "BTCUSDT") # current Open Positions for one symbol


# __Coin-margined Account__

# In[ ]:


client.futures_coin_account() # Account details


# In[ ]:


client.futures_coin_account_balance()  # Asset Balance details


# In[ ]:


client.futures_coin_position_information() # current Open Positions


# In[ ]:





# ## Changing Settings and Modes

# In[ ]:


client


# __Target Leverage__

# In[ ]:


client.futures_position_information(symbol = "BTCUSDT")


# In[ ]:


client.futures_change_leverage(symbol = "BTCUSDT", leverage = 5)


# __Margin Mode (ISOLATED vs. CROSSED)__

# In[ ]:


client.futures_change_margin_type(symbol = "BTCUSDT", marginType = "ISOLATED")


# In[ ]:


client.futures_change_margin_type(symbol = "BTCUSDT", marginType = "CROSSED")


# __Position Mode (Hedging vs. Netting)__

# In[ ]:


client.futures_get_position_mode() # position mode ()


# In[ ]:


client.futures_change_position_mode(dualSidePosition = True)


# In[ ]:





# ## Placing Market Orders (Part 1)

# In[ ]:


client


# In[ ]:


client.futures_position_information(symbol = "BTCUSDT") # no open position


# In[ ]:


client.futures_get_open_orders(symbol = "BTCUSDT") # no open orders


# __More information on placing Orders via the Futures API:__

# https://binance-docs.github.io/apidocs/futures/en/#new-order-trade

# In[ ]:


# client.futures_create_order()


# __Go Long 0.01 BTCUSDT, Leverage = 10__

# In[ ]:


leverage = 10
size = 0.01


# In[ ]:


client.futures_change_leverage(symbol = "BTCUSDT", leverage = leverage)


# In[ ]:


order_open = client.futures_create_order(symbol = "BTCUSDT", side = "BUY",
                                         type = "MARKET", quantity = size)
order_open


# In[ ]:


client.futures_get_order(symbol = "BTCUSDT", orderId = order_open["orderId"]) # check order status


# In[ ]:


client.futures_position_information(symbol = "BTCUSDT") 


# In[ ]:


order_close = client.futures_create_order(symbol = "BTCUSDT", side = "SELL",
                                          type = "MARKET", quantity = size, reduce_Only = True)
order_close


# In[ ]:


client.futures_get_order(symbol = "BTCUSDT", orderId = order_close["orderId"]) # check order status


# In[ ]:


client.futures_position_information(symbol = "BTCUSDT") 


# In[ ]:





# ## Trade and Income History

# In[ ]:


client


# https://binance-docs.github.io/apidocs/futures/en/#account-trade-list-user_data

# In[ ]:


client.futures_account_trades(symbol = "BTCUSDT") # get all trades in the last 7 days


# In[ ]:


df = pd.DataFrame(client.futures_account_trades(symbol = "BTCUSDT"))
df


# In[ ]:


df.tail(2) # two most recent trades


# In[ ]:


client.futures_income_history(symbol = "BTCUSDT") # recent income history (7 days)


# In[ ]:


pd.DataFrame(client.futures_income_history(symbol = "BTCUSDT")).tail(3)


# In[ ]:





# ## Placing Market Orders (Part 2)

# In[ ]:


client


# __Go Short 0.01 BTCUSDT, Leverage = 15__

# In[ ]:


leverage = 15
size = 0.01


# In[ ]:


client.futures_change_leverage(symbol = "BTCUSDT", leverage = leverage)


# In[ ]:


order_open = client.futures_create_order(symbol = "BTCUSDT", side = "SELL",
                                         type = "MARKET", quantity = size)
order_open


# In[ ]:


client.futures_position_information(symbol = "BTCUSDT") 


# In[ ]:


order_close = client.futures_create_order(symbol = "BTCUSDT", side = "BUY",
                                          type = "MARKET", quantity = size, reduceOnly = True)
order_close


# In[ ]:


df = pd.DataFrame(client.futures_account_trades(symbol = "BTCUSDT"))


# In[ ]:


df.tail(2)


# In[ ]:





# ## Getting Historical Futures Market Data

# In[ ]:


from binance.client import Client
import pandas as pd


# In[ ]:


client = Client(api_key = api_key, api_secret = secret_key, tld = "com", testnet = True) # Testnet!!!


# valid intervals - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

# __Spot__

# In[ ]:


client.get_historical_klines(symbol = "BTCUSDT", interval = "1d", 
                             start_str = "2020-01-01", end_str = None, limit = 1000)


# __Futures__

# In[ ]:


client.futures_historical_klines(symbol = "BTCUSDT", interval = "1d",
                                 start_str = "2020-01-01", end_str = None, limit = 1000)


# In[ ]:


def get_history(symbol, interval, start, end = None):
    bars = client.futures_historical_klines(symbol = symbol, interval = interval,
                                        start_str = start, end_str = end, limit = 1000)
    df = pd.DataFrame(bars)
    df["Date"] = pd.to_datetime(df.iloc[:,0], unit = "ms")
    df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                  "Clos Time", "Quote Asset Volume", "Number of Trades",
                  "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df.set_index("Date", inplace = True)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors = "coerce")
    
    return df


# In[ ]:


data = get_history(symbol = "BTCUSDT", interval = "1d", start = "2020-01-01")
data


# In[ ]:


data.info()


# In[ ]:





# ## Streaming Future Prices in real-time

# In[ ]:


from binance import ThreadedWebsocketManager
import pandas as pd


# In[ ]:


df = pd.DataFrame(columns = ["Open", "High", "Low", "Close", "Volume", "Complete"])
df


# In[ ]:


def stream_candles(msg):
    ''' define how to process incoming WebSocket messages '''
    
    # extract the required items from msg
    event_time = pd.to_datetime(msg["E"], unit = "ms")
    start_time = pd.to_datetime(msg["k"]["t"], unit = "ms")
    first   = float(msg["k"]["o"])
    high    = float(msg["k"]["h"])
    low     = float(msg["k"]["l"])
    close   = float(msg["k"]["c"])
    volume  = float(msg["k"]["v"])
    complete=       msg["k"]["x"]
    
    # print out
    print("Time: {} | Price: {}".format(event_time, close))
    
    # feed df (add new bar / update latest bar)
    df.loc[start_time] = [first, high, low, close, volume, complete]


# In[ ]:


twm = ThreadedWebsocketManager(testnet = True)
twm.start()


# valid intervals - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

# __Spot__

# In[ ]:


# twm.start_kline_socket(callback = stream_candles, symbol = "BTCUSDT", interval = "1m")


# __Futures__

# In[ ]:


twm.start_kline_futures_socket(callback = stream_candles, symbol = "BTCUSDT", interval = "1m")


# In[ ]:


twm.stop()


# In[ ]:


df


# In[ ]:


df.info()


# In[ ]:





# ## The Futures Trader

# _Disclaimer: <br>
# The following illustrative examples are for general information and educational purposes only. <br>
# It is neither investment advice nor a recommendation to trade, invest or take whatsoever actions.<br>
# The below code should only be used in combination with the Binance Futures Testnet and NOT with a Live Trading Account._

# In[ ]:


from binance.client import Client
from binance import ThreadedWebsocketManager
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time


# In[ ]:


class FuturesTrader():  # Triple SMA Crossover
    
    def __init__(self, symbol, bar_length, sma_s, sma_m, sma_l, units, position = 0, leverage = 5):
        
        self.symbol = symbol
        self.bar_length = bar_length
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        self.units = units
        self.position = position
        self.leverage = leverage # NEW
        self.cum_profits = 0 # NEW
        #self.trades = 0 
        #self.trade_values = []
        
        #*****************add strategy-specific attributes here******************
        self.SMA_S = sma_s
        self.SMA_M = sma_m
        self.SMA_L = sma_l
        #************************************************************************
    
    def start_trading(self, historical_days):
        
        client.futures_change_leverage(symbol = self.symbol, leverage = self.leverage) # NEW
        
        self.twm = ThreadedWebsocketManager(testnet = True) # testnet
        self.twm.start()
        
        if self.bar_length in self.available_intervals:
            self.get_most_recent(symbol = self.symbol, interval = self.bar_length,
                                 days = historical_days)
            self.twm.start_kline_futures_socket(callback = self.stream_candles,
                                        symbol = self.symbol, interval = self.bar_length) # Adj: start_kline_futures_socket
        # "else" to be added later in the course 
    
    def get_most_recent(self, symbol, interval, days):
    
        now = datetime.utcnow()
        past = str(now - timedelta(days = days))
    
        bars = client.futures_historical_klines(symbol = symbol, interval = interval,
                                            start_str = past, end_str = None, limit = 1000) # Adj: futures_historical_klines
        df = pd.DataFrame(bars)
        df["Date"] = pd.to_datetime(df.iloc[:,0], unit = "ms")
        df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                      "Clos Time", "Quote Asset Volume", "Number of Trades",
                      "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df.set_index("Date", inplace = True)
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors = "coerce")
        df["Complete"] = [True for row in range(len(df)-1)] + [False]
        
        self.data = df
    
    def stream_candles(self, msg):
        
        # extract the required items from msg
        event_time = pd.to_datetime(msg["E"], unit = "ms")
        start_time = pd.to_datetime(msg["k"]["t"], unit = "ms")
        first   = float(msg["k"]["o"])
        high    = float(msg["k"]["h"])
        low     = float(msg["k"]["l"])
        close   = float(msg["k"]["c"])
        volume  = float(msg["k"]["v"])
        complete=       msg["k"]["x"]
        
        # print out
        print(".", end = "", flush = True) 
    
        # feed df (add new bar / update latest bar)
        self.data.loc[start_time] = [first, high, low, close, volume, complete]
        
        # prepare features and define strategy/trading positions whenever the latest bar is complete
        if complete == True:
            self.define_strategy()
            self.execute_trades()
        
    def define_strategy(self):
        
        data = self.data.copy()
        
        #******************** define your strategy here ************************
        data = data[["Close"]].copy()
        
        data["SMA_S"] = data.Close.rolling(window = self.SMA_S).mean()
        data["SMA_M"] = data.Close.rolling(window = self.SMA_M).mean()
        data["SMA_L"] = data.Close.rolling(window = self.SMA_L).mean()
        
        data.dropna(inplace = True)
                
        cond1 = (data.SMA_S > data.SMA_M) & (data.SMA_M > data.SMA_L)
        cond2 = (data.SMA_S < data.SMA_M) & (data.SMA_M < data.SMA_L)
        
        data["position"] = 0
        data.loc[cond1, "position"] = 1
        data.loc[cond2, "position"] = -1
        #***********************************************************************
        
        self.prepared_data = data.copy()
    
    def execute_trades(self): # Adj! 
        if self.prepared_data["position"].iloc[-1] == 1: # if position is long -> go/stay long
            if self.position == 0:
                order = client.futures_create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING LONG")  
            elif self.position == -1:
                order = client.futures_create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = 2 * self.units)
                self.report_trade(order, "GOING LONG")
            self.position = 1
        elif self.prepared_data["position"].iloc[-1] == 0: # if position is neutral -> go/stay neutral
            if self.position == 1:
                order = client.futures_create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL") 
            elif self.position == -1:
                order = client.futures_create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL")
            self.position = 0
        if self.prepared_data["position"].iloc[-1] == -1: # if position is short -> go/stay short
            if self.position == 0:
                order = client.futures_create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING SHORT") 
            elif self.position == 1:
                order = client.futures_create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = 2 * self.units)
                self.report_trade(order, "GOING SHORT")
            self.position = -1
    
    def report_trade(self, order, going): # Adj!
        
        time.sleep(0.1)
        order_time = order["updateTime"]
        trades = client.futures_account_trades(symbol = self.symbol, startTime = order_time)
        order_time = pd.to_datetime(order_time, unit = "ms")
        
        # extract data from trades object
        df = pd.DataFrame(trades)
        columns = ["qty", "quoteQty", "commission","realizedPnl"]
        for column in columns:
            df[column] = pd.to_numeric(df[column], errors = "coerce")
        base_units = round(df.qty.sum(), 5)
        quote_units = round(df.quoteQty.sum(), 5)
        commission = -round(df.commission.sum(), 5)
        real_profit = round(df.realizedPnl.sum(), 5)
        price = round(quote_units / base_units, 5)
        
        # calculate cumulative trading profits
        self.cum_profits += round((commission + real_profit), 5)
        
        # print trade report
        print(2 * "\n" + 100* "-")
        print("{} | {}".format(order_time, going)) 
        print("{} | Base_Units = {} | Quote_Units = {} | Price = {} ".format(order_time, base_units, quote_units, price))
        print("{} | Profit = {} | CumProfits = {} ".format(order_time, real_profit, self.cum_profits))
        print(100 * "-" + "\n")


# In[ ]:


client = Client(api_key = api_key, api_secret = secret_key, tld = "com", testnet = True)


# In[ ]:


symbol = "BTCUSDT"
bar_length = "1m"
sma_s = 10
sma_m = 20
sma_l = 50
units = 0.001
position = 0
leverage = 10


# In[ ]:


trader = FuturesTrader(symbol = symbol, bar_length = bar_length,
                       sma_s = sma_s, sma_m = sma_m, sma_l = sma_l, 
                       units = units, position = position, leverage = leverage)


# In[ ]:


trader.start_trading(historical_days = 1/24)


# In[ ]:


trader.twm.stop()


# In[ ]:


trader.prepared_data


# In[ ]:




