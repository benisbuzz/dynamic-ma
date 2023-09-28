import pandas as pd
import numpy as np
import talib as ta



class DynamicMa:
    def __init__(self, prices: pd.Series, timeperiods: list[int]):
        self.price_data = prices
        self.ma_gradient_data = {
            timeperiod: ta.EMA(prices, timeperiod=timeperiod).pct_change()
            for timeperiod in timeperiods
        }

    def get_all_postions(self) -> dict[int, pd.Series]:
        gradients = self.ma_gradient_data
        return {
            timeperiod: (~(gradient < 0) & ~(gradient.isna())).astype(int)
            for timeperiod, gradient in gradients.items()
        }

    def get_net_position(self, all_positions: dict[int, pd.Series]) -> pd.Series:
        all_positions = self.get_all_postions()
        df = pd.concat(all_positions.values(), axis=1)
        return df.mean(axis=1)

    def get_daily_return(self, net_positions: pd.Series) -> pd.Series:
        return self.price_data.pct_change() * net_positions.shift(1)

    def get_cum_return(self, daily_return: pd.Series) -> pd.Series:
        return (daily_return+1).cumprod()
    
    def get_sharpe(self, daily_return: pd.Series) -> float:
        return np.mean(daily_return) / np.std(daily_return) * np.sqrt(252)
    
    def get_profit_factor(self, daily_return: pd.Series) -> float:
        return len(daily_return[daily_return > 0]) / len(daily_return[daily_return < 0])
    
    def get_std(self, daily_return: pd.Series) -> float:
        return np.std(daily_return)
    
    def get_max_drawdown(self, cum_return: pd.Series) -> float:
        prices = self.price_data
        dd = (cum_return.cummax() - cum_return)
        end = dd.idxmax()
        start = dd[:end][dd == 0].index[-1]
        return (prices[end] - prices[start]) / prices[start]
