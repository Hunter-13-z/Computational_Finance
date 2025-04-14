import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from skfolio.datasets import load_sp500_dataset, load_sp500_implied_vol_dataset
from skfolio.preprocessing import prices_to_returns
import numpy as np
import statsmodels.api as sm

ticker = "AAPL"
start_date = "2010-01-01"
end_date = datetime.datetime.now().strftime("%Y-%m-%d")

data = yf.download(ticker, start=start_date, end=end_date)
data.reset_index(inplace=True)
data["Return"] = np.log(data["Close"] / data["Close"].shift(1))
data["Avg Return"] = data["Return"].expanding().mean()
data["Daily Volatility"] = data["Return"].expanding().var()
# plt.figure(figsize=(14, 6))
# plt.plot(data["Date"], data["Avg Return"], label="Average Daily Log Return", color='blue')
# plt.plot(data["Date"], data["Daily Volatility"], label="Expanding Variance (Volatility²)", color='red')
# plt.title("AAPL: Expanding Mean & Volatility of Daily Log Returns")
# plt.xlabel("Date")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.xticks(rotation=45)
# plt.show()

close_all = data['Close'].values
dates_all = data['Date'].values

def historical_mean_strict(close):
    n = len(close)
    delta_t = 1
    mu = 0
    for i in range(n - 1):
        mu += (1 / delta_t) * (close[i + 1] - close[i]) / close[i]
    mu = mu / n
    return mu

def historical_var_strict(close):
    n = len(close)
    delta_t = 1
    sigma = 0
    for j in range(n - 1):
        sigma += (1 / delta_t) * ((close[j + 1] - close[j]) / close[j])**2
    mu = historical_mean_strict(close)
    sigma = (1 / (n - 1)) * sigma - (n / (n - 1)) * mu**2
    sigma = np.sqrt(sigma)
    return sigma

mu=historical_mean_strict(close_all)
sigma=historical_var_strict(close_all)

print(f"the historical mean of APPLE is:{mu}")
print(f"the historical volatility of APPLE is:{sigma}")


#
# # sigma_Parkinson=0
# # high=data['High'].values
# # low=data['Low'].values
# # k=0
# # close = data['Close'].values
# # n = len(close)
# # while k<n-1:
# #     sigma_Parkinson=sigma_Parkinson+1/(4*math.log(2)*n)*(math.log(high[k]/low[k]))**2
# #     k+=1
#
# def windowfunction(window):
#     rolling_mu = []
#     rolling_sigma = []
#     rolling_dates = []
#
#     for i in range(window, len(close_all)):
#         window_close = close_all[i - window:i + 1]
#         mu = historical_mean_strict(window_close)
#         sigma = historical_var_strict(window_close)
#         rolling_mu.append(mu)
#         rolling_sigma.append(sigma)
#         rolling_dates.append(dates_all[i])
#
#     return rolling_dates,rolling_mu,rolling_sigma
#
# # rolling_dates, rolling_mu,rolling_sigma=windowfunction(30)
# # plt.figure(figsize=(12, 6))
# # plt.plot(rolling_dates, rolling_mu, label="μ̂ (30-day)", color='blue')
# # plt.plot(rolling_dates, rolling_sigma, label="σ̂² (30-day)", color='red')
# # plt.title("30-day Rolling Estimators of μ̂ and σ̂² (AAPL)")
# # plt.xlabel("Date")
# # plt.ylabel("Value")
# # plt.legend()
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()
#
# # def sigma_m_windows(close_all,m):
# #     sigma_list=[]
# #     n=len(close_all)
# #     for i in range(0,n-m,m):
# #         segment=close_all[i:i+m+1]
# #         if len(segment)<m+1:
# #             continue
# #         sigma=historical_var_strict(segment)
# #         sigma_list.append(sigma)
# #     return sum(sigma_list)/len(sigma_list)
# #
# # m_list = [i * 50 for i in range(1, 40)]  # m = 5, 10, ..., 95
# # sigma_m=[sigma_m_windows(close_all,m) for m in m_list]
# # plt.plot(m_list, sigma_m, marker='o')
# # plt.title("Volatility Signature Plot (σ̂² vs. m)")
# # plt.xlabel("Sampling interval m (days)")
# # plt.ylabel("Average Rolling σ̂²")
# # plt.grid(True)
# # plt.show()
#
# prices = load_sp500_dataset()
# implied_vol = load_sp500_implied_vol_dataset()
#
# prices_aapl = prices["AAPL"].loc["2010":]
# iv_aapl = implied_vol["AAPL"].loc["2010":]
#
# rolling_dates,_,realized_vol = windowfunction(30)
# rv_series = pd.Series(realized_vol, index=rolling_dates)
#
# # 合并成一个 DataFrame
# df = pd.DataFrame({
#     "Realized Variance (30d)": rv_series,
#     "Implied Volatility (approx daily var)": (iv_aapl/np.sqrt(252))
# }).dropna()
#
# # 可视化
# plt.figure(figsize=(12, 6))
# plt.plot(df.index, df["Realized Variance (30d)"], label="Realized Volatility", color='blue')
# plt.plot(df.index, df["Implied Volatility (approx daily var)"], label="Implied Volatility", color='red')
# plt.title("AAPL: Realized vs. Implied Volatility")
# plt.xlabel("Date")
# plt.ylabel("Volatility")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# spx_symbol = "^SPX"
# today = "2025-04-08"
# end_date = pd.to_datetime(today)
# start_date=end_date-datetime.timedelta(days=365)
#
# spx_data=yf.download(spx_symbol,start_date,end_date)
# lastBusDay=spx_data.index[-1]
# vix_data=yf.download("^VIX",start=lastBusDay,end=lastBusDay+datetime.timedelta(days=1))
#
# spx_ticker=yf.Ticker("^SPX")
# expiry_date="2025-05-08"
# chain=spx_ticker.option_chain(expiry_date)
# calls_df=chain.calls
# puts_df=chain.puts
#
# # print("Calls Head:")
# # print(calls_df.head())
# #
# # print("Puts Head:")
# # print(puts_df.head())
#
# calls_df.to_csv("spx_call.csv",index=False)
# puts_df.to_csv("spx_puts.csv",index=False)
#
# tau=30/365
# r=0.05
# calls_df['midPrice'] = (calls_df['bid'] + calls_df['ask']) / 2
# puts_df['midPrice'] = (puts_df['bid'] + puts_df['ask']) / 2
# CallandPuts = pd.merge(
#     calls_df[['strike', 'midPrice']],
#     puts_df[['strike', 'midPrice']],
#     on='strike',
#     suffixes=('_call', '_put')
# )
# CallandPuts['abs_diff'] = np.abs(CallandPuts['midPrice_call'] - CallandPuts['midPrice_put'])
# F_strike=CallandPuts.loc[CallandPuts['abs_diff'].idxmin(),'strike']
#
# puts=CallandPuts[CallandPuts['strike']<F_strike]
# calls=CallandPuts[CallandPuts['strike']>F_strike]
# dK_puts=np.diff(puts['strike'])
# dK_calls=np.diff(calls['strike'])
# int_puts=np.sum((puts['midPrice_put'].values[:-1]/(puts['strike'].values[:-1]**2))*dK_puts)
# int_calls=np.sum((calls['midPrice_call'].values[:-1]/(calls['strike'].values[:-1]**2))*dK_calls)
#
# vix_est_squared=(2*np.exp(r*tau)/tau)*(int_puts+int_calls)
# vix_est=np.sqrt(vix_est_squared)
#
# print(vix_est*100,vix_data['Close'])
# vix_hist = yf.download("^VIX", start="2010-01-01", end="2025-04-08")["Close"]
# rolling_dates,_,realized_vol = windowfunction(30)
# rv_series = pd.Series(realized_vol, index=pd.to_datetime(rolling_dates), name="Realized_Var")
# realized_var = rv_series ** 2*252
# vix_series = pd.Series((vix_hist / 100).squeeze(), index=vix_hist.index, name="VIX")
# df_rvandvix = pd.concat([realized_var, vix_series], axis=1).dropna()


# plt.figure(figsize=(12, 6))
# plt.plot(df_rvandvix.index, df_rvandvix["Realized_Var"], label="Realized Variance (30d)")
# plt.plot(df_rvandvix.index, df_rvandvix["VIX"]**2, label="VIX²(year)")
# plt.title("Realized Variance vs. VIX²")
# plt.xlabel("Date")
# plt.ylabel("Variance")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# corr = df_rvandvix["Realized_Var"].corr(df_rvandvix["VIX"])
# print(f"Correlation：{corr:.3f}")

# prices=load_sp500_dataset()
# implied_vol=load_sp500_implied_vol_dataset()
# X=prices_to_returns(prices)
# X=X.loc["2010":]
# implied_vol.tail()
# ticker = "AAPL"
# returns = X[ticker]
# sp500_returns = pd.Series(returns.squeeze(), index=returns.index, name="SPX")
# df_SPXandVIX = pd.concat([sp500_returns, vix_series], axis=1).dropna().astype(float)
# df_SPXandRV = pd.concat([sp500_returns, rv_series], axis=1).dropna().astype(float)
#
# X_vix = sm.add_constant(df_SPXandVIX["SPX"])
# X_rv = sm.add_constant(df_SPXandRV["SPX"])
# model_vix = sm.OLS(df_SPXandVIX["VIX"], X_vix).fit()
# model_rv = sm.OLS(df_SPXandRV["Realized_Var"], X_rv).fit()
# print("=== VIX 回归结果 ===")
# print(model_vix.summary())
#
# print("\n=== Realized Var 回归结果 ===")
# print(model_rv.summary())