import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def Stockprice(S0,r,sigma,delta_t,T):
    prices=[S0]
    for i in range(T-1):
        Z=np.random.randn()
        S_next=prices[-1]**np.exp((r-1/2*(sigma**2))*delta_t+sigma*np.sqrt(delta_t)*Z)
        prices.append(S_next)
    return prices
S0 = 100
r = 0.06
sigma = 0.2
delta_t = 1 / 252
T = 252
K=99
StockPrice=Stockprice(S0, r, sigma, delta_t, T)

def BlackScholes(S,K,T,r,sigma):
    if T==0:
        return max(S-K,0),1.0 if S>K else 0.0
    d1=(np.log(S/K)+(r+1/2*(sigma**2)*T))/(sigma*np.sqrt(T))
    d2=d1 - sigma * np.sqrt(T)
    delta=norm.cdf(d1)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return delta,call_price

callprice,delta0=BlackScholes(S0,K,T,r,sigma)
portfolio={
    "day":0,
    "stockprice":100,
    "callprice":callprice,
    "shares":delta0,
    "cash":callprice-S0*delta0,
    "value":...
}

def realize_portfolio(portfolio,hedge_freq,sigma):
    # history = []

    for t in range(1,T):
        S=StockPrice[t]
        t1=T-t
        if t % hedge_freq == 0:
            _, delta = BlackScholes(S, K, t1, r, sigma)
            delta_diff = delta - portfolio["shares"]
            portfolio["cash"] -= delta_diff * S
            portfolio["shares"] = delta

        portfolio["stockprice"]=S
        portfolio["day"]=t
        portfolio["value"]=portfolio["shares"] * S + portfolio["cash"]

        # history.append(portfolio.copy())
    return portfolio

final_price = StockPrice[-1]
final_option_value = max(final_price - K, 0)
errors = []
# frequencies = [1, 2,3,4,5,6,7]
#
# for freq in frequencies:
#     portfolio1=realize_portfolio(portfolio,freq,sigma)
#     final_portfolio_value = portfolio["shares"] * final_price + portfolio["cash"]
#     hedge_error = final_portfolio_value - final_option_value
#     errors.append(hedge_error)
#
# plt.plot(frequencies, np.abs(errors), marker='o')
# plt.xlabel("Hedge Frequency (days)")
# plt.ylabel("Absolute Hedging Error")
# plt.grid(True)
# plt.show()

option_volatility=[0.01,0.1,0.2,0.4,0.6,0.8]
for option_sigma in option_volatility:
    callprice, delta0 = BlackScholes(S0, K, T, r, option_sigma)
    portfolio_0 = {
        "day": 0,
        "stockprice": 100,
        "callprice": callprice,
        "shares": delta0,
        "cash": callprice - S0 * delta0,
        "value": ...
    }
    portfolio = realize_portfolio(portfolio_0,1,option_sigma)
    final_portfolio_value = portfolio["shares"] * final_price + portfolio["cash"]
    hedge_error = final_portfolio_value - final_option_value
    errors.append(hedge_error)

plt.plot(option_volatility, np.abs(errors), marker='o')
plt.title("Hedging Error vs. Option Volatility")
plt.xlabel("Option Volatility")
plt.ylabel("Absolute Hedging Error")
plt.grid(True)
plt.show()

