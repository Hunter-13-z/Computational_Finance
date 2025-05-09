import warnings
warnings.filterwarnings("ignore")
import numpy as np

import scipy as scp
import scipy.stats as ss
import matplotlib.pyplot as plt
import scipy.special as scps
from statsmodels.graphics.gofplots import qqplot
from scipy.linalg import cholesky
from functools import partial
#from FMNM.probabilities import Heston_pdf, Q1, Q2
#from FMNM.cython.heston import Heston_paths_log, Heston_paths
from scipy.optimize import minimize
#from FMNM.BS_pricer import BS_pricer
#from FMNM.Parameters import Option_param
#from FMNM.Processes import Diffusion_process
from IPython.display import display
import sympy
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import minimize
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.special import gamma as gamma_func
import math
from tqdm import tqdm

sympy.init_printing()


def display_matrix(m):
    display(sympy.Matrix(m))

#time
np.random.seed(seed=42)

N = 10000  # time steps
M = 10000  # number of paths
T = 1
T_vec, dt = np.linspace(0, T, N, retstep=True)
dt_sq = np.sqrt(dt)

S0 = 100  # spot price
X0 = np.log(S0)  # log price
v0 = 0.04  # spot variance

r = 0.05
mu = 0.1  # drift
rho = -0.2  # correlation coefficient
kappa = 2  # mean reversion coefficient
theta = 0.04  # long-term variance
xi = 0  # Vol of Vol - Volatility of instantaneous variance
std_asy = np.sqrt(theta * xi**2 / (2 * kappa))  # asymptotic standard deviation for the CIR process
assert 2 * kappa * theta > xi**2  # Feller condition

# Generate random Brownian Motion
MU = np.array([0, 0])
COV = np.matrix([[1, rho], [rho, 1]])
Z = ss.multivariate_normal.rvs(mean=MU, cov=COV, size=(N - 1, M))
Z_S = Z[:, :, 0]  # Stock Brownian motion:     W_1
Z_V = Z[:, :, 1]  # Variance Brownian motion:  W_2

# Initialize vectors
S_euler = np.zeros((N, M))
V_euler = np.zeros((N, M))
S_milstein = np.zeros((N, M))
V_milstein = np.zeros((N, M))

S_euler[0, :] = S0
V_euler[0, :] = v0
S_milstein[0, :] = S0
V_milstein[0, :] = v0

v = np.zeros(N)

# Generate paths with progress bar
for t in tqdm(range(0, N - 1), desc="Simulating paths"):
    # Euler:
    Vt_e = np.maximum(V_euler[t, :], 0)
    sqrt_V_e = np.sqrt(Vt_e)

    V_euler[t + 1, :] = V_euler[t, :] + kappa * (theta - Vt_e) * dt + xi * sqrt_V_e * dt_sq * Z_V[t, :]
    V_euler[t + 1, :] = np.abs(V_euler[t + 1, :])

    S_euler[t + 1, :] = S_euler[t, :] + r * S_euler[t, :] * dt + sqrt_V_e * dt_sq * S_euler[t, :] * Z_S[t, :]

    # Milstein:
    Vt_m = np.maximum(V_milstein[t, :], 0)
    sqrt_V_m = np.sqrt(Vt_m)

    V_milstein[t + 1, :] = V_milstein[t, :] + kappa * (theta - Vt_m) * dt + xi * sqrt_V_m * dt_sq * Z_V[t, :] + 0.25 * xi**2 * dt * (Z_V[t, :]**2 - 1)
    V_milstein[t + 1, :] = np.abs(V_milstein[t + 1, :])

    S_milstein[t + 1, :] = S_milstein[t, :] + r * S_milstein[t, :] * dt + sqrt_V_m * dt_sq * S_milstein[t, :] * Z_S[t, :] + 0.5 * Vt_m * S_milstein[t, :] * dt * (Z_S[t, :]**2 - 1)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
#
# ax1.plot(T_vec, S_euler[:, :100])
# ax1.set_title("Stock Price Paths (Euler Scheme)")
# ax1.set_xlabel("Time")
# ax1.set_ylabel("S(t)")
#
# ax2.plot(T_vec, S_milstein[:, :100])
# ax2.set_title("Stock Price Paths (Milstein Scheme)")
# ax2.set_xlabel("Time")
# ax2.set_ylabel("S(t)")
#
# plt.tight_layout()
# plt.show()

# fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 5))
#
# ax3.plot(T_vec, V_euler[:, :4], alpha=0.7)
# ax3.set_title("Variance Paths (Euler Scheme)")
# ax3.set_xlabel("Time")
# ax3.set_ylabel("V(t)")
# ax3.axhline(theta, color="black", linestyle="--", label="Long-run mean θ")
# ax3.axhline(theta + std_asy, color="black", linewidth=0.8)
# ax3.axhline(theta - std_asy, color="black", linewidth=0.8)
# ax3.legend()
#
# ax4.plot(T_vec, V_milstein[:, :4], alpha=0.7)
# ax4.set_title("Variance Paths (Milstein Scheme)")
# ax4.set_xlabel("Time")
# ax4.set_ylabel("V(t)")
# ax4.axhline(theta, color="black", linestyle="--", label="Long-run mean θ")
# ax4.axhline(theta + std_asy, color="black", linewidth=0.8)
# ax4.axhline(theta - std_asy, color="black", linewidth=0.8)
# ax4.legend()
#
# plt.tight_layout()
# plt.show()

#Average Price
K = 100
avg_price_euler = np.mean(S_euler[1:, :], axis=0)
avg_price_milstein = np.mean(S_milstein[1:, :], axis=0)
payoff_euler = np.maximum(avg_price_euler - K, 0)
payoff_milstein = np.maximum(avg_price_milstein - K, 0)
# #Display first 10 average prices and payoffs
# df = pd.DataFrame({
#     "Avg Price (Euler)": avg_price_euler[:10],
#     "Payoff (Euler)": payoff_euler[:10],
#     "Avg Price (Milstein)": avg_price_milstein[:10],
#     "Payoff (Milstein)": payoff_milstein[:10],
# })
# display(df.round(2))

#Monte Carlo Estimator and its Standard Error
price_euler = np.exp(-r * T) * np.mean(payoff_euler)
std_euler = np.std(np.exp(-r * T) * payoff_euler) / np.sqrt(M)

price_milstein = np.exp(-r * T) * np.mean(payoff_milstein)
std_milstein = np.std(np.exp(-r * T) * payoff_milstein) / np.sqrt(M)
#visualization
df_results = pd.DataFrame({
    "Method": ["Euler", "Milstein"],
    "Price Estimate": [price_euler, price_milstein],
    "Standard Error": [std_euler, std_milstein],
})
pd.set_option("display.precision", 6)
display(df_results)

#Comparison to GBM
K = 100
avg_price_euler_null = np.mean(S_euler[1:, :], axis=0)
avg_price_milstein_null = np.mean(S_milstein[1:, :], axis=0)
payoff_euler_null = np.maximum(avg_price_euler_null - K, 0)
payoff_milstein_null = np.maximum(avg_price_milstein_null - K, 0)

# GBM
sigma = np.sqrt(theta)
S_gbm = np.zeros((N, M))
S_gbm[0, :] = S0

for t in range(1, N):
    Z = np.random.normal(size=M)
    S_gbm[t, :] = S_gbm[t-1, :] * np.exp((r - 0.5 * sigma**2) * dt + sigma * dt_sq * Z)

avg_price_gbm = np.mean(S_gbm[1:, :], axis=0)
payoff_gbm = np.maximum(avg_price_gbm - K, 0)
price_gbm = np.exp(-r * T) * np.mean(payoff_gbm)
std_gbm = np.std(np.exp(-r * T) * payoff_gbm) / np.sqrt(M)

print("Heston (ξ=0) - Euler     :", np.exp(-r * T) * np.mean(payoff_euler_null))
print("Heston (ξ=0) - Milstein  :", np.exp(-r * T) * np.mean(payoff_milstein_null))
print("GBM (Simulated path)     :", price_gbm)
print("Std Error (GBM)          :", std_gbm)
#task 2
#2-1
sigma = np.sqrt(theta)

#2-2
K = 100
sigma_sq = sigma**2
sigma_tilde = sigma * np.sqrt((2 * N + 1) / (6 * (N + 1)))
r_tilde = 0.5 * (r - 0.5 * sigma_sq + 0.5 * sigma_tilde**2)

d1 = (np.log(S0 / K) + (r_tilde + 0.5 * sigma_tilde**2) * T) / (sigma_tilde * np.sqrt(T))
d2 = d1 - sigma_tilde * np.sqrt(T)
price = S0 * np.exp((r_tilde - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
print(price)

#2-3


# #task3
# #3-1
# # Initialize GBM paths (Euler and Milstein)
# S_gbm_euler = np.zeros((N, M))
# S_gbm_milstein = np.zeros((N, M))
# S_gbm_euler[0, :] = S0
# S_gbm_milstein[0, :] = S0
#
# sigma_bs = np.sqrt(theta)  # constant volatility
#
# # Simulate GBM paths using Euler and Milstein methods
# for t in tqdm(range(0, N - 1), desc="Simulating GBM paths"):
#     Z = Z_S[t, :]
#     # Euler
#     S_gbm_euler[t+1,:] = S_gbm_euler[t,:] + r * S_gbm_euler[t, :] * dt + sigma_bs * dt_sq * S_gbm_euler[t, :] * Z
#     # Milstein
#     S_gbm_milstein[t+1,:] = S_gbm_milstein[t,:] + r * S_gbm_milstein[t, :] * dt + sigma_bs * dt_sq * S_gbm_milstein[t,:] * Z + 0.5 * sigma_bs ** 2 * S_gbm_milstein[t,:] * dt * (Z ** 2 - 1)
# # fig, (ax5, ax6) = plt.subplots(1, 2, figsize=(18, 5))
# #
# # ax5.plot(T_vec, S_gbm_euler[:, :100])
# # ax5.set_title("GBM Stock Price Paths (Euler Scheme)")
# # ax5.set_xlabel("Time")
# # ax5.set_ylabel("S(t)")
# #
# # ax6.plot(T_vec, S_gbm_milstein[:, :100])
# # ax6.set_title("GBM Stock Price Paths (Milstein Scheme)")
# # ax6.set_xlabel("Time")
# # ax6.set_ylabel("S(t)")
# #
# # plt.tight_layout()
# # plt.show()
#
# #3-2
# #Heston Payoff
# K = 100
# avg_price_euler = np.mean(S_euler[1:, :], axis=0)
# avg_price_milstein = np.mean(S_milstein[1:, :], axis=0)
# payoff_euler = np.maximum(avg_price_euler - K, 0)
# payoff_milstein = np.maximum(avg_price_milstein - K, 0)
# #GBM Payoff
# avg_price_euler_gbm = np.mean(S_gbm_euler[1:, :], axis=0)
# avg_price_milstein_gbm = np.mean(S_gbm_milstein[1:, :], axis=0)
# payoff_euler_gbm = np.maximum(avg_price_euler_gbm - K, 0)
# payoff_milstein_gbm = np.maximum(avg_price_milstein_gbm - K, 0)
# # df_compare = pd.DataFrame({
# #     'Heston_Euler': payoff_euler[:10],
# #     'Heston_Milstein': payoff_milstein[:10],
# #     'GBM_Euler': payoff_euler_gbm[:10],
# #     'GBM_Milstein': payoff_milstein_gbm[:10],
# # })
# #
# # # 显示结果
# # print("Payoff Comparison for First 10 Paths:")
# # display(df_compare)
#
# #3-3
# Y_bar_euler = np.mean(payoff_euler)
# Y_bar_milstein = np.mean(payoff_milstein)
# X_bar_euler = np.mean(payoff_euler_gbm)
# X_bar_milstein = np.mean(payoff_milstein_gbm)
# C_G = price
# c = 1
# C_CV1 = Y_bar_euler + c * (C_G - X_bar_euler)
# C_CV2 = Y_bar_milstein + c * (C_G - X_bar_milstein)
# # print(C_CV1,C_CV2)
#
# #3-4
# var_cv1 = np.var(np.exp(-r * T) * (payoff_euler + c * (C_G - payoff_euler_gbm))) / M
# var_cv2 = np.var(np.exp(-r * T) * (payoff_milstein + c * (C_G - payoff_milstein_gbm))) / M
# print(var_cv1,var_cv2)
#
# std_cv1 = np.std(np.exp(-r * T) * (payoff_euler + c * (C_G - payoff_euler_gbm))) / np.sqrt(M)
# std_cv2 = np.std(np.exp(-r * T) * (payoff_milstein + c * (C_G - payoff_milstein_gbm))) / np.sqrt(M)
# print(std_cv1,std_cv2)
#
# df_2 = pd.DataFrame([
#         {"Method": "Euler (Heston)", "Price": price_euler, "Std Error": std_euler},
#         {"Method": "Milstein (Heston)", "Price": price_milstein, "Std Error": std_milstein},
#         {"Method": "Control Variate-Euler", "Price": np.exp(-r * T) * C_CV1, "Std Error": std_cv1},
#         {"Method": "Control Variate-Milstein", "Price": np.exp(-r * T) * C_CV2, "Std Error": std_cv2}
#     ])
# print(df_2)

# def MC(M_values, N_values, xi_values, rho_values, K_values):
#     M = M_values  # number of paths
#     N = N_values  # time steps
#
#     xi = xi_values
#     rho = rho_values
#     K = K_values
#
#     T = 1
#     T_vec, dt = np.linspace(0, T, N, retstep=True)
#     dt_sq = np.sqrt(dt)
#
#     S0 = 100  # spot price
#     X0 = np.log(S0)  # log price
#     v0 = 0.04  # spot variance
#
#     r = 0.05
#     mu = 0.1  # drift
#     rho = -0.2  # correlation coefficient
#     kappa = 2  # mean reversion coefficient
#     theta = 0.04  # long-term variance
#     xi = 0.3  # Vol of Vol - Volatility of instantaneous variance
#     std_asy = np.sqrt(theta * xi ** 2 / (2 * kappa))  # asymptotic standard deviation for the CIR process
#     assert 2 * kappa * theta > xi ** 2  # Feller condition
#
#     # Generate random Brownian Motion
#     MU = np.array([0, 0])
#     COV = np.matrix([[1, rho], [rho, 1]])
#     Z = ss.multivariate_normal.rvs(mean=MU, cov=COV, size=(N - 1, M))
#     Z_S = Z[:, :, 0]  # Stock Brownian motion:     W_1
#     Z_V = Z[:, :, 1]  # Variance Brownian motion:  W_2
#
#     # Initialize vectors
#     S_euler = np.zeros((N, M))
#     V_euler = np.zeros((N, M))
#     S_milstein = np.zeros((N, M))
#     V_milstein = np.zeros((N, M))
#
#     S_euler[0, :] = S0
#     V_euler[0, :] = v0
#     S_milstein[0, :] = S0
#     V_milstein[0, :] = v0
#
#     v = np.zeros(N)
#
#     S_euler_null = np.zeros((N, M))
#     V_euler_null = np.zeros((N, M))
#     S_milstein_null = np.zeros((N, M))
#     V_milstein_null = np.zeros((N, M))
#
#     S_euler_null[0, :] = S0
#     V_euler_null[0, :] = v0
#     S_milstein_null[0, :] = S0
#     V_milstein_null[0, :] = v0
#
#     results = []
#     # Step1: Simulation: Euler;Milstein;GBM
#     for t in tqdm(range(0, N - 1), desc="Simulating paths"):
#         # Euler:
#         Vt_e = np.maximum(V_euler[t, :], 0)
#         sqrt_V_e = np.sqrt(Vt_e)
#         V_euler[t + 1, :] = V_euler[t, :] + kappa * (theta - Vt_e) * dt + xi * sqrt_V_e * dt_sq * Z_V[t, :]
#         V_euler[t + 1, :] = np.abs(V_euler[t + 1, :])
#         S_euler[t + 1, :] = S_euler[t, :] + r * S_euler[t, :] * dt + sqrt_V_e * dt_sq * S_euler[t, :] * Z_S[t, :]
#
#         # Milstein:
#         Vt_m = np.maximum(V_milstein[t, :], 0)
#         sqrt_V_m = np.sqrt(Vt_m)
#         V_milstein[t + 1, :] = V_milstein[t, :] + kappa * (theta - Vt_m) * dt + xi * sqrt_V_m * dt_sq * Z_V[t,:] + 0.25 * xi ** 2 * dt * (Z_V[t, :] ** 2 - 1)
#         V_milstein[t + 1, :] = np.abs(V_milstein[t + 1, :])
#         S_milstein[t + 1, :] = S_milstein[t, :] + r * S_milstein[t, :] * dt + sqrt_V_m * dt_sq * S_milstein[t, :] * Z_S[t,:] + 0.5 * Vt_m * S_milstein[t,:] * dt * (Z_S[t, :] ** 2 - 1)
#         # GBM:
#         V_euler_null[t + 1, :] = V_euler_null[t, :] + kappa * (theta - Vt_e) * dt
#         V_euler_null[t + 1, :] = np.abs(V_euler_null[t + 1, :])
#         S_euler_null[t + 1, :] = S_euler_null[t, :] + r * S_euler_null[t, :] * dt + sqrt_V_e * dt_sq * S_euler_null[t,:] * Z_S[t, :]
#
#         V_milstein_null[t + 1, :] = V_milstein_null[t, :] + kappa * (theta - Vt_m) * dt
#         V_milstein_null[t + 1, :] = np.abs(V_milstein_null[t + 1, :])
#         S_milstein_null[t + 1, :] = S_milstein_null[t, :] + r * S_milstein_null[t,:] * dt + sqrt_V_m * dt_sq * S_milstein_null[t,:] * Z_S[t,:] + 0.5 * Vt_m * S_milstein_null[t,:] * dt * (Z_S[t, :] ** 2 - 1)
#     # Step2: Compute arithmetic Prices and Payoffs
#     avg_price_euler = np.mean(S_euler[1:, :], axis=0)
#     avg_price_milstein = np.mean(S_milstein[1:, :], axis=0)
#     payoff_euler = np.maximum(avg_price_euler - K, 0)
#     payoff_milstein = np.maximum(avg_price_milstein - K, 0)
#
#     # Step3: Compute rude MC estimator:
#     price_euler_rMC = np.exp(-r * T) * np.mean(payoff_euler)
#     std_euler_rMC = np.std(np.exp(-r * T) * payoff_euler) / np.sqrt(M)
#
#     price_milstein_rMC = np.exp(-r * T) * np.mean(payoff_milstein)
#     std_milstein_rMC = np.std(np.exp(-r * T) * payoff_milstein) / np.sqrt(M)
#
#     # Step4: Compute geometric Prices and Payoffs
#     sigma = np.sqrt(theta)
#     sigma_sq = sigma ** 2
#     sigma_tilde = sigma * np.sqrt((2 * N + 1) / (6 * (N + 1)))
#     r_tilde = 0.5 * (r - 0.5 * sigma_sq + 0.5 * sigma_tilde ** 2)
#
#     d1 = (np.log(S0 / K) + (r_tilde + 0.5 * sigma_tilde ** 2) * T) / (sigma_tilde * np.sqrt(T))
#     d2 = d1 - sigma_tilde * np.sqrt(T)
#     price_geo = S0 * np.exp((r_tilde - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#
#     log_S_gbm_euler = np.log(S_euler_null[1:, :])
#     geo_avg_gbm_euler = np.exp(np.mean(log_S_gbm_euler, axis=0))
#     payoff_gbm_geo_euler = np.maximum(geo_avg_gbm_euler - K, 0)
#
#     log_S_gbm_milstein = np.log(S_milstein_null[1:, :])
#     geo_avg_gbm_milstein = np.exp(np.mean(log_S_gbm_milstein, axis=0))
#     payoff_gbm_geo_milstein = np.maximum(geo_avg_gbm_milstein - K, 0)
#
#     # Step5: Compute control variate MC:
#     Y_bar_euler = np.mean(payoff_euler)
#     Y_bar_milstein = np.mean(payoff_milstein)
#     X_bar_euler = np.mean(payoff_gbm_geo_euler)
#     X_bar_milstein = np.mean(payoff_gbm_geo_milstein)
#     C_G = price_geo
#     c = 1
#
#     price_euler_cvMC = np.exp(-r * T) * (Y_bar_euler + c * (C_G - X_bar_euler))
#     std_euler_cvMC = np.std(np.exp(-r * T) * (payoff_euler + c * (C_G - payoff_gbm_geo_euler))) / np.sqrt(M)
#
#     price_milstein_cvMC = np.exp(-r * T) * (Y_bar_milstein + c * (C_G - X_bar_milstein))
#     std_milstein_cvMC = np.std(np.exp(-r * T) * (payoff_milstein + c * (C_G - payoff_gbm_geo_milstein))) / np.sqrt(M)
#
#     # Step6: Print results:
#     results = pd.DataFrame([
#         {"Method": "Rude MC estimator-Euler (Heston)", "Price": price_euler_rMC, "Std Error": std_euler_rMC},
#         {"Method": "Rude MC estimator-Milstein (Heston)", "Price": price_milstein_rMC, "Std Error": std_milstein_rMC},
#         {"Method": "Control Variate MC estimator-Euler", "Price": price_euler_cvMC, "Std Error": std_euler_cvMC},
#         {"Method": "Control Variate MC estimator-Milstein", "Price": price_milstein_cvMC,
#          "Std Error": std_milstein_cvMC}
#     ])
#
#     return results
## 2.4.1
# M_values = [1000, 2000, 5000, 10000]
# results_list_M = []
#
# for M in M_values:
#     result = MC(M, 10000, 0.3, -0.2, 100)
#     result["M"] = M
#     results_list_M.append(result)
#
# df_M = pd.concat(results_list_M)
# print(df_M)

##2.4.2-1
# xi_values = [0.1, 1]
# results_list_xi = []
#
# for xi in xi_values:
#     result = MC(10000, 10000, xi, -0.2, 100)
#     result["xi"] = xi
#     results_list_xi.append(result)
#
# df_xi = pd.concat(results_list_xi)
# print(df_xi)

##2.4.2-2
# rho_values = [-0.9, 0, 0.5]
# results_list_rho = []
#
# for rho in rho_values:
#     result = MC(10000, 10000, 0.3, rho, 100)
#     result["rho"] = rho
#     results_list_rho.append(result)
#
# df_rho = pd.concat(results_list_rho)
# print(df_rho)

##2.4.2-3
# K_values = [90, 100, 110]
# results_list_K = []
#
# for K in K_values:
#     result = MC(10000, 10000, 0.3, -0.2, K)
#     result["K"] = K
#     results_list_K.append(result)
#
# df_K = pd.concat(results_list_K)
# print(df_K)

##2.4.3
# N_values = [1000, 2000, 5000, 10000]
# results_list_N = []
#
# for N in N_values:
#     result = MC(10000, N, 0.3, -0.2, 100)
#     result["N"] = N
#     results_list_N.append(result)
#
# df_N = pd.concat(results_list_N)
# print(df_N)