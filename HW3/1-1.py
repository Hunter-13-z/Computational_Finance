import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import solve_banded
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib import cm

#1-1
#MC method
# Parameters
S0 = 100         # Initial stock price
K = 100          # Strike price
r = 0.05        # Risk-free interest rate
sigma = 0.2      # Volatility
T = 1.0          # Time to maturity (in years)
N = 10000      # Number of simulations
B = 120 # Barrier Price

# Monte Carlo Simulation
np.random.seed(42)
Z = np.random.randn(N)
ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
payoffs = (ST >= K).astype(int)  # Indicator function
price_mc = np.exp(-r * T) * np.mean(payoffs)

# Closed-Form Solution
d_minus = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
price_closed = np.exp(-r * T) * norm.cdf(d_minus)

# Results
print(f"Monte Carlo Price: {price_mc:.6f}")
print(f"Closed-Form Price: {price_closed:.6f}")
print(f"Absolute Error: {abs(price_mc - price_closed):.6f}")

#1-1-2
#Numerical Pricing via Finite Difference Methods
def binary_option_implicit_scheme(S0, K, T, r, sigma, S_max=250, M=2000, N=2000):
    dS = S_max / N
    dt = T / M
    S = np.linspace(0, S_max, N + 1)
    t = np.linspace(0, T, M + 1)
    V = np.zeros((M + 1, N + 1))

    V[-1, S >= K] = 1.0

    j = np.arange(1, N)
    Sj = j * dS
    lambda_j = 0.5 * dt * sigma ** 2 * Sj ** 2 / dS ** 2

    lower = -lambda_j[1:]
    main = 1 + 2 * lambda_j
    upper = -lambda_j[:-1]
    A = sp.diags([lower, main, upper], offsets=[-1, 0, 1], shape=(N - 1, N - 1), format="csc")

    for n in reversed(range(M)):
        # V[n, 0] = 0.0  # At S = 0, binary option worthless
        # V[n, -1] = 1.0  # At S = S_max, option surely pays 1
        V[n, 1:N] = spla.spsolve(A, V[n + 1, 1:N])

    price = np.interp(S0, S, V[0, :])
    return price, S, t, V


price_implicit_scheme, S_imp, t_imp, V_imp = binary_option_implicit_scheme(S0, K, T, r, sigma)

# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# T_grid, S_grid = np.meshgrid(t_imp, S_imp)
# ax.plot_surface(S_grid, T_grid, V_imp.T, cmap=cm.coolwarm)
# ax.set_title("Binary Option Surface - Implicit Discretization Scheme")
# ax.set_xlabel("Asset Price S")
# ax.set_ylabel("Time t")
# ax.set_zlabel("Option Value", labelpad=0)
# ax.view_init(elev=30, azim=225)
# plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
# plt.show()

# print(price_implicit_scheme)

def binary_option_crank_nicolson_scheme(S0, K, T, r, sigma, S_max=250, M=2000, N=2000):
    dS = S_max / N
    dt = T / M
    S = np.linspace(0, S_max, N + 1)
    t = np.linspace(0, T, M + 1)
    V = np.zeros((M + 1, N + 1))

    V[-1, S >= K] = 1.0

    j = np.arange(1, N)
    Sj = j * dS

    a = 0.25 * dt * (sigma**2 * Sj**2 / dS**2 - r * Sj / dS)
    b = -0.5 * dt * (sigma**2 * Sj**2 / dS**2 + r)
    c = 0.25 * dt * (sigma**2 * Sj**2 / dS**2 + r * Sj / dS)

    A_lower = -a[1:]
    A_main = 1 - b
    A_upper = -c[:-1]
    A = sp.diags([A_lower, A_main, A_upper], offsets=[-1, 0, 1], format="csc")

    B_lower = a[1:]
    B_main = 1 + b
    B_upper = c[:-1]
    B = sp.diags([B_lower, B_main, B_upper], offsets=[-1, 0, 1], format="csc")

    for n in reversed(range(M)):
        RHS = B @ V[n + 1, 1:N]
        V[n, 1:N] = spla.spsolve(A, RHS)

    price = np.interp(S0, S, V[0, :])
    return price, S, t, V

price_cn_scheme, S_cn, t_cn, V_cn = binary_option_crank_nicolson_scheme(S0, K, T, r, sigma)

# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# T_grid, S_grid = np.meshgrid(t_cn, S_cn)
# ax.plot_surface(S_grid, T_grid, V_cn.T, cmap=cm.viridis)
# ax.set_title("Binary Option Surface - Crank-Nicolson Scheme")
# ax.set_xlabel("Asset Price S")
# ax.set_ylabel("Time t")
# ax.set_zlabel("Option Value", labelpad=0)
# ax.set_zlim(0, 1)
# ax.view_init(elev=30, azim=225)
# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# plt.show()

# print(price_cn_scheme)


# def plot_sensitivity_curves_fast(method="implicit"):
#     S_range = np.linspace(0, 250, 300)  # 资产价格范围
#     base_K = 100
#     base_mu = 0.05
#     base_sigma = 0.2
#     base_T = 1.0
#
#     # 参数变化范围
#     mu_range = [0.01, 0.05, 0.1]
#     sigma_range = [0.1, 0.2, 0.3]
#     K_range = [95, 105, 115]
#     T_range = [0.5, 1.0, 2.0]
#
#     param_sets = [
#         ('mu', mu_range),
#         ('sigma', sigma_range),
#         ('K', K_range),
#         ('T', T_range),
#     ]
#
#     fig, axs = plt.subplots(2, 2, figsize=(14, 10))
#     titles = ['Risk-Free Rate ($\mu$)', 'Volatility ($\sigma$)', 'Strike Price (K)', 'Time to Maturity (T)']
#
#     for idx, (param, values) in enumerate(param_sets):
#         ax = axs[idx // 2, idx % 2]
#         for val in values:
#             # 根据变化的参数，设置不同的值，其余保持不变
#             if param == 'mu':
#                 r_val = val
#                 sigma_val, K_val, T_val = base_sigma, base_K, base_T
#             elif param == 'sigma':
#                 sigma_val = val
#                 r_val, K_val, T_val = base_mu, base_K, base_T
#             elif param == 'K':
#                 K_val = val
#                 r_val, sigma_val, T_val = base_mu, base_sigma, base_T
#             else:  # T
#                 T_val = val
#                 r_val, sigma_val, K_val = base_mu, base_sigma, base_K
#
#             # 只计算一次 PDE 解，之后通过插值计算曲线
#             if method == "implicit":
#                 _, S_grid, _, V_grid = binary_option_implicit_scheme(0, K_val, T_val, r_val, sigma_val)
#             else:
#                 _, S_grid, _, V_grid = binary_option_crank_nicolson_scheme(0, K_val, T_val, r_val, sigma_val)
#
#             # 直接提取 t=0 时刻的价格曲线
#             V_curve = np.interp(S_range, S_grid, V_grid[0, :])
#
#             ax.plot(S_range, V_curve, label=f"{param}={val}")
#
#         ax.set_title(f"{method.capitalize()} - Sensitivity to {titles[idx]}")
#         ax.set_xlabel("Asset Price S")
#         ax.set_ylabel("Option Value")
#         ax.grid(True)
#         ax.legend()
#
#     plt.suptitle(f"Sensitivity Analysis (Optimized): {method.capitalize()} Scheme", fontsize=16)
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.show()
#
# # 调用优化后的函数
# plot_sensitivity_curves_fast(method="implicit")
# plot_sensitivity_curves_fast(method="cn")

#1-2-1
#Anlytical Price
def barrier_option_analytical(S, K, B, r, sigma, T):
    def Indicator(S,B):
        if S >= B:
            return 0
        else:
            return 1

    tau = T
    sqrt_tau = np.sqrt(tau)

    def delta_pm(z, sign):
        return (np.log(z) + (r + sign * 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)

    # First big term
    term1 = S * Indicator(S,B) * (norm.cdf(delta_pm(S / K, +1)) - norm.cdf(delta_pm(S / B, +1)))

    # Second big term
    pow_factor1 = (B / S)**(1 + 2 * r / sigma**2)
    term2 = -pow_factor1 * S * (norm.cdf(delta_pm(B**2 / (K * S), 1)) - norm.cdf(delta_pm(B / S, 1)))

    # Third big term
    term3 = -np.exp(-r * tau) * K * Indicator(S,B) * (norm.cdf(delta_pm(S / K, -1)) - norm.cdf(delta_pm(S / B, -1)))

    # Fourth big term
    pow_factor2 = (S / B)**(1 - 2 * r / sigma**2)
    term4 = np.exp(-r * tau) * K * pow_factor2 * (norm.cdf(delta_pm(B**2 / (K * S), -1)) - norm.cdf(delta_pm(B / S, -1)))

    price = term1 + term2 + term3 + term4
    return price

#Normal Monte Carlo
def monte_carlo_barrier_call(S0, K, B, r, sigma, T, N_paths, M_steps):
    dt = T / M_steps
    Z = np.random.randn(N_paths, M_steps)
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_S_paths = np.log(S0) + np.cumsum(increments, axis=1)
    S_paths = np.exp(log_S_paths)

    # 敲出判断：看是否最大值超过 B
    knocked_out = (S_paths >= B).any(axis=1)
    S_T = S_paths[:, -1]

    payoffs = np.where(knocked_out, 0.0, np.exp(-r * T) * np.maximum(S_T - K, 0))
    return np.mean(payoffs)

#Adjusted Monte Carlo
def adjusted_monte_carlo_barrier_call(S0, K, B, r, sigma, T, N_paths, M_steps, beta_1=0.5826):
    H_adj = B * np.exp(-beta_1 * sigma * np.sqrt(T / M_steps))
    return monte_carlo_barrier_call(S0, K, H_adj, r, sigma, T, N_paths, M_steps)

#Compare Price
N_paths = 100000
M_steps = 50
price_analytical = barrier_option_analytical(S0, K, B, r, sigma, T)
price_mc = monte_carlo_barrier_call(S0, K, B, r, sigma, T, N_paths, M_steps)
price_adj_mc = adjusted_monte_carlo_barrier_call(S0, K, B, r, sigma, T, N_paths, M_steps)

# 输出结果
print(f"Analytical Price: {price_analytical:.6f}")
print(f"Standard MC Price: {price_mc:.6f}")
print(f"Adjusted MC Price: {price_adj_mc:.6f}")

# true_price = barrier_option_analytical(S0, K, B, r, sigma, T)
# M_steps = 50
# N_list = [1000, 5000, 10000, 50000, 100000]
# errors_mc = []
# errors_adj_mc = []

# for N_paths in N_list:
#     price_mc = monte_carlo_barrier_call(S0, K, B, r, sigma, T, N_paths, M_steps)
#     price_adj_mc = adjusted_monte_carlo_barrier_call(S0, K, B, r, sigma, T, N_paths, M_steps)
#     errors_mc.append(abs(price_mc - true_price))
#     errors_adj_mc.append(abs(price_adj_mc - true_price))
#
# plt.loglog(N_list, errors_mc, 'o-', label='Standard MC Error')
# plt.loglog(N_list, errors_adj_mc, 's-', label='Adjusted MC Error')
# plt.xlabel('Number of Simulation Paths')
# plt.ylabel('Absolute Error')
# plt.title('Convergence of MC and Adjusted MC')
# plt.grid(True, which="both")
# plt.legend()
# plt.show()

#1-2-2
def barrier_option_implicit_scheme(S0, K, B, T, r, sigma, S_max=250, M=2000, N=2000):
    dS = S_max / N
    dt = T / M
    S = np.linspace(0, S_max, N + 1)
    t = np.linspace(0, T, M + 1)
    V = np.zeros((M + 1, N + 1))

    # Terminal condition: at maturity
    V[-1, :] = np.maximum(S - K, 0)
    V[-1, S >= B] = 0.0

    # Pre-compute coefficients
    j = np.arange(1, N)
    Sj = j * dS
    lambda_j = 0.5 * dt * sigma ** 2 * Sj ** 2 / dS ** 2

    lower = -lambda_j[1:]
    main = 1 + 2 * lambda_j + r * dt
    upper = -lambda_j[:-1]

    A = sp.diags([lower, main, upper], offsets=[-1, 0, 1], shape=(N - 1, N - 1), format="csc")

    # Backward time-stepping
    for n in reversed(range(M)):
        # Apply boundary conditions
        if S[-1] >= B:
            V[n, -1] = 0.0
        else:
            V[n, -1] = S[-1] - K  # At S = S_max, barrier exceeded, worthless

        # Solve for interior points
        V_new = spla.spsolve(A, V[n + 1, 1:N])
        V[n, 1:N] = V_new

        # Apply barrier condition after solving
        V[n, S >= B] = 0.0

    # Interpolate final price
    price = np.interp(S0, S, V[0, :])
    return price, S, t, V

price_bar_im_scheme, S_bar_im, t_bar_im, V_bar_im = barrier_option_implicit_scheme(S0, K, B, T, r, sigma, S_max=250, M=2000, N=2000)
print(price_bar_im_scheme)

# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# T_grid, S_grid = np.meshgrid(t_bar_im, S_bar_im)
# ax.plot_surface(S_grid, T_grid, V_bar_im.T, cmap=cm.coolwarm)
# ax.set_title("Barrier Option Surface - Implicit Discretization Scheme")
# ax.set_xlabel("Asset Price S")
# ax.set_ylabel("Time t")
# ax.set_zlabel("Option Value", labelpad=0)
# ax.view_init(elev=30, azim=225)
# plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
# plt.show()

def plot_barrier_sensitivity(method="implicit"):
    S_range = np.linspace(0, 250, 300)  # Asset price range
    base_K = 100
    base_r = 0.05
    base_sigma = 0.2
    base_T = 1.0
    B = 120  # Barrier level

    # Parameter ranges
    r_range = [0.01, 0.05, 0.1]
    sigma_range = [0.1, 0.2, 0.3]
    K_range = [95, 105, 115]
    T_range = [0.5, 1.0, 2.0]

    param_sets = [
        ('r', r_range),
        ('sigma', sigma_range),
        ('K', K_range),
        ('T', T_range),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    titles = ['Risk-Free Rate ($r$)', 'Volatility ($\\sigma$)', 'Strike Price ($K$)', 'Time to Maturity ($T$)']

    for idx, (param, values) in enumerate(param_sets):
        ax = axs[idx // 2, idx % 2]
        for val in values:
            # Fix parameters based on current analysis
            if param == 'r':
                r_val = val
                sigma_val, K_val, T_val = base_sigma, base_K, base_T
            elif param == 'sigma':
                sigma_val = val
                r_val, K_val, T_val = base_r, base_K, base_T
            elif param == 'K':
                K_val = val
                r_val, sigma_val, T_val = base_r, base_sigma, base_T
            else:
                T_val = val
                r_val, sigma_val, K_val = base_r, base_sigma, base_K

            # Compute once and interpolate for all S_range
            _, S_grid, _, V_grid = barrier_option_implicit_scheme(0, K_val, B, T_val, r_val, sigma_val, M=500, N=500)
            V_curve = np.interp(S_range, S_grid, V_grid[0, :])

            ax.plot(S_range, V_curve, label=f"{param}={val}")

        ax.set_title(f"Sensitivity to {titles[idx]}")
        ax.set_xlabel("Asset Price $S$")
        ax.set_ylabel("Option Value")
        ax.grid(True)
        ax.legend()

    plt.suptitle("Barrier Option Sensitivity Analysis (Optimized)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Run the sensitivity analysis plot
plot_barrier_sensitivity()