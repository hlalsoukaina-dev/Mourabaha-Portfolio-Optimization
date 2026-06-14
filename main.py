import pandas as pd
import numpy as np
from scipy.optimize import minimize

# --- 1. Data Setup ---
df = pd.read_excel('data.xlsx')
sectors = ['Immobilière', 'Automobile', 'Equipement', 'Matieres Premieres']

# --- 2. Optimization Logic ---
def negative_sharpe(weights, cov_matrix, mean_returns):
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    # Annualizing return and volatility
    return -(portfolio_return * 12) / (portfolio_vol * np.sqrt(12))

# --- 3. Run Optimization ---
returns_data = df[sectors].pct_change().fillna(0)
cov_matrix = returns_data.cov()
mean_returns = returns_data.mean()

# Define bounds: Real Estate (40-60%), Others (0-40%)
bounds = ((0.4, 0.6), (0, 0.4), (0, 0.4), (0, 0.4))
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Optimization process
result = minimize(negative_sharpe, [0.4, 0.2, 0.2, 0.2], method='SLSQP', 
                  bounds=bounds, constraints=constraints, 
                  args=(cov_matrix, mean_returns))

# --- 4. Display Results ---
print("--- Optimized Portfolio Allocation ---")
for sector, weight in zip(sectors, result.x):
    print(f"{sector}: {weight:.2%}")
