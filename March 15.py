# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 17:59:01 2025

@author: malla
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate simple synthetic data
np.random.seed(42)
data = np.array([18, 19, 20, 21, 22])  # Small dataset

# Define prior hyperparameters
prior_mu = 19
prior_sigma = 2

# Compute posterior parameters
posterior_mu = (np.mean(data) + prior_mu) / 2
posterior_sigma = np.sqrt(np.var(data) / len(data))

# Sample from posterior
posterior_samples = np.random.normal(posterior_mu, posterior_sigma, size=1000)

# Plot posterior distribution
plt.hist(posterior_samples, bins=20, density=True, color='pink', edgecolor='black')
plt.xlabel("μ")
plt.ylabel("Density")
plt.title("Posterior distribution of μ")
plt.show()

# Summary statistics
print(f"Posterior mean of μ: {np.mean(posterior_samples):.4f}")
print(f"Posterior standard deviation of μ: {np.std(posterior_samples):.4f}")
