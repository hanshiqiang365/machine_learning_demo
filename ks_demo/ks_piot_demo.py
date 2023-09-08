#author:hanshiqiang365

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Generate random data and a normal distribution for demonstration purposes
np.random.seed(0)
data = np.random.normal(0, 1, 1000)
theoretical = np.sort(np.random.normal(0, 1, 1000))

# Compute KS statistic
ks_statistic, p_value = stats.ks_2samp(data, theoretical)

# Plot KS Plot
plt.figure(figsize=(10, 6))
plt.plot(np.sort(data), np.linspace(0, 1, len(data), endpoint=False), label='Data CDF')
plt.plot(theoretical, np.linspace(0, 1, len(theoretical), endpoint=False), label='Theoretical CDF', linestyle='--')
plt.title(f"KS Plot (KS Statistic = {ks_statistic:.2f})")
plt.legend()
plt.xlabel("Value")
plt.ylabel("CDF")
plt.grid(True)
plt.show()

