import matplotlib.pyplot as plt
import numpy as np

# Sample data generation
np.random.seed(0)
time = np.linspace(240, 300, 600)
ratio_delta = np.random.normal(0, 0.05, size=time.size)

# Plot the entire data
plt.figure(figsize=(14, 5))
plt.plot(time, ratio_delta)
plt.title('20 min Ratio_delta graph')
plt.xlabel('Time (seconds)')
plt.ylabel('Ratio_delta')
plt.show()

# Plot a zoomed section of the data (e.g., between 240 and 250 seconds)
plt.figure(figsize=(14, 5))
plt.plot(time, ratio_delta)
plt.title('Zoomed 240-250 seconds Ratio_delta graph')
plt.xlabel('Time (seconds)')
plt.ylabel('Ratio_delta')
plt.xlim(240, 250)
plt.show()
