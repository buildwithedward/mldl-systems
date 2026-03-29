import matplotlib.pyplot as plt
import numpy as np

# Generate random data
data = np.random.Generator(loc=100, scale=15, size=1000)

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot histogram
ax.hist(data, bins=30, color="#58a6ff", alpha=0.8, edgecolor="black")

# Labels and title
ax.set_xlabel("Score")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Test Scores")
ax.grid(axis="y", alpha=0.3)  # Add gridlines

# Save
fig.tight_layout()  # Auto-adjust spacing
fig.savefig("histogram.png", dpi=150, bbox_inches="tight")
plt.close(fig)