import numpy as np
import pandas as pd

time = np.arange(0, 2000)

traffic = (
    30
    + 20*np.sin(time/40)        # daily wave
    + 10*np.sin(time/200)       # long-term trend
    + np.random.normal(0, 5, 2000)
)

traffic = np.maximum(traffic, 0)

df = pd.DataFrame({"vehicle_count": traffic})
df.to_csv("backend/ml/data/synthetic_dataset.csv", index=False)

print("Synthetic dataset created.")
