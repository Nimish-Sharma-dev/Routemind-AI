import pandas as pd
import numpy as np

# Load raw metro dataset
df = pd.read_csv("backend/ml/data/public_dataset_raw.csv")

df["date_time"] = pd.to_datetime(df["date_time"])
df["hour"] = df["date_time"].dt.hour
df["dayofweek"] = df["date_time"].dt.dayofweek

traffic = df["traffic_volume"].astype(float)

# -----------------------
# Bangalore Simulation Logic
# -----------------------

# Increase baseline (BLR heavier traffic)
traffic = traffic * 1.5

# Morning peak boost (8–11 AM)
traffic += np.where(df["hour"].between(8, 11), 250, 0)

# Evening peak heavy boost (4–9 PM)
traffic += np.where(df["hour"].between(16, 21), 500, 0)

# Weekend mall spike
traffic += np.where(df["dayofweek"] >= 5, 200, 0)

# Random congestion spikes
np.random.seed(42)
spikes = np.random.choice(len(traffic), size=300, replace=False)
traffic.iloc[spikes] += np.random.randint(300, 900, size=300)

traffic = np.maximum(traffic, 0)

blr_df = pd.DataFrame({
    "vehicle_count": traffic
})

blr_df.to_csv("backend/ml/data/public_dataset.csv", index=False)

print("Bangalore-style dataset created.")
