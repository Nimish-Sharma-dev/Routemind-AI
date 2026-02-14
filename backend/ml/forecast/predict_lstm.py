
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------------
# Model Definition (MUST match training)
# ------------------------

class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out


# ------------------------
# Load Model + Scaler
# ------------------------

model = LSTMRegressor()
model.load_state_dict(torch.load("lstm_model.pth"))
model.eval()

scaler = joblib.load("scaler.save")

# ------------------------
# Load Dataset
# ------------------------

df = pd.read_csv("backend/ml/density/traffic_data.csv")

data = df["vehicle_count"].values.reshape(-1, 1)

# Normalize using SAME scaler
data_scaled = scaler.transform(data)

# ------------------------
# Create Sequences
# ------------------------

sequence_length = 5

X = []
y_actual = []

for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i+sequence_length])
    y_actual.append(data_scaled[i+sequence_length])

X = np.array(X)
y_actual = np.array(y_actual)

X_tensor = torch.tensor(X, dtype=torch.float32)

# ------------------------
# Make Predictions
# ------------------------

with torch.no_grad():
    predictions_scaled = model(X_tensor).numpy()

# ------------------------
# Inverse Transform
# ------------------------

predictions = scaler.inverse_transform(predictions_scaled)
y_actual_real = scaler.inverse_transform(y_actual)

# ------------------------
# Plot Results
# ------------------------

plt.figure(figsize=(12, 6))
plt.plot(y_actual_real, marker='o')
plt.plot(predictions, marker='x')
plt.legend()
plt.title("LSTM Traffic Forecast")
plt.xlabel("Time Step")
plt.ylabel("Vehicle Count")
plt.tight_layout()
plt.savefig("backend/ml/forecast/forecast_plot.png")
print("Plot saved as forecast_plot.png")

