import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------------
# Load scaler
# ------------------------

scaler = joblib.load("scaler.save")

# ------------------------
# Load last sequence from dataset
# ------------------------

df = pd.read_csv("backend/ml/data/combined_dataset.csv")
data = df["vehicle_count"].values.reshape(-1, 1)
data_scaled = scaler.transform(data)

sequence_length = 24

last_sequence = data_scaled[-sequence_length:]
current_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)

# ------------------------
# Define model (same architecture)
# ------------------------

class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

model = LSTMRegressor()
model.load_state_dict(torch.load("lstm_model.pth"))
model.eval()

# ------------------------
# Recursive 24-hour forecast
# ------------------------

future_predictions = []

with torch.no_grad():
    for _ in range(24):
        prediction = model(current_sequence)
        future_predictions.append(prediction.item())

        prediction_reshaped = prediction.unsqueeze(0)
        current_sequence = torch.cat(
            (current_sequence[:, 1:, :], prediction_reshaped),
            dim=1
        )

# Convert back to real scale
future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

# ------------------------
# Plot
# ------------------------

plt.figure()
plt.plot(future_predictions)
plt.title("Predicted Bangalore Traffic (Next 24 Hours)")
plt.xlabel("Hour Ahead")
plt.ylabel("Vehicle Count")
plt.savefig("backend/ml/forecast/blr_24h_forecast.png")
print("Forecast graph saved as blr_24h_forecast.png")
