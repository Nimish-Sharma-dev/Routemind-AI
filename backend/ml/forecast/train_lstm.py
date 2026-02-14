import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# ------------------------
# Load Dataset
# ------------------------

df = pd.read_csv("backend/ml/density/traffic_data.csv")

data = df["vehicle_count"].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Save scaler for inference
joblib.dump(scaler, "scaler.save")

# ------------------------
# Create Sequences
# ------------------------

sequence_length = 5

X = []
y = []

for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length])
    y.append(data[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# ------------------------
# Time-based Split (NO random split)
# ------------------------

train_size = int(len(X_tensor) * 0.8)

X_train = X_tensor[:train_size]
X_test = X_tensor[train_size:]

y_train = y_tensor[:train_size]
y_test = y_tensor[train_size:]

# ------------------------
# Define LSTM Model
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

model = LSTMRegressor()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ------------------------
# Training Loop
# ------------------------

EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

# ------------------------
# Evaluation
# ------------------------

model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
    print("Test MSE:", test_loss.item())

# ------------------------
# Save Model
# ------------------------

torch.save(model.state_dict(), "lstm_model.pth")
print("Model saved successfully.")
