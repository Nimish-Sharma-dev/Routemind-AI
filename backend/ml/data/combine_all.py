import pandas as pd

synthetic = pd.read_csv("backend/ml/data/synthetic_dataset.csv")
public = pd.read_csv("backend/ml/data/public_dataset.csv")

# Optional: if you have YOLO dataset
# yolo = pd.read_csv("backend/ml/data/yolo_dataset.csv")
# combined = pd.concat([synthetic, public, yolo], ignore_index=True)

combined = pd.concat([synthetic, public], ignore_index=True)

combined.to_csv("backend/ml/data/combined_dataset.csv", index=False)

print("Combined dataset created.")
print("Total rows:", len(combined))
