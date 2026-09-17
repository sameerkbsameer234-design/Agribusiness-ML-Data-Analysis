import pandas as pd
import joblib
from pathlib import Path


data_path = "../data/cleaned/crop_yield_cleaned.csv"
df = pd.read_csv(data_path)


target = "Yield"

features = [
    "Crop",
    "Crop_Year",
    "Season",
    "State",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

model_path = Path("../models/final_crop_yield_model.pkl")

loaded_model = joblib.load(model_path)


sample = df.sample(1, random_state=42)

sample_features = sample[features]

actual_yield = sample[target].iloc[0]


predicted_yield = loaded_model.predict(sample_features)[0]


print("\n========== CROP YIELD PREDICTION DEMONSTRATION ==========")

print("\nInput Information:")

for feature in features:
    print(f"{feature}: {sample[feature].iloc[0]}")


print("\nActual Yield:", actual_yield)

print("Predicted Yield:", predicted_yield)


absolute_error = abs(actual_yield - predicted_yield)

print("Absolute Predicted Error:", absolute_error)


prediction_result = sample[features].copy()

prediction_result["Actual_Yield"] = actual_yield
prediction_result["Predicted_Yield"] = predicted_yield
prediction_result["Absolute_Error"] = absolute_error

output_path = "../data/cleaned/sample_prediction.csv"

prediction_result.to_csv(output_path, index=False)


print("\nPrediction result saved to:")
print(output_path)

print("\n==================================================")
