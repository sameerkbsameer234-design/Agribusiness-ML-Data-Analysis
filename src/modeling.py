import pandas as pd
from sklearn.model_selection import train_test_split

data_path = "../data/cleaned/crop_yield_cleaned.csv"

df = pd.read_csv(data_path)

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())


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

x = df[features]
y = df[target]


print("\n" + "=" * 60)
print("FEATURE AND TARGET INFORMATION")
print("=" * 60)

print("Target Variable:", target)
print("Number of Features:", len(features))

print("\nSelected Features:")
for feature in features:
    print("-", feature)

print("\nFeature Dataset Shape:", x.shape)
print("Target Dataset Shape:", y.shape)


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42
)


print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print("Training Samples:", x_train.shape[0])
print("Testing Samples:", x_test.shape[0])

print("Training Features Shape:", x_train.shape)
print("Testing Features Shape:", x_test.shape)

print("\nStep 2 completed successfully.")


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


categorical_features = [
    "Crop",
    "Season",
    "State"
]

numerical_features = [
    "Crop_Year",
    "Annual_Rainfall",
    "Area",
    "Fertilizer",
    "Pesticide"
]

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_transformer,
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

print("\n" + "=" * 60)
print("DATA PREPROCESSING CONFIGURED")
print("=" * 60)

print("Categorical Features:")
for feature in categorical_features:
    print("-", feature)

print("\nNumerical Features:")
for feature in numerical_features:
    print("-", feature)

print("\nEncoding Method: OneHotEncoding")
print("Unknown Categories: Ignored")

print("\nPreprocessing pipeline configured successfully.")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

linear_regression_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

print("\n" + "=" * 60)
print("TRAINING LINEAR REGRESSION MODEL")
print("=" * 60)

linear_regression_model.fit(x_train, y_train)

print("Linear Regression model trained successfully.")


y_pred_linear = linear_regression_model.predict(x_test)

print("\nPrediction generated successfully.")


mae_linear = mean_absolute_error(y_test, y_pred_linear)

mse_linear = mean_squared_error(y_test, y_pred_linear)

rmse_linear = np.sqrt(mse_linear)

r2_linear = r2_score(
    y_test,
    y_pred_linear
)


print("\n" + "=" * 60)
print("LINEAR REGRESSION PERFORMANCE")
print("=" * 60)

print(f"MAE: {mae_linear:.4f}")
print(f"MSE: {mse_linear:.4f}")
print(f"RMSE: {rmse_linear:.4f}")
print(f"R2 Score: {r2_linear:.4f}")


print("\n" + "=" * 60)
print("BASELINE MODEL INTERPRETATION")
print("=" * 60)

print(
    "The model performance will be compared with "
    "tree-based models in the next stages."
)

print("\nStep 4 completed successfully.")

from sklearn.ensemble import RandomForestRegressor

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST REGRESSOR")
print("=" * 60)

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt"
        ))
    ]
)

random_forest_model.fit(x_train, y_train)

print("Random Forest model trained successfully.")


y_pred_rf = random_forest_model.predict(x_test)

print("Random Forest Prediction generated successfully.")

mae_rf = mean_absolute_error(y_test, y_pred_rf)

mse_rf = mean_squared_error(y_test, y_pred_rf)

rmse_rf = np.sqrt(mse_rf)

r2_rf = r2_score(y_test, y_pred_rf)

print("\n" + "=" * 60)
print("RANDOM FOREST PERFORMANCE")
print("=" * 60)

print(f"MAE: {mae_rf:.4f}")
print(f"MSE: {mse_rf:.4f}")
print(f"RMSE: {rmse_rf:.4f}")
print(f"R2 Score: {r2_rf:.4f}")



print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)


comparison = pd.DataFrame({
    "Model" : [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE" : [
        mae_linear,
        mae_rf
    ],
    "MSE" : [
        mse_linear,
        mse_rf
    ],
    "RMSE" : [
        rmse_linear,
        rmse_rf
    ],
    "R2 Score" : [
        r2_linear,
        r2_rf
    ]
})

print(comparison.to_string(index=False))

print("\nStep 5 completed successfully.")

from sklearn.ensemble import GradientBoostingRegressor

print("\n" + "=" * 60)
print("TRAINING GRADIENT BOOSTING REGRESSOR")
print("=" * 60)


gradient_boosting_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(
            n_estimators=200,
            random_state=42,
            learning_rate=0.05,
            max_depth=3
        ))
    ]
)

gradient_boosting_model.fit(x_train, y_train)

print("Gradient Boosting model trained successfully.")


y_pred_gb = gradient_boosting_model.predict(x_test)

print("Gradient Boosting Prediction generated successfully.")


mae_gb = mean_absolute_error(y_test, y_pred_gb)

mse_gb = mean_squared_error(y_test, y_pred_gb)

rmse_gb = np.sqrt(mse_gb)

r2_gb = r2_score(y_test, y_pred_gb)


print("\n" + "=" * 60)
print("GRADIENT BOOSTING PERFORMANCE")
print("=" * 60)

print(f"MAE: {mae_gb:.4f}")
print(f"MSE: {mse_gb:.4f}")
print(f"RMSE: {rmse_gb:.4f}")
print(f"R2 Score: {r2_gb:.4f}")

print("\n" + "=" * 60)
print("THREE-MODEL PERFORMANCE COMPARISON")
print("=" * 60)

model_comparison = pd.DataFrame({
    "Model" : [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "MAE" : [
        mae_linear,
        mae_rf,
        mae_gb
    ],
    "MSE" : [
        mse_linear,
        mse_rf,
        mse_gb
    ],
    "RMSE" : [
        rmse_linear,
        rmse_rf,
        rmse_gb
    ],
    "R2" : [
        r2_linear,
        r2_rf,
        r2_gb
    ]
})

print(model_comparison.to_string(index=False))


best_model_name = model_comparison.loc[
    model_comparison["R2"].idxmax(), "Model"
]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Best Model Based on Highest R2:", best_model_name)
print("\nStep 6 completed successfully.")

import matplotlib.pyplot as plt

print("\n" + "=" * 60)
print("BEST MODEL - DETAILED EVALUATION")
print("=" * 60)

best_model = gradient_boosting_model


final_predictions = best_model.predict(x_test)

residuals = y_test - final_predictions

absolute_errors = np.abs(residuals)


final_mae = mean_absolute_error(y_test, final_predictions)

final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)

final_r2 = r2_score(y_test, final_predictions)


print("\nFinal Model : Gradient Boosting Regressor")


print(f"MAE : {final_mae:.4f}")
print(f"MSE : {final_mse:.4f}")
print(f"RMSE : {final_rmse:.4f}")
print(f"R2 Score: {final_r2:.4f}")


final_metrics = pd.DataFrame({
    "Metric" : [
        "MAE",
        "MSE",
        "RMSE",
        "R2"
    ],
    "Value" : [
        final_mae,
        final_mse,
        final_rmse,
        final_r2
    ]
})

final_metrics.to_csv(
    "../data/cleaned/final_model_metrics.csv", index=False
)

print("\nFinal model metrics saved.")


prediction_results = pd.DataFrame({
    "Actual_Yield" : y_test.values,
    "Predicted_Yield" : final_predictions,
    "Residual" : residuals,
    "Absolute_Error" : absolute_errors
})

prediction_results.to_csv(
    "../data/cleaned/prediction_results.csv", index=False
)

print("Prediction results saved.")


plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    final_predictions,
    alpha=0.5
)

min_value = min(
    y_test.min(),
    final_predictions.min()
)
max_value = max(
    y_test.max(),
    final_predictions.max()
)
plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Gradient Boosting - Actual vs Predicted Yield")

plt.tight_layout()

plt.savefig(
    "../visualizations/week_3/actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

plt.figure(figsize=(8, 6))

plt.hist(
    residuals,
    bins=40
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Gradient Boosting - Residuals Distribution")

plt.tight_layout()

plt.savefig(
    "../visualizations/week_3/residuals_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



plt.figure(figsize=(8, 6))

plt.scatter(
    final_predictions,
    residuals,
    alpha=0.5
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted Yield")
plt.ylabel("Residuals")
plt.title("Gradient Boosting - Residuals vs Predicted Yield")

plt.tight_layout()

plt.savefig(
    "../visualizations/week_3/residuals_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\n" + "=" * 60)
print("Step 7 completed successfully.")
print("=" * 60)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

fitted_preprocessor = best_model.named_steps["preprocessor"]

fitted_model = best_model.named_steps["model"]

feature_names = fitted_preprocessor.get_feature_names_out()

importance_values = fitted_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature" : feature_names,
    "Importance" : importance_values
})

feature_importance = feature_importance.sort_values(
    by="Importance", ascending=False
).reset_index(drop=True)

print("\nTop 20 Important Features:")

print(
    feature_importance.head(20).to_string(index=False)
)


feature_importance.to_csv(
    "../data/cleaned/feature_importance.csv", index=False
)

print("\nFeature importance saved successfully.")


top_features = feature_importance.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.title("Gradient Boosting - Top 15 Feature Importances")

plt.tight_layout()

plt.savefig(
    "../visualizations/week_3/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n" + "=" * 60)
print("Step 8 completed successfully.")
print("=" * 60)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE VISUALIZATION")
print("=" * 60)


plt.figure(figsize=(9, 6))

plt.bar(
    model_comparison["Model"],
    model_comparison["R2"]
)

plt.xlabel("Machine Learning Model")
plt.ylabel("R² Score")
plt.title("Model Comparison - R² Score")

plt.ylim(0, 1.05)

for i, value in enumerate(model_comparison["R2"]):
    plt.text(
        i,
        value + 0.02,
        f"{value:.4f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "../visualizations/week_3/model_r2_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


plt.figure(figsize=(9, 6))

plt.bar(
    model_comparison["Model"],
    model_comparison["RMSE"]
)

plt.xlabel("Machine Learning Model")
plt.ylabel("RMSE")
plt.title("Model Comparison - RMSE")

for i, value in enumerate(model_comparison["RMSE"]):
    plt.text(
        i,
        value,
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    "../visualizations/week_3/model_rmse_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


plt.figure(figsize=(9, 6))

plt.bar(
    model_comparison["Model"],
    model_comparison["MAE"]
)

plt.xlabel("Machine Learning Model")
plt.ylabel("MAE")
plt.title("Model Comparison - MAE")

for i, value in enumerate(model_comparison["MAE"]):
    plt.text(
        i,
        value,
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    "../visualizations/week_3/model_mae_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nModel comparison charts saved successfully.")

print("\n" + "=" * 60)
print("STEP 9 COMPLETED SUCCESSFULLY.")
print("=" * 60)

from sklearn.model_selection import RandomizedSearchCV

print("\n" + "=" * 60)
print("HYPERPARAMETER OPTIMIZATION")
print("=" * 60)


tuning_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(
            random_state=42
        ))
    ]
)

param_distributions = {
    "model__n_estimators": [100, 150, 200],
    "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
    "model__max_depth": [2, 3, 4],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

random_search = RandomizedSearchCV(
    estimator=tuning_pipeline,
    param_distributions=param_distributions,
    n_iter=6,
    scoring="r2",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nStarting hyperparameter search...")
print("Number of parameter combinations tested: 6")
print("Cross-validation folds: 3")

random_search.fit(x_train, y_train)


print("\n" + "=" * 60)
print("BEST HYPERPARAMETERS")
print("=" * 60)

print(random_search.best_params_)

print(f"\nBest Cross-Validation R²: "
      f"{random_search.best_score_:.4f}")

tuned_model = random_search.best_estimator_

y_pred_tuned = tuned_model.predict(x_test)

tuned_mae = mean_absolute_error(y_test, y_pred_tuned)

tuned_mse = mean_squared_error(y_test, y_pred_tuned)

tuned_rmse = np.sqrt(tuned_mse)

tuned_r2 = r2_score(y_test, y_pred_tuned)



print("\n" + "=" * 60)
print("TUNED MODEL TEST PERFORMANCE")
print("=" * 60)

print(f"MAE : {tuned_mae:.4f}")
print(f"MSE : {tuned_mse:.4f}")
print(f"RMSE : {tuned_rmse:.4f}")
print(f"R2 : {tuned_r2:.4f}")



tuning_comparison = pd.DataFrame({
    "Model": [
        "Original Gradient Boosting",
        "Tuned Gradient Boosting"
    ],
    "MAE": [
        final_mae,
        tuned_mae
    ],
    "RMSE": [
        final_rmse,
        tuned_rmse
    ],
    "R2": [
        final_r2,
        tuned_r2
    ]
})


print("\n" + "=" * 60)
print("ORIGINAL VS TUNED MODEL")
print("=" * 60)

print(
    tuning_comparison.to_string(index=False)
)


tuning_comparison.to_csv(
    "../data/cleaned/tuning_comparison.csv", index=False
)

best_parameters = pd.DataFrame({
    "Parameter": list(random_search.best_params_.keys()),
    "Best_Value": list(random_search.best_params_.values())
})

best_parameters.to_csv(
    "../data/cleaned/best_hyperparameters.csv", index=False
)

print("\nTuning results saved successfully.")

print("\n" + "=" * 60)
print("STEP 10 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\n" + "=" * 60)
print("FINAL MODEL SELECTION")
print("=" * 60)


if tuned_r2 >= final_r2:
    final_model = tuned_model
    final_model_name = "Tuned Gradient Boosting"
    final_predictions = y_pred_tuned

    selected_mae = tuned_mae
    selected_mse = tuned_mse
    selected_rmse = tuned_rmse
    selected_r2 = tuned_r2
else:
    final_model = best_model
    final_model_name = "Original Gradient Boosting"
    final_predictions = best_model.predict(x_test)

    selected_mae = final_mae
    selected_mse = final_mse
    selected_rmse = final_rmse
    selected_r2 = final_r2

print("Selected Final Model:", final_model_name)

print(f"\nFinal MAE : {selected_mae:.4f}")
print(f"Final MSE : {selected_mse:.4f}")
print(f"Final RMSE : {selected_rmse:.4f}")
print(f"Final R2 : {selected_r2:.4f}")


training_predictions = final_model.predict(x_train)

train_r2 = r2_score(
    y_train,
    training_predictions
)

train_mae = mean_absolute_error(
    y_train,
    training_predictions
)

train_rmse = np.sqrt(
    mean_squared_error(
        y_train,
        training_predictions
    )
)


print("\n" + "=" * 60)
print("TRAINING VS TESTING PERFORMANCE")
print("=" * 60)

print(f"Training MAE : {train_mae:.4f}")
print(f"Testing MAE : {selected_mae:.4f}")

print(f"Training RMSE : {train_rmse:.4f}")
print(f"Testing RMSE : {selected_rmse:.4f}")

print(f"Training R2 : {train_r2:.4f}")
print(f"Testing R2 : {selected_r2:.4f}")


r2_gap = train_r2 - selected_r2

print("\nGeneralization Gap (R²):", round(r2_gap, 4))

if r2_gap < 0.10:
    print(
        "Interpretation: The training and testing R²"
        "are relatively close, indicating good generalization."
    )
else:
    print(
        "Interpretation: There is a noticeable difference "
        "between training and testing performance, so "
        "possible overfitting should be considered."
    )


validation_summary = pd.DataFrame({
    "Model": [final_model_name],
    "Training_MAE": [train_mae],
    "Testing_MAE": [selected_mae],
    "Training_RMSE": [train_rmse],
    "Testing_RMSE": [selected_rmse],
    "Training_R2": [train_r2],
    "Testing_R2": [selected_r2],
    "R2_Generalization_Gap": [r2_gap]
})

validation_summary.to_csv(
    "../data/cleaned/final_validation_summary.csv", index=False
)

print("\nFinal validation summary saved successfully.")


final_prediction_results = pd.DataFrame({
    "Actual_Yield" : y_test.values,
    "Predicted_Yield" : final_predictions,
    "Residual" : y_test.values - final_predictions,
    "Absolute_Error" : np.abs(
        y_test.values - final_predictions
    )
})

final_prediction_results.to_csv(
    "../data/cleaned/final_prediction_results.csv", index=False
)

print("Final prediction results saved successfully.")

print("\n" + "=" * 60)
print("STEP 11 COMPLETED SUCCESSFULLY")
print("=" * 60)


import joblib
from pathlib import Path

print("\n" + "=" * 60)
print("SAVING FINAL MACHINE LEARNING MODEL")
print("=" * 60)

models_dir = Path("../models")
models_dir.mkdir(parents=True, exist_ok=True)

model_path = models_dir / "final_crop_yield_model.pkl"


joblib.dump(
    final_model,
    model_path
)

print("Final model saved successfully.")
print("Model path : ", model_path)


loaded_model = joblib.load(model_path)

print("\nSaved model loaded successfully.")

verification_predictions = loaded_model.predict(x_test)

verification_r2 = r2_score(
    y_test,
    verification_predictions
)

print(f"Verification R² : {verification_r2:.4f}")



print("\n" + "=" * 60)
print("FINAL MODEL INFORMATION")
print("=" * 60)


print("Model:", final_model_name)
print("Target:", target)
print("Training Samples:", x_train.shape[0])
print("Testing Samples:", x_test.shape[0])
print("Testing R²:", round(selected_r2, 4))

print("Testing RMSE:", round(selected_rmse, 4))
print("Testing MAE:", round(selected_mae, 4))

print("\nFinal model verification completed successfully.")

print("\n" + "=" * 60)
print("STEP 12 COMPLETED SUCCESSFULLY")
print("=" * 60)

summary = {
    "Final Model": final_model_name,
    "MAE": selected_mae,
    "MSE": selected_mse,
    "RMSE": selected_rmse,
    "R2": selected_r2,
    "Training_R2": train_r2,
    "Testing_R2": selected_r2,
    "R2_Generalization_Gap": r2_gap
}

summary_df = pd.DataFrame([summary])

summary_path = "../data/cleaned/final_model_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\n===== FINAL MODEL SUMMARY =====")
print(summary_df.to_string(index=False))

print("\nFinal model summary saved to:")
print(summary_path)


