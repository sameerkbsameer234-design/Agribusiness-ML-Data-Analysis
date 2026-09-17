# Agribusiness ML & Data Analysis

## Agricultural Crop Yield Prediction Using Machine Learning

This project focuses on analyzing agricultural crop data from Indian states and developing a machine learning model to predict crop yield.

The project was developed as part of the Machine Learning Data Analyst – Agribusiness internship.

---

## Project Objective

The main objective of this project is to analyze agricultural data, identify meaningful patterns and relationships, and build a machine learning regression model for agricultural crop yield prediction.

The project covers:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Statistical analysis
- Data visualization
- Machine learning model development
- Model comparison
- Hyperparameter optimization
- Model evaluation
- Prediction validation
- Feature importance analysis

---

## Dataset

The dataset contains agricultural crop information from different states of India.

### Dataset Features

| Feature | Description |
|---|---|
| Crop | Name of the agricultural crop |
| Crop_Year | Year of cultivation |
| Season | Agricultural cultivation season |
| State | Indian state |
| Area | Cultivated area |
| Production | Crop production |
| Annual_Rainfall | Annual rainfall |
| Fertilizer | Fertilizer usage |
| Pesticide | Pesticide usage |
| Yield | Crop yield |

The target variable for the machine learning model is:

**Yield**

---

## Project Workflow

### 1. Data Cleaning

The dataset was inspected for:

- Missing values
- Duplicate records
- Negative values
- Invalid production/yield combinations
- Data consistency

After cleaning, the final dataset contained:

**19,685 records and 10 features.**

---

### 2. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand:

- Crop distribution
- State-wise agricultural production
- Seasonal patterns
- Year-wise trends
- Crop yield variation
- Production patterns
- Correlation between numerical variables
- Outliers and unusual observations

Multiple visualizations were created using Python.

---

### 3. Machine Learning

The machine learning problem was formulated as:

**Supervised Learning → Regression**

Target variable:

**Yield**

The following input features were used:

- Crop
- Crop_Year
- Season
- State
- Area
- Annual_Rainfall
- Fertilizer
- Pesticide

Production was excluded from the predictive feature set because it has a direct structural relationship with area and yield and could make the prediction setup less realistic.

---

## Machine Learning Models

Three regression models were developed and compared:

### Linear Regression

Used as the baseline regression model.

### Random Forest Regressor

An ensemble tree-based regression model used to capture nonlinear relationships.

### Gradient Boosting Regressor

An ensemble boosting model used to improve prediction performance by sequentially learning from previous errors.

---

## Data Preprocessing

Categorical variables were encoded using:

**One-Hot Encoding**

Categorical features:

- Crop
- Season
- State

Numerical features were passed directly to the models.

A Scikit-learn Pipeline and ColumnTransformer were used to keep preprocessing and modelling together.

---

## Model Evaluation

The models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Initial Model Comparison

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 148.40 | 920.26 | 0.000046 |
| Random Forest | 21.66 | 276.93 | 0.909447 |
| Gradient Boosting | 12.72 | 161.57 | 0.969176 |

The initial evaluation showed that Gradient Boosting produced the strongest test-set performance among the three models.

---

## Hyperparameter Optimization

RandomizedSearchCV with 3-fold cross-validation was used to search for improved Gradient Boosting configurations.

The parameters explored included:

- Number of estimators
- Learning rate
- Maximum depth
- Minimum samples split
- Minimum samples leaf

The final model was selected by comparing the tuned model with the original Gradient Boosting model using test-set performance.

---

## Model Validation

The selected final model was evaluated on both training and testing data to examine generalization performance.

Additional validation included:

- Actual vs Predicted Yield
- Residual Distribution
- Residuals vs Predicted Values
- MAE
- RMSE
- R² Score
- Training vs Testing performance

---

## Feature Importance

Feature importance was extracted from the final Gradient Boosting model to understand which input variables contributed most to the model's predictions.

Feature importance represents model contribution and should not be interpreted as direct causal influence.

---

## Prediction Demonstration

A real record from the cleaned dataset was selected to demonstrate the final model's prediction capability.

The demonstration compares:

- Actual Yield
- Predicted Yield
- Absolute Prediction Error

The prediction result is stored in:

`data/cleaned/sample_prediction.csv`

---

## Project Structure

```text
Agribusiness-ML-Data-Analysis/
│
├── data/
│   └── cleaned/
│
├── models/
│   └── final_crop_yield_model.pkl
│
├── src/
│   ├── data_cleaning.py
│   ├── eda_analysis.py
│   ├── modeling.py
│   └── predict_yield.py
│
├── visualizations/
│   ├── week1/
│   ├── week2_eda/
│   └── Week 3 visualizations
│
├── reports/
│
├── README.md
└── requirements.txt
