# Agricultural Crop Yield Analysis in Indian States

## Machine Learning Data Analyst – Agribusiness Internship

### Week 1: Data Collection & Cleaning
### Week 2: Exploratory Data Analysis (EDA)

---

## 📌 Project Overview

This project focuses on analyzing agricultural crop-yield data across Indian states using Python-based data analytics and machine learning techniques.

The objective of the project is to transform raw agricultural data into a clean, structured, and analytically useful dataset and then perform Exploratory Data Analysis (EDA) to identify important patterns related to crop production, agricultural yield, rainfall, cultivated area, fertilizer usage, pesticide usage, states, seasons, and yearly trends.

The project is being developed as part of the **Machine Learning Data Analyst – Agribusiness Internship**.

---

## 🎯 Project Objectives

The major objectives of this project are:

- Collect and understand agricultural crop-yield data.
- Perform data cleaning and validation.
- Handle inconsistent and invalid records.
- Analyze categorical and numerical variables.
- Study crop-wise agricultural performance.
- Compare agricultural performance across Indian states.
- Analyze seasonal patterns.
- Identify year-wise agricultural trends.
- Study correlations between agricultural variables.
- Detect and investigate statistical outliers.
- Perform Crop × Season analysis.
- Perform State × Crop analysis.
- Generate meaningful visual insights.
- Prepare the dataset for future machine learning applications.

---

# 📊 Dataset Overview

The project uses an agricultural crop-yield dataset containing records from Indian states.

### Final Dataset Statistics

| Metric | Value |
|---|---:|
| Total Records | 19,685 |
| Features | 10 |
| Crops | 55 |
| States | 30 |
| Seasons | 6 |
| Missing Values | 0 |
| Duplicate Rows | 0 |
| Negative Values | 0 |

---

## 🧾 Dataset Features

| Feature | Description |
|---|---|
| `Crop` | Name of the agricultural crop |
| `Crop_Year` | Year of the agricultural record |
| `Season` | Agricultural season |
| `State` | Indian state |
| `Area` | Cultivated area |
| `Production` | Total agricultural production |
| `Annual_Rainfall` | Annual rainfall |
| `Fertilizer` | Fertilizer usage |
| `Pesticide` | Pesticide usage |
| `Yield` | Agricultural yield |

---

# 🧹 Week 1 – Data Collection & Cleaning

During Week 1, the dataset was inspected and cleaned before performing exploratory analysis.

The cleaning workflow included:

1. Dataset loading and structural inspection.
2. Column and data-type verification.
3. Missing-value analysis.
4. Duplicate-record detection.
5. Numerical-value validation.
6. Identification of inconsistent production and yield records.
7. Validation of extreme observations.
8. Creation of the final cleaned dataset.

The final cleaned dataset contains:

**19,685 records × 10 features**

The cleaned dataset was used as the input for Week 2 Exploratory Data Analysis.

---

# 🔎 Week 2 – Exploratory Data Analysis

Week 2 focuses on understanding the statistical structure and agricultural patterns present in the cleaned dataset.

The EDA workflow includes:

### 1. Dataset Structure Analysis

- Shape
- Columns
- Data types
- Descriptive statistics
- Missing values
- Duplicate records
- Unique category counts

### 2. Categorical Analysis

Analysis was performed for:

- Crops
- States
- Seasons

Frequency distributions were used to identify the most represented categories.

### 3. Numerical Distribution Analysis

The following numerical variables were analyzed:

- Area
- Production
- Annual Rainfall
- Fertilizer
- Pesticide
- Yield

Histograms and density plots were used to understand distributions, variability, skewness, and extreme values.

---

# 🌾 Crop-wise Analysis

Crop-level analysis was performed using:

- Average Yield
- Total Production
- Number of Records

This allows agricultural productivity to be compared separately from overall production scale.

### Key Finding

**Coconut recorded the highest average Yield at 8,652.0 in the analyzed dataset.**

Coconut also recorded the highest total Production.

These results demonstrate the importance of using both average productivity and aggregate production when comparing crops.

---

# 🗺️ State-wise Analysis

State-level analysis was performed using:

- Average Yield
- Total Production
- Number of Records

### Key Findings

**Goa recorded the highest average Yield at 354.78.**

**Kerala recorded the highest total Production in the analyzed dataset.**

State-level differences may reflect variations in crop composition, cultivated area, climate, agricultural practices, and environmental conditions.

---

# 🌦️ Season-wise Analysis

Six agricultural seasons were analyzed:

- Kharif
- Rabi
- Whole Year
- Summer
- Autumn
- Winter

### Key Findings

Kharif is the most represented season with **8,229 records**.

Whole Year recorded the highest average Yield at **413.0** and also the highest total Production.

Seasonal results were interpreted together with record counts to avoid misleading comparisons caused by unequal representation.

---

# 📅 Year-wise Trend Analysis

Year-wise analysis was performed for:

- Total Production
- Average Yield
- Total Cultivated Area
- Average Annual Rainfall

Temporal analysis helps identify changes in agricultural performance over time.

However, observed trends are interpreted as associations rather than direct causal relationships because agricultural production can be affected by several interacting factors.

---

# 📈 Correlation Analysis

Correlation analysis was performed on the numerical variables to understand their linear relationships.

### Correlation with Yield

| Variable | Correlation with Yield |
|---|---:|
| Production | 0.5708 |
| Annual Rainfall | 0.0208 |
| Fertilizer | 0.0029 |
| Crop Year | 0.0025 |
| Area | 0.0019 |
| Pesticide | 0.0018 |

### Major Observation

Production shows the strongest positive linear relationship with Yield among the analyzed numerical variables.

However, correlation does not imply causation.

The weak linear correlation of rainfall, fertilizer, pesticide, and area with Yield does not necessarily mean that these variables are agriculturally unimportant. Their effects may be nonlinear, crop-specific, state-specific, or dependent on interactions with other variables.

---

# ⚠️ Outlier Analysis

Statistical outliers were identified using the **Interquartile Range (IQR) method**.

The IQR approach uses:

- Q1 – First Quartile
- Q3 – Third Quartile
- IQR = Q3 − Q1
- Lower Bound = Q1 − 1.5 × IQR
- Upper Bound = Q3 + 1.5 × IQR

### Outlier Summary

| Variable | Outliers | Percentage |
|---|---:|---:|
| Area | 3,076 | 15.62% |
| Production | 3,373 | 17.13% |
| Annual Rainfall | 1,527 | 7.76% |
| Fertilizer | 3,093 | 15.71% |
| Pesticide | 3,036 | 15.42% |
| Yield | 3,065 | 15.57% |

The identified outliers were not automatically removed because extreme agricultural observations may represent genuine differences in cultivation scale, environmental conditions, or productivity.

The outlier analysis is therefore treated as an analytical diagnostic rather than a direct data-deletion rule.

---

# 🔬 Crop × Season Analysis

A two-dimensional Crop × Season analysis was performed to investigate how crop performance varies across agricultural seasons.

The analysis includes:

- Record Count
- Average Yield
- Total Production
- Total Area
- Average Rainfall

A minimum record threshold was considered when ranking combinations to avoid over-interpreting combinations supported by very few observations.

This analysis provides a deeper view of crop-season behavior that cannot be identified through crop-level or season-level analysis alone.

---

# 🧭 State × Crop Analysis

State × Crop analysis was performed to understand regional crop specialization and productivity.

The analysis evaluates:

- Record Count
- Average Yield
- Total Production
- Total Area
- Average Rainfall

Combining State and Crop provides additional agricultural context because the performance of a crop can vary substantially between regions.

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook / Google Colab
- Git
- GitHub

---

# 📁 Project Structure

```text
Agribusiness-ML-Data-Analysis/
│
├── data/
│   └── cleaned/
│
├── src/
│   ├── data_cleaning.py
│   └── eda_analysis.py
│
├── visualizations/
│   ├── week1/
│   └── week2_eda/
│
├── reports/
│   └── Week2_EDA_Report.pdf
│
├── requirements.txt
└── README.md
