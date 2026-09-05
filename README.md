# Agricultural Crop Yield Data Collection, Cleaning and Analysis

## Project Overview

This project focuses on collecting, cleaning, preprocessing, and analyzing an agricultural crop yield dataset. The dataset contains information about crops, crop years, seasons, states, cultivated area, production, annual rainfall, fertilizer usage, pesticide usage, and crop yield.

The main objective is to prepare a reliable and analysis-ready agricultural dataset by identifying data-quality issues, handling invalid records, investigating outliers, performing statistical analysis, and generating meaningful visualizations.

This project was completed as part of the Machine Learning Data Analyst – Agribusiness internship.

---

## Objectives

- Collect a publicly available agricultural dataset.
- Understand the structure and characteristics of the dataset.
- Identify missing values and duplicate records.
- Check and handle negative or invalid values.
- Investigate zero-value records.
- Identify statistical outliers using the IQR method.
- Prepare a cleaned dataset for further analysis.
- Perform statistical and agricultural analysis.
- Generate meaningful visualizations.
- Extract useful insights related to crop yield, production, states, seasons, and rainfall.

---

## Dataset

The dataset contains **19,689 records and 10 columns** before cleaning.

### Features

| Column | Description |
|---|---|
| Crop | Name of the agricultural crop |
| Crop_Year | Year of crop production |
| Season | Agricultural season |
| State | State where the crop was produced |
| Area | Cultivated area |
| Production | Crop production |
| Annual_Rainfall | Annual rainfall |
| Fertilizer | Fertilizer usage |
| Pesticide | Pesticide usage |
| Yield | Crop yield |

---

## Data Cleaning Process

The following data-quality checks and preprocessing steps were performed:

1. Dataset structure and data types were inspected.
2. Missing values were checked.
3. Duplicate records were identified.
4. Negative values were checked.
5. Zero-production and zero-yield records were investigated.
6. Inconsistent records were identified.
7. Four records with zero production but non-zero yield were removed.
8. Statistical outliers were identified using the Interquartile Range (IQR) method.
9. Agricultural outliers were retained because extreme values may represent genuine large-scale agricultural observations.
10. The cleaned dataset was validated after preprocessing.

---

## Final Dataset

After cleaning:

- Rows: **19,685**
- Columns: **10**
- Missing values: **0**
- Duplicate rows: **0**
- Negative values: **0**
- Zero Production records: **108**
- Zero Yield records: **112**
- Unique Crops: **55**
- Unique States: **30**
- Unique Seasons: **6**

---

## Statistical Analysis

Statistical summaries were generated for the numerical variables including:

- Crop Year
- Area
- Production
- Annual Rainfall
- Fertilizer
- Pesticide
- Yield

The analysis includes count, mean, standard deviation, minimum, quartiles, median, and maximum values.

---

## Visualizations

The project includes the following visualizations:

1. Top 10 Crops by Number of Records
2. Top 10 Crops by Average Yield
3. Annual Rainfall vs Crop Yield
4. Top 10 Crops by Total Production
5. Correlation Heatmap
6. Boxplots for Outlier Analysis

All visualizations were created using Python visualization libraries.

---

## Key Findings

- Coconut recorded the highest average yield with an average value of **8652.0**.
- Coconut also recorded the highest total production in the dataset.
- Goa had the highest average yield among the states, with an average yield of **354.78**.
- Kerala recorded the highest total production among the states.
- The Whole Year season had the highest average yield at **413.0**.
- The Whole Year season also recorded the highest total production.
- Production showed a moderate positive correlation with Yield (**0.5708**).
- Annual Rainfall showed a very weak positive correlation with Yield (**0.0208**).
- Fertilizer and Pesticide showed very weak linear correlations with Yield in this dataset.

> Correlation values indicate statistical association and should not be interpreted as proof of causation.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- PyCharm
- GitHub

---

## Project Structure

```text
ML_Week1/
│
├── data/
│   ├── raw/
│   │   └── crop_yield.csv
│   │
│   └── cleaned/
│       ├── crop_yield_cleaned.csv
│       ├── statistical_summary.csv
│       ├── crop_summary.csv
│       ├── state_summary.csv
│       ├── season_summary.csv
│       └── final_key_findings.txt
│
├── src/
│   └── data_cleaning.py
│
├── visualizations/
│   ├── 01_top_10_crops.png
│   ├── 02_top_10_average_yield.png
│   ├── 03_rainfall_vs_yield.png
│   ├── 04_top_10_production.png
│   ├── 05_correlation_heatmap.png
│   └── boxplots/
│
├── README.md
└── requirements.txt

## Conclusion

This project successfully collected, cleaned, preprocessed, and analyzed an agricultural crop yield dataset. A systematic data-quality assessment was performed to identify missing values, duplicate records, negative values, inconsistent zero values, and statistical outliers.

After the cleaning process, the final dataset contained 19,685 records and 10 features, with no missing values, duplicate rows, or negative values. The analysis also explored crop-wise, state-wise, and season-wise agricultural patterns using statistical summaries and visualizations.

The findings provide useful insights into crop yield, production, rainfall, and agricultural resource usage. The cleaned dataset can serve as a reliable foundation for future machine learning applications such as crop-yield prediction, production forecasting, and agricultural decision-support systems.

---

## Author

**Sameer KB**

Machine Learning Data Analyst – Agribusiness Intern


