import pandas as pd
from scipy.cluster.hierarchy import inconsistent

file_path = "../data/raw/crop_yield.csv"

df = pd.read_csv("../data/raw/crop_yield.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Values:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe(include="all"))

numerical_columns = [
    "Crop_Year",
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield"
]

print("\n" + "=" * 60)
print("NUMERICAL DATA QUALITY CHECK")
print("=" * 60)

for column in numerical_columns:
    print(f"\n{column}")
    print(f"Minimum :{df[column].min()}")
    print(f"Maximum :{df[column].max()}")
    print(f"Mean :{df[column].mean():.2f}")
    print(f"Median :{df[column].median():.2f}")

print("\n" + "=" * 60)
print("OUTLIER ANALYSIS")
print("=" * 60)

outlier_columns = [
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield"
]

for column in outlier_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    percentage = (len(outliers) / len(df)) * 100

    print(f"\n{column}")
    print(f"Q1: {Q1:.2f}")
    print(f"Q3: {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower Bound: {lower_bound:.2f}")
    print(f"Upper Bound: {upper_bound:.2f}")
    print(f"Outliers: {len(outliers)}")
    print(f"Outlier Percentage: {percentage:.2f}%")


import matplotlib.pyplot as plt
import seaborn as sns

for column in outlier_columns:

    plt.figure(figsize=(10, 5))

    sns.boxplot(x=df[column])

    plt.title(f"Boxplot of {column}")
    plt.xlabel(column)

    plt.tight_layout()

    plt.savefig(f"../visualizations/{column}_boxplot.png",
                dpi=300,
                bbox_inches="tight")
    plt.close()

print("\nBoxplot created successfully.")

print("\n" + "=" * 60)
print("NEGATIVE VALUE CHECK")
print("=" * 60)

positive_columns = [
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield"
]

for column in positive_columns:

    negative_count = (df[column] < 0).sum()

    print(f"{column}: {negative_count} negative values")

print("\n" + "=" * 60)
print("ZERO VALUE CHECK")
print("=" * 60)

for column in positive_columns:
    zero_count = (df[column] == 0).sum()

    print(f"{column} : {zero_count} zero values")

print("\n" + "=" * 60)
print("ZERO PRODUCTION / ZERO YIELD ANALYSIS")
print("=" * 60)

zero_production = df[df["Production"] == 0]

zero_yield = df[df["Yield"] == 0]

print("\nZero Production Records:", len(zero_production))
print("Zero Yield Records:", len(zero_yield))

both_zero = df[
    (df["Production"] == 0) & (df["Yield"] == 0)
]

print("Both Production & Yield are Zero:", len(both_zero))

print("\nSample Zero Records:")
print(both_zero.head(10))

print("\n" + "=" * 60)
print("ZERO PRODUCTION AND ZERO YIELD RECORD ANALYSIS")
print("=" * 60)

print("\nNumber of records:", len(both_zero))

print("\nCrop distribution:")
print(both_zero["Crop"].value_counts())

print("\nSeason distribution:")
print(both_zero["Season"].value_counts())

print("\nState distribution:")
print(both_zero["State"].value_counts())

print("\nSample records:")
print(both_zero.head(10).to_string(index=False))

inconsistent_zero = df[
    ((df["Production"] == 0) & (df["Yield"] != 0)) |
    ((df["Production"] != 0) & (df["Yield"] == 0))
]

print("\n" + "=" * 60)
print("INCONSISTENT ZERO RECORDS")
print("=" * 60)

print("\nNumber of inconsistent records:", len(inconsistent_zero))

print("\nInconsistent records:")
print(inconsistent_zero.to_string(index=False))

invalid_records = df[
    (df["Production"] == 0) & (df["Yield"] > 0)
]

print("\n" + "=" * 60)
print("INVALID RECORDS IDENTIFIED")
print("=" * 60)

print("Records to remove:", len(invalid_records))

df_cleaned = df.drop(invalid_records.index).copy()

print("Original dataset shape:", df.shape)
print("Cleaned dataset shape:", df_cleaned.shape)

output_path = "../data/cleaned/crop_yield_cleaned.csv"

df_cleaned.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("CLEANED DATASET SAVED")
print("=" * 60)

print("File:", output_path)
print("Rows:", len(df_cleaned))
print("Columns:", len(df_cleaned.columns))

print("\n" + "=" * 60)
print("FINAL DATA QUALITY VALIDATION")
print("=" * 60)

print("\nMissing values:")
print(df_cleaned.isnull().sum())

print("\nTotal missing values:", df_cleaned.isnull().sum().sum())

print("\nDuplicate rows:", df_cleaned.duplicated().sum())

print("\nNegative values:")

for column in positive_columns:
    print(f"{column}: {(df_cleaned[column] < 0).sum()}")

print("\nZero Production:", (df_cleaned["Production"] == 0).sum())

print("\nZero Yield:", (df_cleaned["Yield"] == 0).sum())

print("\nFinal dataset shape:")
print(df_cleaned.shape)

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY OF CLEANED DATASET")
print("=" * 60)

statistical_summary = df_cleaned.describe()

print(statistical_summary)

print("\n" + "=" * 60)
print("CROP SUMMARY")
print("=" * 60)

print("\nNumber of unique crops:", df_cleaned["Crop"].nunique())

print("\nTop 10 crops by number of records:")
print(df_cleaned["Crop"].value_counts().head(10))

print("\n" + "=" * 60)
print("STATE SUMMARY")
print("=" * 60)

print("\nNumber of unique states:", df_cleaned["State"].nunique())

print("\nTop 10 states by number of records:")
print(df_cleaned["State"].value_counts().head(10))

top_crops = df_cleaned["Crop"].value_counts().head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_crops.values,
    y=top_crops.index
)

plt.title("Top 10 Crops by Number of Records")
plt.xlabel("Number of Records")
plt.ylabel("Crop")

plt.tight_layout()

plt.savefig("../visualizations/01_top_10_crops.png", dpi=300,bbox_inches="tight")

plt.close()
print("\nVisualization 1 created successfully.")

average_yield = (
    df_cleaned.groupby("Crop")["Yield"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=average_yield.values,
    y=average_yield.index
)

plt.title("Top 10 Crops by Average Yield")
plt.xlabel("Average Yield")
plt.ylabel("Crop")

plt.tight_layout()

plt.savefig("../visualizations/02_top_10_average_yield.png", dpi=300, bbox_inches="tight")

plt.close()

print("Visualization 2 created successfully.")

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df_cleaned,
    x="Annual_Rainfall",
    y="Yield",
    alpha=0.5
)

plt.title("Annual Rainfall vs Crop Yield")
plt.xlabel("Annual Rainfall")
plt.ylabel("Yield")

plt.tight_layout()

plt.savefig(
    "../visualizations/03_rainfall_vs_yield.png", dpi=300, bbox_inches="tight"
)

plt.close()

print("Visualization 3 created successfully.")

top_production = (
    df_cleaned
    .groupby("Crop")["Production"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_production.values,
    y=top_production.index
)

plt.title("Top 10 Crops by Total Production")

plt.xlabel("Total Production")
plt.ylabel("Crop")

plt.tight_layout()

plt.savefig(
    "../visualizations/04_top_10_production.png", dpi=300, bbox_inches="tight"
)

plt.close()

print("Visualization 4 created successfully.")

correlation_columns = [
    "Crop_Year",
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield"
]

correlation_matrix = df_cleaned[correlation_columns].corr()

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Agricultural Variables")

plt.tight_layout()

plt.savefig(
    "../visualizations/05_correlation_heatmap.png", dpi=300, bbox_inches="tight"
)

plt.close()

print("Visualization 5 created successfully.")

print("\n" + "=" * 60)
print("FINAL CLEANED DATASET VALIDATION")
print("=" * 60)

print("\nFinal Shape:")
print(df_cleaned.shape)

print("\nMissing Values:")
print(df_cleaned.isnull().sum())

print("\nTotal Missing Values:")
print(df_cleaned.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df_cleaned.duplicated().sum())

print("\nNegative Values:")

for column in positive_columns:
    print(f"{column}:{(df_cleaned[column] < 0).sum()}")

print("\nZero Production:")
print((df_cleaned["Production"] == 0).sum())

print("\nZero Yield:")
print((df_cleaned["Yield"] == 0).sum())

print("\nUnique Crops:")
print(df_cleaned["Crop"].nunique())

print("\nUnique States:")
print(df_cleaned["State"].nunique())

print("\nUnique Seasons:")
print(df_cleaned["Season"].nunique())

statistical_summary = df_cleaned.describe()

statistical_summary.to_csv(
    "../data/cleaned/statistical_summary.csv"
)

crop_summary = (
    df_cleaned
    .groupby("Crop")
    .agg(
        Record_Count=("Crop", "size"),
        Average_Yield=("Yield", "mean"),
        Total_Production=("Production", "sum"),
        Average_Rainfall=("Annual_Rainfall", "mean")
    )
    .sort_values("Average_Yield", ascending=False)
)

crop_summary.to_csv(
    "../data/cleaned/crop_summary.csv"
)

state_summary = (
    df_cleaned
    .groupby("State")
    .agg(
        Record_Count=("State", "size"),
        Average_Yield=("Yield", "mean"),
        Total_Production=("Production", "sum"),
        Average_Rainfall=("Annual_Rainfall", "mean")
    )
    .sort_values("Average_Yield", ascending=False)

)

state_summary.to_csv(
    "../data/cleaned/state_summary.csv"
)

season_summary = (
    df_cleaned
    .groupby("Season")
    .agg(
        Record_Count=("Season", "size"),
        Average_Yield=("Yield", "mean"),
        Total_Production=("Production", "sum"),
        Average_Rainfall=("Annual_Rainfall", "mean")
    )
    .sort_values("Average_Yield", ascending=False)
)

season_summary.to_csv(
    "../data/cleaned/season_summary.csv"
)

print("\n" + "=" * 60)
print("ANALYSIS RESULT FILES SAVED")
print("=" * 60)

print("statistical_summary.csv")
print("crop_summary.csv")
print("state_summary.csv")
print("season_summary.csv")

# ============================================================
# KEY AGRICULTURAL INSIGHTS
# ============================================================

print("\n" + "=" * 60)
print("KEY AGRICULTURAL INSIGHTS")
print("=" * 60)


# 1. Highest average yield crop
highest_yield_crop = (
    crop_summary["Average_Yield"]
    .idxmax()
)

highest_yield_value = (
    crop_summary["Average_Yield"]
    .max()
)

print("\n1. Highest Average Yield Crop:")
print(highest_yield_crop)
print("Average Yield:", round(highest_yield_value, 2))


# 2. Highest total production crop
highest_production_crop = (
    crop_summary["Total_Production"]
    .idxmax()
)

highest_production_value = (
    crop_summary["Total_Production"]
    .max()
)

print("\n2. Highest Total Production Crop:")
print(highest_production_crop)
print("Total Production:",
      round(highest_production_value, 2))


# 3. Highest average yield state
highest_yield_state = (
    state_summary["Average_Yield"]
    .idxmax()
)

highest_state_yield_value = (
    state_summary["Average_Yield"]
    .max()
)

print("\n3. Highest Average Yield State:")
print(highest_yield_state)
print("Average Yield:",
      round(highest_state_yield_value, 2))


# 4. Highest total production state
highest_production_state = (
    state_summary["Total_Production"]
    .idxmax()
)

highest_state_production_value = (
    state_summary["Total_Production"]
    .max()
)

print("\n4. Highest Total Production State:")
print(highest_production_state)
print("Total Production:",
      round(highest_state_production_value, 2))


# 5. Best season by average yield
best_yield_season = (
    season_summary["Average_Yield"]
    .idxmax()
)

best_season_yield_value = (
    season_summary["Average_Yield"]
    .max()
)

print("\n5. Highest Average Yield Season:")
print(best_yield_season)
print("Average Yield:",
      round(best_season_yield_value, 2))


# 6. Best season by total production
best_production_season = (
    season_summary["Total_Production"]
    .idxmax()
)

best_season_production_value = (
    season_summary["Total_Production"]
    .max()
)

print("\n6. Highest Total Production Season:")
print(best_production_season)
print("Total Production:",
      round(best_season_production_value, 2))


# 7. Correlation with Yield
print("\n7. Correlation with Yield:")

yield_correlation = (
    correlation_matrix["Yield"]
    .sort_values(ascending=False)
)

print(yield_correlation)


findings_path = "../data/cleaned/final_key_findings.txt"

with open(findings_path, "w", encoding="utf-8") as file:

    file.write("AGRICULTURAL DATA ANALYSIS - KEY FINDINGS\n")
    file.write("=" * 60 + "\n\n")

    file.write("DATASET OVERVIEW\n")
    file.write("-" * 60 + "\n")
    file.write(f"Final dataset size: {df_cleaned.shape[0]} rows x {df_cleaned.shape[1]} columns\n")
    file.write(f"Unique crops: {df_cleaned['Crop'].nunique()}\n")
    file.write(f"Unique states: {df_cleaned['State'].nunique()}\n")
    file.write(f"Unique seasons: {df_cleaned['Season'].nunique()}\n\n")

    file.write("DATA QUALITY\n")
    file.write("-" * 60 + "\n")
    file.write(f"Missing values: {df_cleaned.isnull().sum().sum()}\n")
    file.write(f"Duplicate rows: {df_cleaned.duplicated().sum()}\n")
    file.write("Negative values: 0\n")
    file.write(f"Zero production records: {(df_cleaned['Production'] == 0).sum()}\n")
    file.write(f"Zero yield records: {(df_cleaned['Yield'] == 0).sum()}\n\n")

    file.write("KEY AGRICULTURAL FINDINGS\n")
    file.write("-" * 60 + "\n")
    file.write(
        f"Highest average-yield crop: {highest_yield_crop} "
        f"({highest_yield_value:.2f})\n"
    )

    file.write(
        f"Highest total-production crop: {highest_production_crop} "
        f"({highest_production_value:.2f})\n"
    )

    file.write(
        f"Highest average-yield state: {highest_yield_state} "
        f"({highest_state_yield_value:.2f})\n"
    )

    file.write(
        f"Highest total-production state: {highest_production_state} "
        f"({highest_state_production_value:.2f})\n"
    )

    file.write(
        f"Highest average-yield season: {best_yield_season} "
        f"({best_season_yield_value:.2f})\n"
    )

    file.write(
        f"Highest total-production season: {best_production_season} "
        f"({best_season_production_value:.2f})\n\n"
    )

    file.write("CORRELATION WITH YIELD\n")
    file.write("-" * 60 + "\n")

    for variable, value in yield_correlation.items():
        file.write(f"{variable}: {value:.6f}\n")

print("\nFinal key findings saved successfully.")
print("File:", findings_path)


