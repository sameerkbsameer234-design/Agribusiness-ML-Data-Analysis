import pandas as pd
import numpy as np

file_path = "../data/cleaned/crop_yield_cleaned.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)
print(df.columns.tolist())

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("LAST 5 ROWS")
print("=" * 60)
print(df.tail())

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(df.info())

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)
print(df.duplicated().sum())

print("\n" + "=" * 60)
print("UNIQUE VALUES")
print("=" * 60)

for column in df.columns:
    print(f"{column}:{df[column].nunique()} unique values")


import matplotlib.pyplot as plt
import seaborn as sns

top_crops = df["Crop"].value_counts().head(10)

print("\n" + "=" * 60)
print("TOP 10 CROPS BY NUMBER OF RECORDS")
print("=" * 60)
print(top_crops)

plt.figure(figsize=(10,6))
sns.barplot(x=top_crops.values, y=top_crops.index)

plt.title("Top 10 Crops by Number of Records")
plt.xlabel("Number of Records")
plt.ylabel("Crop")
plt.tight_layout()

plt.savefig("../visualizations/eda/06_top_10_crops_records.png", dpi=300)
plt.show()


season_counts = df["Season"].value_counts()

print("\n" + "=" * 60)
print("SEASON DISTRIBUTION")
print("=" * 60)
print(season_counts)

plt.figure(figsize=(10, 6))
sns.barplot(x=season_counts.index, y=season_counts.values)

plt.title("Distribution of Agriculture Records by Season")
plt.xlabel("Season")
plt.ylabel("Number of Records")
plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig("../visualizations/eda/07_season_distribution.png", dpi=300)
plt.show()

state_counts = df["State"].value_counts().head(10)

print("\n" + "=" * 60)
print("TOP 10 STATE BY NUMBER OF RECORDS")
print("=" * 60)
print(state_counts)

plt.figure(figsize=(10, 6))
sns.barplot(x=state_counts.values, y=state_counts.index)

plt.title("Top 10 States by Number of Agriculture Records")
plt.xlabel("Number of Records")
plt.ylabel("State")

plt.tight_layout()

plt.savefig("../visualizations/eda/08_top_10_states_records.png", dpi=300)

plt.show()

numerical_columns = [
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield"
]

print("\n" + "=" * 60)
print("NUMERICAL DISTRIBUTION ANALYSIS")
print("=" * 60)

for column in numerical_columns:

    print(f"\n{column}")
    print("-" * 40)
    print("Mean :", df[column].mean())
    print("Median :", df[column].median())
    print("Std :", df[column].std())
    print("Min :", df[column].min())
    print("Max :", df[column].max())

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df[column],
        bins=50,
        kde=True
    )

    plt.title(f"Distribution of {column}")

    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()

    safe_name = column.lower()
    plt.savefig(f"../visualizations/eda/04_distribution_{safe_name}.png", dpi=300)

    plt.show()

crop_yield = df.groupby("Crop")["Yield"].mean().sort_values(ascending=False)

print("\n" + "=" * 60)
print("TOP 10 CROPS BY AVERAGE YIELD")
print("=" * 60)
print(crop_yield.head(10))

plt.figure(figsize=(10, 6))

sns.barplot(
    x=crop_yield.head(10).values,
    y=crop_yield.head(10).index
)

plt.title("Top 10 Crops by Average Yield")
plt.xlabel("Average Yield")
plt.ylabel("Crop")
plt.tight_layout()

plt.savefig("../visualizations/eda/09_top_10_average_yield.png", dpi=300)
plt.show()

crop_production = (
    df.groupby("Crop")["Production"].sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("TOP 10 CROPS BY TOTAL PRODUCTION")
print("=" * 60)
print(crop_production.head(10))

plt.figure(figsize=(10, 6))

sns.barplot(
    x=crop_production.head(10).values,
    y=crop_production.head(10).index
)

plt.title("Top 10 Crops by Total Production")
plt.xlabel("Total Production")
plt.ylabel("Crop")
plt.tight_layout()

plt.savefig("../visualizations/eda/010_top_10_total_production.png", dpi=300)

plt.show()

state_yield = (
    df.groupby("State")["Yield"].mean()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("TOP 10 STATES BY AVERAGE YIELD")
print("=" * 60)
print(state_yield.head(10))

plt.figure(figsize=(10, 6))

sns.barplot(
    x=state_yield.head(10).values,
    y=state_yield.head(10).index
)

plt.title("Top 10 States by Average Yield")
plt.xlabel("Average Yield")
plt.ylabel("State")
plt.tight_layout()

plt.savefig("../visualizations/eda/011_top_10_states_average_yield.png", dpi=300)

plt.show()

state_production = (
    df.groupby("State")["Production"].sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("TOP 10 STATES BY TOTAL PRODUCTION")
print("=" * 60)
print(state_production.head(10))

plt.figure(figsize=(10, 6))

sns.barplot(
    x=state_production.head(10).values,
    y=state_production.head(10).index
)

plt.title("Top 10 States by Total Agriculture Production")
plt.xlabel("Total Production")
plt.ylabel("State")
plt.tight_layout()

plt.savefig("../visualizations/eda/012_states_total_production.png", dpi=300)

plt.show()

yearly_analysis = (
    df.groupby("Crop_Year").agg(
        Total_Production=("Production", "sum"),
        Average_Yield=("Yield", "mean"),
        Total_Area=("Area", "sum"),
        Average_Rainfall=("Annual_Rainfall", "mean")
    )
    .reset_index()
)

print("\n" + "=" * 60)
print("YEAR-WISE AGRICULTURE TREND ANALYSIS")
print("=" * 60)

print(yearly_analysis.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.plot(
    yearly_analysis["Crop_Year"],
    yearly_analysis["Total_Production"],
    marker="o"
)
plt.title("Year-wise Total Agriculture Production")
plt.xlabel("Crop_Year")
plt.ylabel("Total Production")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("../visualizations/eda/013_yearly_total_production.png", dpi=300)

plt.show()

plt.figure(figsize=(10, 6))

plt.plot(
    yearly_analysis["Crop_Year"],
    yearly_analysis["Average_Yield"],
    marker="o"
)

plt.title("Year-wise Average Crop Yield")
plt.xlabel("Crop_Year")
plt.ylabel("Average Yield")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("../visualizations/eda/014_yearly_average_yield.png", dpi=300)

plt.show()

plt.figure(figsize=(10, 6))

plt.plot(
    yearly_analysis["Crop_Year"],
    yearly_analysis["Average_Rainfall"],
    marker="o"
)

plt.title("Year-wise Average Annual Rainfall")
plt.xlabel("Crop_Year")
plt.ylabel("Average Annual Rainfall")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("../visualizations/eda/015_yearly_average_rainfall.png", dpi=300)

plt.show()

numerical_columns = [
    "Crop_Year",
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield"
]

correlation_matrix = df[numerical_columns].corr()

print("\n" + "=" * 60)
print("CORRELATION WITH CROP YIELD")
print("=" * 60)

yield_correlation = (
    correlation_matrix["Yield"].drop("Yield")
    .sort_values(ascending=False)
)

print(yield_correlation)

print("\n" + "=" * 60)
print("STRONGEST RELATIONSHIPS WITH YIELD")
print("=" * 60)

for variable, value in yield_correlation.items():
    print(f"{variable:20s} : {value:.4f}")


plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Production",
    y="Yield",
    alpha=0.4
)

plt.title("Production vs Crop Yield")
plt.xlabel("Production")
plt.ylabel("Yield")
plt.tight_layout()

plt.savefig("../visualizations/eda/015_production_vs_yield.png", dpi=300)

plt.show()

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Area",
    y="Yield",
    alpha=0.4
)

plt.title("Area vs Crop Yield")
plt.xlabel("Area")
plt.ylabel("Yield")
plt.tight_layout()

plt.savefig("../visualizations/eda/016_area_vs_yield.png", dpi=300)

plt.show()


plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual_Rainfall",
    y="Yield",
    alpha=0.4
)

plt.title("Annual Rainfall vs Crop Yield")
plt.xlabel("Annual Rainfall")
plt.ylabel("Yield")
plt.tight_layout()

plt.savefig("../visualizations/eda/017_rainfall_vs_yield.png", dpi=300)

plt.show()

print("\n" + "=" * 60)
print("DETAILED OUTLIER ANALYSIS - IQR METHOD")
print("=" * 60)

numerical_columns = [
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield"
]

outlier_results = []

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_mask = (
    (df[column] < lower_bound) | (df[column] > upper_bound)
    )

    outlier_count = outlier_mask.sum()
    outlier_percentage = (outlier_count / len(df)) * 100

    outlier_results.append({
        "Column": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier_Count": outlier_count,
        "Outlier_Percentage": outlier_percentage
    })

    print(f"\n{column}")
    print("-" * 60)
    print(f"Q1 : {Q1:.4f}")
    print(f"Q3 : {Q3:.4f}")
    print(f"IQR : {IQR:.4f}")
    print(f"Lower Bound : {lower_bound:.4f}")
    print(f"Upper Bound : {upper_bound:.4f}")
    print(f"Outlier Count : {outlier_count}")
    print(f"Outlier Percentage : {outlier_percentage:.2f}")

    outlier_summary = pd.DataFrame(outlier_results)

    outlier_summary.to_csv(
        "../data/cleaned/outlier_analysis.csv", index=False
    )

    print("\n" + "=" * 60)
    print("OUTLIER SUMMARY")
    print("=" * 60)

    print(outlier_summary.to_string(index=False))


    print("\n" + "=" * 60)
    print("EXTREME VALUE EXAMPLES")
    print("=" * 60)

    for column in numerical_columns:

        print(f"\n--- {column} ---")

        print("Lowest values:")
        print(
            df[["Crop", "State", "Crop_Year", column]]
            .sort_values(column)
            .head(3)
            .to_string(index=False)
        )

        print("\nHighest values:")
        print(
            df[["Crop", "State", "Crop_Year", column]]
            .sort_values(column, ascending=False)
            .head(3)
            .to_string(index=False)
        )

plt.figure(figsize=(10, 6))

sns.barplot(
    data=outlier_summary,
    x="Outlier_Percentage",
    y="Column"
)

plt.title("Outlier Percentage by Numerical Variable")
plt.xlabel("Outlier Percentage (%)")
plt.ylabel("Numerical Variable")

plt.tight_layout()

plt.savefig("../visualizations/eda/018_outlier_percentage.png", dpi=300)

plt.show()

print("\n" + "=" * 60)
print("SEASON-WISE AGRICULTURAL ANALYSIS")
print("=" * 60)

season_analysis = (
    df.groupby("Season")
    .agg(
        Record_Count=("Crop", "count"),
        Average_Yield=("Yield", "mean"),
        Total_Production=("Production", "sum"),
        Total_Area=("Area", "sum"),
        Average_Rainfall=("Annual_Rainfall", "mean")
    )
    .sort_values("Average_Rainfall", ascending=False)
    .reset_index()
)

print("\nSeason-wise Summary:")
print(season_analysis.to_string(index=False))

best_yield_season = season_analysis.iloc[0]

print("\n" + "=" * 60)
print("HIGHEST AVERAGE YIELD SEASON")
print("=" * 60)

print("Season :", best_yield_season["Season"])
print("Average Yield :", round(best_yield_season["Average_Yield"], 2))

best_production_season = (
    season_analysis
    .sort_values("Total_Production", ascending=False).iloc[0]
)

print("\n" + "=" * 60)
print("HIGHEST TOTAL PRODUCTION SEASON")
print("=" * 60)

print("Season :", best_production_season["Season"])

print(
    "Total Production :",
    round(best_production_season["Total_Production"], 2)
)


season_analysis.to_csv(
    "../data/cleaned/season_eda_summary.csv", index=False
)


yield_by_season = (
    df.groupby("Season")["Yield"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=yield_by_season.values,
    y=yield_by_season.index
)

plt.title("Average Crop Yield by Season")
plt.xlabel("Average Yield")
plt.ylabel("Season")

plt.tight_layout()

plt.savefig("../visualizations/eda/016_season_average_yield.png", dpi=300)

plt.show()

production_by_season = (
    df.groupby("Season")["Production"].sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=production_by_season.values,
    y=production_by_season.index
)

plt.title("Total Agricultural Production by Season")
plt.xlabel("Total Production")
plt.ylabel("Season")

plt.tight_layout()

plt.savefig(
    "../visualizations/eda/017_season_total_production.png", dpi=300
)

plt.show()

print("\n" + "=" * 60)
print("CROP x SEASON ANALYSIS")
print("=" * 60)

crop_season_summary = (
    df.groupby(["Crop", "Season"])
    .agg(
        Record_Count=("Crop", "size"),
        Average_Yield=("Yield", "mean"),
        Total_Production=("Production", "sum"),
        Total_Area=("Area", "sum"),
        Average_Rainfall=("Annual_Rainfall", "mean")
    )
    .reset_index()
)

crop_season_summary["Average_Yield"] = crop_season_summary["Average_Yield"].round(2)
crop_season_summary["Average_Rainfall"] = crop_season_summary["Average_Rainfall"].round(2)

crop_season_summary.to_csv(
    "../data/cleaned/crop_season_eda_summary.csv", index=False
)

print("\nCrop x Season summary created.")
print(crop_season_summary.head(10))

reliable_combinations = crop_season_summary[
    crop_season_summary["Record_Count"] >= 10
].copy()

top_crop_seasons = reliable_combinations.sort_values(
    by="Average_Yield", ascending=False
).head(10)

print("\nTop 10 Crop x Season Combinations by Average Yield:")
print(
    top_crop_seasons[
        ["Crop", "Season", "Record_Count", "Average_Yield"]
    ].to_string(index=False)
)

best_season_per_crop = (
    reliable_combinations
    .sort_values("Average_Yield", ascending=False)
    .groupby("Crop")
    .first()
    .reset_index()
)

print("\nBest-performing season for each crop:")
print(
    best_season_per_crop[
        ["Crop", "Season", "Record_Count", "Average_Yield"]
    ].head(15).to_string(index=False)
)

top_15_crops = (
    df["Crop"]
    .value_counts()
    .head(15)
    .index
)

heatmap_data = (
    crop_season_summary[
        crop_season_summary["Crop"].isin(top_15_crops)
    ]
    .pivot(
        index="Crop",
        columns="Season",
        values="Average_Yield"
    )
)

plt.figure(figsize=(12, 8))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1f",
    cmap="viridis",
    linewidths=0.5
)

plt.title("Average Yield by Crop and Season")
plt.xlabel("Season")
plt.ylabel("Crop")

plt.tight_layout()

plt.savefig("../visualizations/eda/018_crop_season_yield_heatmap.png", dpi=300, bbox_inches="tight")

plt.show()


plot_data = top_crop_seasons.copy()

plot_data["Combination"] = (
    plot_data["Crop"] + " - " + plot_data["Season"]
)

plt.figure(figsize=(12, 7))

sns.barplot(
    data=plot_data,
    x="Average_Yield",
    y="Combination"
)

plt.title("Top 10 Crop x Season Combinations by Average Yield")
plt.xlabel("Average Yield")
plt.ylabel("Crop - Season")

plt.tight_layout()

plt.savefig(
    "../visualizations/eda/019_top_crop_season_combinations.png", dpi=300, bbox_inches="tight"
)

plt.show()

print("\nStep 11 completed successfully.")

print("\n" + "=" * 60)
print("STEP 12: STATE x CROP ANALYSIS")
print("=" * 60)


state_crop_summary = (
    df.groupby(["State", "Crop"])
    .agg(
        Record_Count=("Crop", "size"),
        Average_Yield=("Yield", "mean"),
        Total_Production=("Production", "sum"),
        Total_Area=("Area", "sum")
    )
    .reset_index()
)

state_crop_summary["Average_Yield"] = (
    state_crop_summary["Average_Yield"].round(2)
)

state_crop_summary.to_csv(
    "../data/cleaned/state_crop_eda_summary.csv", index=False
)

print("\nState x Crop Summary Created.")
print(state_crop_summary.head(10))


top_state_crop_production = (
    state_crop_summary.sort_values("Total_Production", ascending=False).head(10)
)

print("\nTop 10 State x Crop Combinations by Total Production:")

print(
    top_state_crop_production[
        [
            "State",
            "Crop",
            "Record_Count",
            "Total_Production",
            "Average_Yield"
        ]
    ].to_string(index=False)
)

reliable_state_crop = state_crop_summary[
    state_crop_summary["Record_Count"] >= 10
].copy()

top_state_crop_yield = (
    reliable_state_crop.sort_values("Average_Yield", ascending=False).head(10)
)

print("\nTop 10 State x Crop Combinations by Average Yield:")

print(
    top_state_crop_yield[
        [
            "State",
            "Crop",
            "Record_Count",
            "Average_Yield"
        ]
    ].to_string(index=False)
)

main_crop_per_state = (
    state_crop_summary.sort_values("Total_Production", ascending=False)
    .groupby("State")
    .first()
    .reset_index()
)

print("\nMain crop by total production for each state:")

print(
    main_crop_per_state[
        [
            "State",
            "Crop",
            "Total_Production",
            "Average_Yield"
        ]
    ].head(15).to_string(index=False)
)

top_states = (
    df["State"]
    .value_counts()
    .head(12)
    .index
)

top_crops = (
    df["Crop"]
    .value_counts()
    .head(12)
    .index
)

heatmap_state_crop = (
    state_crop_summary[
        state_crop_summary["State"].isin(top_states) & state_crop_summary["Crop"].isin(top_crops)
    ]
    .pivot(
        index="State",
        columns="Crop",
        values="Average_Yield"
    )
)

plt.figure(figsize=(15, 9))

sns.heatmap(
    heatmap_state_crop,
    annot=True,
    cmap="viridis",
    linewidths=0.3
)

plt.title("Average Yield Across Major States and Crops")
plt.xlabel("Crop")
plt.ylabel("State")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    "../visualizations/eda/20_state_crop_yield_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


plot_data = top_state_crop_production.copy()

plot_data["Combination"] = (
    plot_data["State"] + " - " + plot_data["Crop"]
)

plt.figure(figsize=(12, 7))

sns.barplot(
    data=plot_data,
    x="Total_Production",
    y="Combination"
)

plt.title("Top 10 State x Crop Combinations by Total Production")
plt.xlabel("Total Production")
plt.ylabel("State - Crop")

plt.tight_layout()

plt.savefig(
    "../visualizations/eda/21_top_state_crop_production.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nStep 12 completed successfully.")




# ============================================================
# STEP 13: FINAL EDA INSIGHTS & STATISTICAL FINDINGS
# ============================================================

print("\n" + "=" * 60)
print("STEP 13: FINAL EDA INSIGHTS & STATISTICAL FINDINGS")
print("=" * 60)


# ------------------------------------------------------------
# 1. Dataset-level statistics
# ------------------------------------------------------------

total_records = len(df)
total_crops = df["Crop"].nunique()
total_states = df["State"].nunique()
total_seasons = df["Season"].nunique()
year_start = df["Crop_Year"].min()
year_end = df["Crop_Year"].max()

print("\nDATASET OVERVIEW")
print("-" * 40)
print(f"Total records       : {total_records:,}")
print(f"Unique crops        : {total_crops}")
print(f"Unique states       : {total_states}")
print(f"Unique seasons      : {total_seasons}")
print(f"Year range          : {year_start} - {year_end}")


# ------------------------------------------------------------
# 2. Overall numerical statistics
# ------------------------------------------------------------

print("\nKEY NUMERICAL STATISTICS")
print("-" * 40)

for column in [
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield"
]:
    print(
        f"{column:20s} "
        f"Mean: {df[column].mean():,.2f} | "
        f"Median: {df[column].median():,.2f} | "
        f"Std: {df[column].std():,.2f}"
    )


# ------------------------------------------------------------
# 3. Highest average-yield crop
# ------------------------------------------------------------

crop_yield_summary = (
    df.groupby("Crop")["Yield"]
      .mean()
      .sort_values(ascending=False)
)

best_yield_crop = crop_yield_summary.index[0]
best_yield_crop_value = crop_yield_summary.iloc[0]

print("\nCROP INSIGHT")
print("-" * 40)
print(
    f"Highest average-yield crop: "
    f"{best_yield_crop} ({best_yield_crop_value:,.2f})"
)


# ------------------------------------------------------------
# 4. Highest total-production crop
# ------------------------------------------------------------

crop_production_summary = (
    df.groupby("Crop")["Production"]
      .sum()
      .sort_values(ascending=False)
)

top_production_crop = crop_production_summary.index[0]
top_production_crop_value = crop_production_summary.iloc[0]

print(
    f"Highest total-production crop: "
    f"{top_production_crop} ({top_production_crop_value:,.2f})"
)


# ------------------------------------------------------------
# 5. Highest average-yield state
# ------------------------------------------------------------

state_yield_summary = (
    df.groupby("State")["Yield"]
      .mean()
      .sort_values(ascending=False)
)

best_yield_state = state_yield_summary.index[0]
best_yield_state_value = state_yield_summary.iloc[0]

print("\nSTATE INSIGHT")
print("-" * 40)
print(
    f"Highest average-yield state: "
    f"{best_yield_state} ({best_yield_state_value:,.2f})"
)


# ------------------------------------------------------------
# 6. Highest total-production state
# ------------------------------------------------------------

state_production_summary = (
    df.groupby("State")["Production"]
      .sum()
      .sort_values(ascending=False)
)

top_production_state = state_production_summary.index[0]
top_production_state_value = state_production_summary.iloc[0]

print(
    f"Highest total-production state: "
    f"{top_production_state} ({top_production_state_value:,.2f})"
)


# ------------------------------------------------------------
# 7. Best season by average yield
# ------------------------------------------------------------

season_yield_summary = (
    df.groupby("Season")["Yield"]
      .mean()
      .sort_values(ascending=False)
)

best_yield_season = season_yield_summary.index[0]
best_yield_season_value = season_yield_summary.iloc[0]

print("\nSEASON INSIGHT")
print("-" * 40)
print(
    f"Highest average-yield season: "
    f"{best_yield_season} ({best_yield_season_value:,.2f})"
)


# ------------------------------------------------------------
# 8. Highest total-production season
# ------------------------------------------------------------

season_production_summary = (
    df.groupby("Season")["Production"]
      .sum()
      .sort_values(ascending=False)
)

top_production_season = season_production_summary.index[0]
top_production_season_value = season_production_summary.iloc[0]

print(
    f"Highest total-production season: "
    f"{top_production_season} ({top_production_season_value:,.2f})"
)


# ------------------------------------------------------------
# 9. Correlation with Yield
# ------------------------------------------------------------

numerical_columns = [
    "Crop_Year",
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield"
]

correlation_with_yield = (
    df[numerical_columns]
    .corr()["Yield"]
    .drop("Yield")
    .sort_values(ascending=False)
)

print("\nCORRELATION WITH YIELD")
print("-" * 40)

for variable, correlation in correlation_with_yield.items():
    print(f"{variable:20s}: {correlation:.4f}")


# ------------------------------------------------------------
# 10. Strongest positive relationship
# ------------------------------------------------------------

strongest_variable = correlation_with_yield.index[0]
strongest_correlation = correlation_with_yield.iloc[0]

print(
    f"\nStrongest positive correlation with Yield: "
    f"{strongest_variable} ({strongest_correlation:.4f})"
)


# ------------------------------------------------------------
# 11. Outlier summary
# ------------------------------------------------------------

print("\nOUTLIER SUMMARY")
print("-" * 40)

outlier_columns = [
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield"
]

outlier_results = []

for column in outlier_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_count = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ).sum()

    outlier_percentage = (
        outlier_count / len(df)
    ) * 100

    outlier_results.append({
        "Variable": column,
        "Outlier_Count": outlier_count,
        "Outlier_Percentage": round(outlier_percentage, 2)
    })

outlier_summary = pd.DataFrame(outlier_results)

print(outlier_summary.to_string(index=False))


# ------------------------------------------------------------
# 12. Save final insights
# ------------------------------------------------------------

insights = []

insights.append(
    f"Dataset contains {total_records:,} records covering "
    f"{total_crops} crops, {total_states} states and "
    f"{total_seasons} seasons from {year_start} to {year_end}."
)

insights.append(
    f"{best_yield_crop} recorded the highest average yield "
    f"among crops ({best_yield_crop_value:,.2f})."
)

insights.append(
    f"{top_production_crop} recorded the highest total production "
    f"({top_production_crop_value:,.2f})."
)

insights.append(
    f"{best_yield_state} recorded the highest average yield "
    f"among states ({best_yield_state_value:,.2f})."
)

insights.append(
    f"{top_production_state} recorded the highest total production "
    f"among states ({top_production_state_value:,.2f})."
)

insights.append(
    f"{best_yield_season} recorded the highest average yield "
    f"among seasons ({best_yield_season_value:,.2f})."
)

insights.append(
    f"{top_production_season} recorded the highest total production "
    f"among seasons ({top_production_season_value:,.2f})."
)

insights.append(
    f"{strongest_variable} showed the strongest positive "
    f"correlation with Yield ({strongest_correlation:.4f})."
)

insights.append(
    "IQR-based outlier analysis identified extreme observations "
    "in several agricultural variables. These observations were "
    "retained because extreme values can represent genuine "
    "differences in crop area, production, rainfall or input usage."
)

with open(
    "../data/cleaned/final_eda_insights.txt",
    "w",
    encoding="utf-8"
) as file:

    for i, insight in enumerate(insights, start=1):
        file.write(f"{i}. {insight}\n")


# Save correlation results
correlation_with_yield.to_csv(
    "../data/cleaned/yield_correlations.csv",
    header=["Correlation"]
)

# Save outlier summary
outlier_summary.to_csv(
    "../data/cleaned/final_outlier_summary.csv",
    index=False
)


print("\nFinal EDA insights saved successfully.")
print("\nStep 13 completed successfully.")




















