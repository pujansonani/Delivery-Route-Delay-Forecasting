import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

RAW_PATH = "data/raw/zomato_delivery.csv"
OUTPUT_DIR = "outputs"
FIGURE_DIR = f"{OUTPUT_DIR}/figures"
REPORT_DIR = f"{OUTPUT_DIR}/reports"

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("\n" + "=" * 65)
print("DELIVERY ROUTE DELAY FORECASTING")
print("STAGE 1: AUTOMATED DATA INSPECTION")
print("=" * 65)

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {RAW_PATH}\n"
        "Check that the raw dataset exists in data/raw/."
    )

df = pd.read_csv(RAW_PATH)

print("\n[1] DATASET OVERVIEW")
print("-" * 45)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Memory usage:", round(df.memory_usage(deep=True).sum() / 1024**2, 2), "MB")

print("\n[2] COLUMN INFORMATION")
print("-" * 45)
df.info()

print("\n[3] MISSING VALUES")
print("-" * 45)
missing = df.isnull().sum().to_frame("Missing_Count")
missing["Missing_Percentage"] = (missing["Missing_Count"] / len(df) * 100).round(2)
missing = missing.sort_values("Missing_Count", ascending=False)
print(missing[missing["Missing_Count"] > 0].to_string())

print("\n[4] DUPLICATE RECORDS")
print("-" * 45)
print("Duplicate rows:", df.duplicated().sum())

print("\n[5] NUMERICAL SUMMARY")
print("-" * 45)
print(df.describe(include="number").T.to_string())

print("\n[6] CATEGORICAL SUMMARY")
print("-" * 45)
categorical = df.select_dtypes(include=["object", "category"])
for col in categorical.columns:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).head(10).to_string())

print("\n[7] TARGET VARIABLE ANALYSIS")
print("-" * 45)

target = "Time_taken (min)"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found.")

print(df[target].describe().to_string())

plt.figure(figsize=(10, 5))
sns.histplot(df[target].dropna(), bins=30, kde=True)
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Number of Deliveries")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/delivery_time_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 4))
sns.boxplot(x=df[target])
plt.title("Delivery Time Outlier Inspection")
plt.xlabel("Delivery Time (minutes)")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/delivery_time_boxplot.png", dpi=300)
plt.close()

print("\n[8] CORRELATION ANALYSIS")
print("-" * 45)

numeric_df = df.select_dtypes(include=np.number)
correlation = numeric_df.corr()

correlation.to_csv(f"{REPORT_DIR}/correlation_matrix.csv")

plt.figure(figsize=(14, 10))
sns.heatmap(correlation, cmap="coolwarm", center=0, annot=True, fmt=".2f")
plt.title("Numerical Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/correlation_heatmap.png", dpi=300)
plt.close()

if "Road_traffic_density" in df.columns:
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="Road_traffic_density", y=target)
    plt.title("Impact of Road Traffic on Delivery Time")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/traffic_vs_delivery_time.png", dpi=300)
    plt.close()

if "Weather_conditions" in df.columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="Weather_conditions", y=target)
    plt.title("Impact of Weather on Delivery Time")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/weather_vs_delivery_time.png", dpi=300)
    plt.close()

if "City" in df.columns:
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="City", y=target)
    plt.title("Delivery Time Across Cities")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/city_vs_delivery_time.png", dpi=300)
    plt.close()

print("\n[9] FEATURE LEAKAGE REVIEW")
print("-" * 45)
print("Target:", target)
print("Potential leakage columns to investigate:")
for col in ["delivery_speed", "Delivery_Duration_Category"]:
    if col in df.columns:
        print("-", col, "(do not use as a predictor without verifying its calculation)")

print("\n[10] SAVING REPORTS")
print("-" * 45)

missing.to_csv(f"{REPORT_DIR}/missing_values_report.csv")
df.describe(include="all").T.to_csv(f"{REPORT_DIR}/dataset_summary.csv")
df.head(100).to_csv(f"{REPORT_DIR}/sample_records.csv", index=False)

with open(f"{REPORT_DIR}/inspection_summary.txt", "w") as f:
    f.write("DELIVERY ROUTE DELAY FORECASTING - DATA INSPECTION\n")
    f.write("=" * 55 + "\n")
    f.write(f"Dataset shape: {df.shape}\n")
    f.write(f"Duplicate rows: {df.duplicated().sum()}\n")
    f.write(f"Target: {target}\n\n")
    f.write("Missing values:\n")
    f.write(missing.to_string())
    f.write("\n\nTarget summary:\n")
    f.write(df[target].describe().to_string())

print("Reports saved in:", REPORT_DIR)
print("Charts saved in:", FIGURE_DIR)

print("\n" + "=" * 65)
print("STAGE 1 COMPLETED SUCCESSFULLY")
print("=" * 65)
