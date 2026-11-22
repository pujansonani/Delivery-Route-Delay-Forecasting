import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

DATA_PATH = "data/processed/delivery_cleaned.csv"
FIG_DIR = "outputs/figures/eda"
REPORT_DIR = "outputs/reports"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
target = "Time_taken (min)"

print("\n" + "=" * 65)
print("STAGE 3: EXPLORATORY DATA ANALYSIS")
print("=" * 65)
print("Dataset shape:", df.shape)

# 1. Target distribution
plt.figure(figsize=(10, 5))
sns.histplot(df[target], kde=True, bins=30)
plt.title("Delivery Time Distribution")
plt.xlabel("Delivery Time (minutes)")
plt.savefig(f"{FIG_DIR}/01_delivery_time_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# 2. Traffic impact
if "Road_traffic_density" in df.columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="Road_traffic_density", y=target)
    plt.title("Traffic Density vs Delivery Time")
    plt.xticks(rotation=20)
    plt.savefig(f"{FIG_DIR}/02_traffic_impact.png", dpi=300, bbox_inches="tight")
    plt.close()

# 3. Weather impact
if "Weather_conditions" in df.columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="Weather_conditions", y=target)
    plt.title("Weather Conditions vs Delivery Time")
    plt.xticks(rotation=25)
    plt.savefig(f"{FIG_DIR}/03_weather_impact.png", dpi=300, bbox_inches="tight")
    plt.close()

# 4. Distance impact
distance_col = "distance_km"
if distance_col in df.columns:
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df, x=distance_col, y=target, alpha=0.35)
    sns.regplot(data=df, x=distance_col, y=target, scatter=False, color="red")
    plt.title("Delivery Distance vs Delivery Time")
    plt.savefig(f"{FIG_DIR}/04_distance_impact.png", dpi=300, bbox_inches="tight")
    plt.close()

# 5. Vehicle condition
if "Vehicle_condition" in df.columns:
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="Vehicle_condition", y=target)
    plt.title("Vehicle Condition vs Delivery Time")
    plt.savefig(f"{FIG_DIR}/05_vehicle_impact.png", dpi=300, bbox_inches="tight")
    plt.close()

# 6. City analysis
if "City" in df.columns:
    city_stats = df.groupby("City")[target].agg(
        ["count", "mean", "median", "std"]
    ).round(2)
    city_stats.to_csv(f"{REPORT_DIR}/city_delivery_analysis.csv")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="City", y=target, errorbar=None)
    plt.title("Average Delivery Time by City")
    plt.xticks(rotation=20)
    plt.savefig(f"{FIG_DIR}/06_city_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

# 7. Order time analysis
if "Order_Time_Period" in df.columns:
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="Order_Time_Period", y=target)
    plt.title("Order Time Period vs Delivery Duration")
    plt.savefig(f"{FIG_DIR}/07_order_period_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

# 8. Multiple deliveries
if "multiple_deliveries" in df.columns:
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="multiple_deliveries", y=target)
    plt.title("Multiple Deliveries vs Delivery Time")
    plt.savefig(f"{FIG_DIR}/08_multiple_deliveries.png", dpi=300, bbox_inches="tight")
    plt.close()

# 9. Correlation analysis
numeric_df = df.select_dtypes(include=np.number)
correlation = numeric_df.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.savefig(f"{FIG_DIR}/09_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

correlation[target].sort_values(ascending=False).to_csv(
    f"{REPORT_DIR}/target_correlations.csv"
)

# 10. Automated analytical report
with open(f"{REPORT_DIR}/eda_insights.txt", "w") as f:
    f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
    f.write("=" * 55 + "\n")
    f.write(f"Dataset shape: {df.shape}\n\n")
    f.write("Target variable statistics:\n")
    f.write(df[target].describe().to_string())
    f.write("\n\nNumerical feature correlations with target:\n")
    f.write(correlation[target].sort_values(ascending=False).to_string())

    for col in ["Road_traffic_density", "Weather_conditions", "City",
                "Vehicle_condition", "Order_Time_Period"]:
        if col in df.columns:
            f.write(f"\n\nAverage delivery duration by {col}:\n")
            f.write(df.groupby(col)[target].agg(["count", "mean", "median"]).round(2).to_string())

print("\nEDA COMPLETED")
print("Charts:", FIG_DIR)
print("Reports:", REPORT_DIR)
print("Generated charts:")
for file in sorted(os.listdir(FIG_DIR)):
    print("-", file)

print("\n" + "=" * 65)
print("STAGE 3 COMPLETED SUCCESSFULLY")
print("=" * 65)
