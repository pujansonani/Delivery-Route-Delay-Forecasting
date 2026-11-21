import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

RAW_PATH = "data/raw/zomato_delivery.csv"
OUTPUT_PATH = "data/processed/delivery_cleaned.csv"
REPORT_PATH = "outputs/reports/preprocessing_report.txt"

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

print("\n" + "=" * 65)
print("STAGE 2: DATA CLEANING & FEATURE ENGINEERING")
print("=" * 65)

df = pd.read_csv(RAW_PATH)
initial_rows = len(df)

print("\nOriginal dataset:", df.shape)

# 1. Standardize column names
df.columns = df.columns.str.strip()

# 2. Remove duplicate records
duplicates = df.duplicated().sum()
df = df.drop_duplicates().copy()

# 3. Clean text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace(
        ["", "NaN", "nan", "NULL", "null", "None"],
        pd.NA
    )

# 4. Convert numerical features
numeric_columns = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Vehicle_condition",
    "multiple_deliveries",
    "Time_taken (min)",
    "distance_km"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 5. Validate target and distance
target = "Time_taken (min)"
df = df.dropna(subset=[target])
df = df[df[target] > 0]

if "distance_km" in df.columns:
    df.loc[df["distance_km"] <= 0, "distance_km"] = np.nan

# 6. Handle invalid age and ratings
if "Delivery_person_Age" in df.columns:
    df.loc[
        ~df["Delivery_person_Age"].between(18, 70),
        "Delivery_person_Age"
    ] = np.nan

if "Delivery_person_Ratings" in df.columns:
    df.loc[
        ~df["Delivery_person_Ratings"].between(1, 5),
        "Delivery_person_Ratings"
    ] = np.nan

# 7. Impute numerical missing values
for col in [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "distance_km",
    "Vehicle_condition",
    "multiple_deliveries"
]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# 8. Handle categorical missing values
categorical_columns = df.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

for col in categorical_columns:
    df[col] = df[col].fillna("Unknown").astype(str)

# 9. Feature engineering: order date
if "Order_Date" in df.columns:
    order_date = pd.to_datetime(
        df["Order_Date"],
        errors="coerce",
        dayfirst=False
    )

    df["Order_Day"] = order_date.dt.day
    df["Order_Month"] = order_date.dt.month
    df["Order_Weekday"] = order_date.dt.dayofweek
    df["Is_Weekend"] = (order_date.dt.dayofweek >= 5).astype(int)

    df["Order_Day"] = df["Order_Day"].fillna(-1).astype(int)
    df["Order_Month"] = df["Order_Month"].fillna(-1).astype(int)
    df["Order_Weekday"] = df["Order_Weekday"].fillna(-1).astype(int)

# 10. Feature engineering: order time
if "Time_Orderd" in df.columns:
    parsed_time = pd.to_datetime(
        df["Time_Orderd"],
        format="%H:%M:%S",
        errors="coerce"
    )

    df["Order_Hour"] = parsed_time.dt.hour.fillna(-1).astype(int)
    df["Order_Minute"] = parsed_time.dt.minute.fillna(-1).astype(int)

    def get_time_period(hour):
        if hour == -1:
            return "Unknown"
        if 5 <= hour < 12:
            return "Morning"
        if 12 <= hour < 17:
            return "Afternoon"
        if 17 <= hour < 21:
            return "Evening"
        return "Night"

    df["Order_Time_Period"] = df["Order_Hour"].apply(get_time_period)

# 11. Feature engineering: delivery coordinates
if all(col in df.columns for col in [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude"
]):
    lat1 = np.radians(df["Restaurant_latitude"])
    lon1 = np.radians(df["Restaurant_longitude"])
    lat2 = np.radians(df["Delivery_location_latitude"])
    lon2 = np.radians(df["Delivery_location_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    df["Calculated_Distance_km"] = (
        6371 * 2 * np.arcsin(np.sqrt(a.clip(0, 1)))
    )

    df["Calculated_Distance_km"] = df[
        "Calculated_Distance_km"
    ].replace([np.inf, -np.inf], np.nan)

# 12. Remove identifiers and target-derived columns
leakage_columns = [
    "ID",
    "Delivery_person_ID",
    "delivery_speed",
    "Delivery_Duration_Category"
]

removed_columns = [col for col in leakage_columns if col in df.columns]

df = df.drop(columns=removed_columns, errors="ignore")

# 13. Final data validation
df = df.replace([np.inf, -np.inf], np.nan)

# Retain rows with a valid target
df = df.dropna(subset=[target])

# 14. Save processed dataset
df.to_csv(OUTPUT_PATH, index=False)

# 15. Generate report
with open(REPORT_PATH, "w") as report:
    report.write("DATA PREPROCESSING REPORT\n")
    report.write("=" * 55 + "\n")
    report.write(f"Original rows: {initial_rows}\n")
    report.write(f"Duplicates removed: {duplicates}\n")
    report.write(f"Final rows: {len(df)}\n")
    report.write(f"Final columns: {len(df.columns)}\n")
    report.write(f"Removed columns: {removed_columns}\n")
    report.write(f"Output: {OUTPUT_PATH}\n\n")
    report.write("Remaining missing values:\n")
    report.write(df.isnull().sum().to_string())
    report.write("\n\nFinal dataset information:\n")
    report.write(df.describe(include="all").to_string())

print("\nPREPROCESSING RESULTS")
print("-" * 50)
print("Original records:", initial_rows)
print("Duplicates removed:", duplicates)
print("Final dataset:", df.shape)
print("Removed columns:", removed_columns)
print("\nRemaining missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\nSaved:", OUTPUT_PATH)
print("Report:", REPORT_PATH)

print("\n" + "=" * 65)
print("STAGE 2 COMPLETED")
print("=" * 65)
