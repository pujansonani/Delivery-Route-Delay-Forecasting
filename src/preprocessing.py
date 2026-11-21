import os
import pandas as pd
import numpy as np

RAW_PATH = "data/raw/zomato_delivery.csv"
PROCESSED_PATH = "data/processed/delivery_cleaned.csv"

def load_data():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")
    return pd.read_csv(RAW_PATH)

def clean_data(df):
    df = df.copy()

    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()

    numeric_cols = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "distance_km",
        "delivery_speed",
        "Time_taken (min)"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Delivery_person_Age"] = df["Delivery_person_Age"].fillna(
        df["Delivery_person_Age"].median()
    )

    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].fillna(
        df["Delivery_person_Ratings"].median()
    )

    df["Time_Orderd"] = df["Time_Orderd"].fillna("Unknown")

    categorical_cols = df.select_dtypes(include="object").columns
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=["Time_taken (min)"])
    df = df[df["Time_taken (min)"] > 0]
    df = df[df["distance_km"] >= 0]

    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"], errors="coerce", dayfirst=True
    )

    df["Order_Day"] = df["Order_Date"].dt.day
    df["Order_Month"] = df["Order_Date"].dt.month
    df["Order_Weekday"] = df["Order_Date"].dt.dayofweek

    df["Distance_Category"] = pd.cut(
        df["distance_km"],
        bins=[-0.01, 3, 7, 12, np.inf],
        labels=["Short", "Medium", "Long", "Very Long"]
    )

    df["Delivery_Duration_Category"] = pd.cut(
        df["Time_taken (min)"],
        bins=[0, 25, 40, np.inf],
        labels=["Fast", "Moderate", "Slow"]
    )

    return df

def main():
    df = load_data()
    cleaned = clean_data(df)

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    cleaned.to_csv(PROCESSED_PATH, index=False)

    print("\n========== PREPROCESSING REPORT ==========")
    print("Original shape:", df.shape)
    print("Cleaned shape:", cleaned.shape)
    print("Duplicates remaining:", cleaned.duplicated().sum())
    print("\nMissing values:")
    print(cleaned.isnull().sum())
    print("\nTarget statistics:")
    print(cleaned["Time_taken (min)"].describe())
    print("\nSaved to:", PROCESSED_PATH)

if __name__ == "__main__":
    main()
