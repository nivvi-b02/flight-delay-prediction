import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "flights_cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "flights_features.csv"


# --------------------------------------------------
# 2. Load cleaned dataset
# --------------------------------------------------

print("Loading cleaned dataset...")

flights = pd.read_csv(
    INPUT_PATH,
    low_memory=False,
    dtype={
        "ORIGIN_AIRPORT": str,
        "DESTINATION_AIRPORT": str
    }
)

print(f"Input shape: {flights.shape}")


# --------------------------------------------------
# 3. Create target variable
# --------------------------------------------------

print("\nCreating delay target...")

flights["IS_DELAYED"] = (
    flights["DEPARTURE_DELAY"] > 15
).astype(int)

print("\nTarget distribution:")
print(flights["IS_DELAYED"].value_counts())

print("\nTarget percentage:")
print(
    flights["IS_DELAYED"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# --------------------------------------------------
# 4. Convert scheduled departure into hour
# --------------------------------------------------

print("\nCreating departure hour...")

flights["DEPARTURE_HOUR"] = (
    flights["SCHEDULED_DEPARTURE"] // 100
)

print(
    flights[["SCHEDULED_DEPARTURE", "DEPARTURE_HOUR"]]
    .head()
)


# --------------------------------------------------
# 5. Create time-of-day feature
# --------------------------------------------------

print("\nCreating time-of-day feature...")

def get_time_period(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


flights["TIME_OF_DAY"] = flights["DEPARTURE_HOUR"].apply(
    get_time_period
)


# --------------------------------------------------
# 6. Create route feature
# --------------------------------------------------

print("\nCreating route feature...")

flights["ROUTE"] = (
    flights["ORIGIN_AIRPORT"].astype(str)
    + "_"
    + flights["DESTINATION_AIRPORT"].astype(str)
)


# --------------------------------------------------
# 7. Remove target leakage
# --------------------------------------------------

print("\nRemoving DEPARTURE_DELAY...")

flights = flights.drop(columns=["DEPARTURE_DELAY"])


# --------------------------------------------------
# 8. Save feature-engineered dataset
# --------------------------------------------------

print("\nSaving feature-engineered dataset...")

flights.to_csv(OUTPUT_PATH, index=False)

print(f"Saved to:")
print(OUTPUT_PATH)

print("\nFinal shape:", flights.shape)

print("\nFinal columns:")
print(flights.columns.tolist())