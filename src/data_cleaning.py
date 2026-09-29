import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FLIGHTS_PATH = PROJECT_ROOT / "data" / "raw" / "flights.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "flights_cleaned.csv"


# --------------------------------------------------
# 2. Load the flight dataset
# --------------------------------------------------

print("Loading flights dataset...")

flights = pd.read_csv(
    FLIGHTS_PATH,
    low_memory=False
)

print(f"Original shape: {flights.shape}")


# --------------------------------------------------
# 3. Remove cancelled flights
# --------------------------------------------------

print("\nRemoving cancelled flights...")

cancelled_count = flights["CANCELLED"].sum()

flights = flights[flights["CANCELLED"] == 0].copy()

print(f"Cancelled flights removed: {cancelled_count}")
print(f"Shape after removing cancelled flights: {flights.shape}")


# --------------------------------------------------
# 4. Remove diverted flights
# --------------------------------------------------

print("\nRemoving diverted flights...")

diverted_count = flights["DIVERTED"].sum()

flights = flights[flights["DIVERTED"] == 0].copy()

print(f"Diverted flights removed: {diverted_count}")
print(f"Shape after removing diverted flights: {flights.shape}")


# --------------------------------------------------
# 5. Remove rows with missing values in
#    important prediction variables
# --------------------------------------------------

print("\nChecking important missing values...")

important_columns = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "DEPARTURE_DELAY",
    "SCHEDULED_TIME",
    "DISTANCE"
]

before = len(flights)

flights = flights.dropna(subset=important_columns)

removed = before - len(flights)

print(f"Rows removed due to missing important values: {removed}")
print(f"Shape after cleaning: {flights.shape}")


# --------------------------------------------------
# 6. Remove unnecessary columns
# --------------------------------------------------

columns_to_remove = [
    "YEAR",
    "TAIL_NUMBER",
    "DEPARTURE_TIME",
    "TAXI_OUT",
    "WHEELS_OFF",
    "ELAPSED_TIME",
    "AIR_TIME",
    "WHEELS_ON",
    "TAXI_IN",
    "ARRIVAL_TIME",
    "ARRIVAL_DELAY",
    "CANCELLATION_REASON",
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY",
    "CANCELLED",
    "DIVERTED"
]

flights = flights.drop(columns=columns_to_remove)


# --------------------------------------------------
# 7. Save cleaned dataset
# --------------------------------------------------

print("\nSaving cleaned dataset...")

flights.to_csv(OUTPUT_PATH, index=False)

print(f"Cleaned dataset saved to:")
print(OUTPUT_PATH)

print("\nFinal shape:", flights.shape)

print("\nRemaining columns:")
print(flights.columns.tolist())