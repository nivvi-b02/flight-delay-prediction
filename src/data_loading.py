import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FLIGHTS_PATH = PROJECT_ROOT / "data" / "raw" / "flights.csv"
AIRLINES_PATH = PROJECT_ROOT / "data" / "raw" / "airlines.csv"
AIRPORTS_PATH = PROJECT_ROOT / "data" / "raw" / "airports.csv"


# --------------------------------------------------
# 2. Load the smaller datasets
# --------------------------------------------------

airlines = pd.read_csv(AIRLINES_PATH)
airports = pd.read_csv(AIRPORTS_PATH)


# --------------------------------------------------
# 3. Basic information about airlines
# --------------------------------------------------

print("\n" + "=" * 60)
print("AIRLINES DATASET")
print("=" * 60)

print("Shape:", airlines.shape)
print("\nColumns:")
print(airlines.columns.tolist())

print("\nFirst 5 rows:")
print(airlines.head())

print("\nMissing values:")
print(airlines.isnull().sum())


# --------------------------------------------------
# 4. Basic information about airports
# --------------------------------------------------

print("\n" + "=" * 60)
print("AIRPORTS DATASET")
print("=" * 60)

print("Shape:", airports.shape)
print("\nColumns:")
print(airports.columns.tolist())

print("\nFirst 5 rows:")
print(airports.head())

print("\nMissing values:")
print(airports.isnull().sum())


# --------------------------------------------------
# 5. Inspect flights dataset
# --------------------------------------------------

print("\n" + "=" * 60)
print("FLIGHTS DATASET")
print("=" * 60)

# Read only the first 5 rows for initial inspection
flights_sample = pd.read_csv(FLIGHTS_PATH, nrows=5)

print("Columns:")
print(flights_sample.columns.tolist())

print("\nFirst 5 rows:")
print(flights_sample)

print("\nData types:")
print(flights_sample.dtypes)


# --------------------------------------------------
# 6. Count total rows in flights.csv
# --------------------------------------------------

print("\nCounting rows in flights.csv...")

row_count = 0

with open(FLIGHTS_PATH, "r", encoding="utf-8") as file:
    row_count = sum(1 for _ in file) - 1

print("Total flight records:", row_count)


# --------------------------------------------------
# 7. Check missing values using chunks
# --------------------------------------------------

print("\nChecking missing values in flights.csv...")

missing_counts = None

for chunk in pd.read_csv(FLIGHTS_PATH, chunksize=100000):

    chunk_missing = chunk.isnull().sum()

    if missing_counts is None:
        missing_counts = chunk_missing
    else:
        missing_counts += chunk_missing


print("\nMissing values:")
print(missing_counts)


# --------------------------------------------------
# 8. Basic statistics for numerical columns
# --------------------------------------------------

print("\n" + "=" * 60)
print("BASIC STATISTICS")
print("=" * 60)

# Use a sample rather than loading all 6M+ rows
flights_sample_large = pd.read_csv(
    FLIGHTS_PATH,
    nrows=100000
)

print(flights_sample_large.describe().T)