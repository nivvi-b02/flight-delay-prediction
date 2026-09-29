import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ==================================================
# 1. SET PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "flights_features.csv"
AIRLINES_PATH = PROJECT_ROOT / "data" / "raw" / "airlines.csv"
AIRPORTS_PATH = PROJECT_ROOT / "data" / "raw" / "airports.csv"

REPORTS_PATH = PROJECT_ROOT / "reports"

REPORTS_PATH.mkdir(exist_ok=True)


# ==================================================
# 2. LOAD DATA
# ==================================================

print("Loading flight data...")

flights = pd.read_csv(
    DATA_PATH,
    low_memory=False,
    dtype={
        "AIRLINE": str,
        "ORIGIN_AIRPORT": str,
        "DESTINATION_AIRPORT": str,
        "ROUTE": str
    }
)

airlines = pd.read_csv(AIRLINES_PATH)

airports = pd.read_csv(AIRPORTS_PATH)

print(f"Flights shape: {flights.shape}")
print(f"Airlines shape: {airlines.shape}")
print(f"Airports shape: {airports.shape}")


# ==================================================
# 3. BASIC TARGET OVERVIEW
# ==================================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

target_counts = flights["IS_DELAYED"].value_counts()

print(target_counts)

print("\nPercentage:")
print(
    flights["IS_DELAYED"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ==================================================
# 4. AIRLINE ANALYSIS
# ==================================================

print("\n" + "=" * 60)
print("AIRLINE DELAY ANALYSIS")
print("=" * 60)

# Rename airline lookup column before merging
airlines_lookup = airlines.rename(
    columns={"AIRLINE": "AIRLINE_NAME"}
)

# Add airline names
flights = flights.merge(
    airlines_lookup,
    left_on="AIRLINE",
    right_on="IATA_CODE",
    how="left"
)

airline_analysis = (
    flights
    .groupby(["AIRLINE", "AIRLINE_NAME"], dropna=False)
    .agg(
        total_flights=("IS_DELAYED", "count"),
        delayed_flights=("IS_DELAYED", "sum"),
        delay_rate=("IS_DELAYED", "mean")
    )
    .reset_index()
)

airline_analysis["delay_rate"] = (
    airline_analysis["delay_rate"] * 100
).round(2)

airline_analysis = airline_analysis.sort_values(
    "delay_rate",
    ascending=False
)

print("\nAirline delay analysis:")
print(airline_analysis.to_string(index=False))


# Save airline results
airline_analysis.to_csv(
    REPORTS_PATH / "airline_delay_analysis.csv",
    index=False
)


# --------------------------------------------------
# Airline delay rate chart
# --------------------------------------------------

plt.figure(figsize=(12, 7))

sns.barplot(
    data=airline_analysis,
    x="delay_rate",
    y="AIRLINE_NAME"
)

plt.title("Flight Delay Rate by Airline")
plt.xlabel("Delay Rate (%)")
plt.ylabel("Airline")
plt.tight_layout()

plt.savefig(
    REPORTS_PATH / "airline_delay_rate.png",
    dpi=300
)

plt.show()


# ==================================================
# 5. MONTHLY ANALYSIS
# ==================================================

print("\n" + "=" * 60)
print("MONTHLY DELAY ANALYSIS")
print("=" * 60)

monthly_analysis = (
    flights
    .groupby("MONTH")
    .agg(
        total_flights=("IS_DELAYED", "count"),
        delayed_flights=("IS_DELAYED", "sum"),
        delay_rate=("IS_DELAYED", "mean")
    )
    .reset_index()
)

monthly_analysis["delay_rate"] = (
    monthly_analysis["delay_rate"] * 100
).round(2)

print(monthly_analysis.to_string(index=False))

monthly_analysis.to_csv(
    REPORTS_PATH / "monthly_delay_analysis.csv",
    index=False
)


plt.figure(figsize=(10, 6))

sns.lineplot(
    data=monthly_analysis,
    x="MONTH",
    y="delay_rate",
    marker="o"
)

plt.title("Flight Delay Rate by Month")
plt.xlabel("Month")
plt.ylabel("Delay Rate (%)")
plt.xticks(range(1, 13))
plt.tight_layout()

plt.savefig(
    REPORTS_PATH / "monthly_delay_rate.png",
    dpi=300
)

plt.show()


# ==================================================
# 6. DAY OF WEEK ANALYSIS
# ==================================================

print("\n" + "=" * 60)
print("DAY OF WEEK ANALYSIS")
print("=" * 60)

day_analysis = (
    flights
    .groupby("DAY_OF_WEEK")
    .agg(
        total_flights=("IS_DELAYED", "count"),
        delayed_flights=("IS_DELAYED", "sum"),
        delay_rate=("IS_DELAYED", "mean")
    )
    .reset_index()
)

day_analysis["delay_rate"] = (
    day_analysis["delay_rate"] * 100
).round(2)

print(day_analysis.to_string(index=False))

day_analysis.to_csv(
    REPORTS_PATH / "day_of_week_delay_analysis.csv",
    index=False
)


plt.figure(figsize=(10, 6))

sns.barplot(
    data=day_analysis,
    x="DAY_OF_WEEK",
    y="delay_rate"
)

plt.title("Flight Delay Rate by Day of Week")
plt.xlabel("Day of Week (1 = Monday, 7 = Sunday)")
plt.ylabel("Delay Rate (%)")
plt.tight_layout()

plt.savefig(
    REPORTS_PATH / "day_of_week_delay_rate.png",
    dpi=300
)

plt.show()


# ==================================================
# 7. DEPARTURE HOUR ANALYSIS
# ==================================================

print("\n" + "=" * 60)
print("DEPARTURE HOUR ANALYSIS")
print("=" * 60)

hour_analysis = (
    flights
    .groupby("DEPARTURE_HOUR")
    .agg(
        total_flights=("IS_DELAYED", "count"),
        delayed_flights=("IS_DELAYED", "sum"),
        delay_rate=("IS_DELAYED", "mean")
    )
    .reset_index()
)

hour_analysis["delay_rate"] = (
    hour_analysis["delay_rate"] * 100
).round(2)

print(hour_analysis.to_string(index=False))

hour_analysis.to_csv(
    REPORTS_PATH / "departure_hour_delay_analysis.csv",
    index=False
)


plt.figure(figsize=(12, 6))

sns.lineplot(
    data=hour_analysis,
    x="DEPARTURE_HOUR",
    y="delay_rate",
    marker="o"
)

plt.title("Flight Delay Rate by Scheduled Departure Hour")
plt.xlabel("Departure Hour")
plt.ylabel("Delay Rate (%)")
plt.xticks(range(0, 24))
plt.tight_layout()

plt.savefig(
    REPORTS_PATH / "departure_hour_delay_rate.png",
    dpi=300
)

plt.show()


# ==================================================
# 8. TIME OF DAY ANALYSIS
# ==================================================

print("\n" + "=" * 60)
print("TIME OF DAY ANALYSIS")
print("=" * 60)

time_analysis = (
    flights
    .groupby("TIME_OF_DAY")
    .agg(
        total_flights=("IS_DELAYED", "count"),
        delayed_flights=("IS_DELAYED", "sum"),
        delay_rate=("IS_DELAYED", "mean")
    )
    .reset_index()
)

time_analysis["delay_rate"] = (
    time_analysis["delay_rate"] * 100
).round(2)

print(time_analysis.to_string(index=False))

time_analysis.to_csv(
    REPORTS_PATH / "time_of_day_delay_analysis.csv",
    index=False
)


plt.figure(figsize=(10, 6))

sns.barplot(
    data=time_analysis,
    x="TIME_OF_DAY",
    y="delay_rate"
)

plt.title("Flight Delay Rate by Time of Day")
plt.xlabel("Time of Day")
plt.ylabel("Delay Rate (%)")
plt.tight_layout()

plt.savefig(
    REPORTS_PATH / "time_of_day_delay_rate.png",
    dpi=300
)

plt.show()


# ==================================================
# 9. ORIGIN AIRPORT ANALYSIS
# ==================================================

print("\n" + "=" * 60)
print("ORIGIN AIRPORT DELAY ANALYSIS")
print("=" * 60)

origin_analysis = (
    flights
    .groupby("ORIGIN_AIRPORT")
    .agg(
        total_flights=("IS_DELAYED", "count"),
        delayed_flights=("IS_DELAYED", "sum"),
        delay_rate=("IS_DELAYED", "mean")
    )
    .reset_index()
)

# Only consider airports with at least 1,000 flights
origin_analysis = origin_analysis[
    origin_analysis["total_flights"] >= 1000
].copy()

origin_analysis["delay_rate"] = (
    origin_analysis["delay_rate"] * 100
).round(2)

origin_analysis = origin_analysis.sort_values(
    "delay_rate",
    ascending=False
)

# Add airport names
origin_analysis = origin_analysis.merge(
    airports[
        ["IATA_CODE", "AIRPORT", "CITY", "STATE"]
    ],
    left_on="ORIGIN_AIRPORT",
    right_on="IATA_CODE",
    how="left"
)

print("\nTop 15 origin airports by delay rate:")
print(
    origin_analysis.head(15).to_string(index=False)
)

origin_analysis.to_csv(
    REPORTS_PATH / "origin_airport_delay_analysis.csv",
    index=False
)


# ==================================================
# 10. DESTINATION AIRPORT ANALYSIS
# ==================================================

print("\n" + "=" * 60)
print("DESTINATION AIRPORT DELAY ANALYSIS")
print("=" * 60)

destination_analysis = (
    flights
    .groupby("DESTINATION_AIRPORT")
    .agg(
        total_flights=("IS_DELAYED", "count"),
        delayed_flights=("IS_DELAYED", "sum"),
        delay_rate=("IS_DELAYED", "mean")
    )
    .reset_index()
)

destination_analysis = destination_analysis[
    destination_analysis["total_flights"] >= 1000
].copy()

destination_analysis["delay_rate"] = (
    destination_analysis["delay_rate"] * 100
).round(2)

destination_analysis = destination_analysis.sort_values(
    "delay_rate",
    ascending=False
)

destination_analysis = destination_analysis.merge(
    airports[
        ["IATA_CODE", "AIRPORT", "CITY", "STATE"]
    ],
    left_on="DESTINATION_AIRPORT",
    right_on="IATA_CODE",
    how="left"
)

print("\nTop 15 destination airports by delay rate:")
print(
    destination_analysis.head(15).to_string(index=False)
)

destination_analysis.to_csv(
    REPORTS_PATH / "destination_airport_delay_analysis.csv",
    index=False
)


# ==================================================
# 11. ROUTE ANALYSIS
# ==================================================

print("\n" + "=" * 60)
print("ROUTE DELAY ANALYSIS")
print("=" * 60)

route_analysis = (
    flights
    .groupby("ROUTE")
    .agg(
        total_flights=("IS_DELAYED", "count"),
        delayed_flights=("IS_DELAYED", "sum"),
        delay_rate=("IS_DELAYED", "mean")
    )
    .reset_index()
)

# Only consider routes with at least 500 flights
route_analysis = route_analysis[
    route_analysis["total_flights"] >= 500
].copy()

route_analysis["delay_rate"] = (
    route_analysis["delay_rate"] * 100
).round(2)

route_analysis = route_analysis.sort_values(
    "delay_rate",
    ascending=False
)

print("\nTop 20 routes by delay rate:")
print(
    route_analysis.head(20).to_string(index=False)
)

route_analysis.to_csv(
    REPORTS_PATH / "route_delay_analysis.csv",
    index=False
)


# ==================================================
# 12. SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)

print(f"\nReports saved to:")
print(REPORTS_PATH)

print("\nGenerated analysis files:")

for file in sorted(REPORTS_PATH.glob("*.csv")):
    print(f" - {file.name}")

print("\nGenerated charts:")

for file in sorted(REPORTS_PATH.glob("*.png")):
    print(f" - {file.name}")