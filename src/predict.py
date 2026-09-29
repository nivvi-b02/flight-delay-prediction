import pandas as pd
import joblib

from pathlib import Path


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.joblib"


# ==================================================
# 2. LOAD MODEL
# ==================================================

print("=" * 60)
print("LOADING RANDOM FOREST MODEL")
print("=" * 60)

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ==================================================
# 3. PREDICTION FUNCTION
# ==================================================

def predict_flight_delay(
    month,
    day,
    day_of_week,
    airline,
    origin_airport,
    destination_airport,
    scheduled_departure,
    scheduled_time,
    distance,
    scheduled_arrival,
    departure_hour,
    time_of_day
):

    # Create a single-row DataFrame
    flight = pd.DataFrame([{
        "MONTH": month,
        "DAY": day,
        "DAY_OF_WEEK": day_of_week,
        "AIRLINE": airline,
        "ORIGIN_AIRPORT": origin_airport,
        "DESTINATION_AIRPORT": destination_airport,
        "SCHEDULED_DEPARTURE": scheduled_departure,
        "SCHEDULED_TIME": scheduled_time,
        "DISTANCE": distance,
        "SCHEDULED_ARRIVAL": scheduled_arrival,
        "DEPARTURE_HOUR": departure_hour,
        "TIME_OF_DAY": time_of_day
    }])

    # Prediction
    prediction = model.predict(flight)[0]

    # Probability of each class
    probabilities = model.predict_proba(flight)[0]

    delay_probability = probabilities[1]
    no_delay_probability = probabilities[0]

    # Convert prediction to readable result
    if prediction == 1:
        result = "DELAYED"
    else:
        result = "NOT DELAYED"

    return {
        "prediction": int(prediction),
        "result": result,
        "delay_probability": delay_probability,
        "no_delay_probability": no_delay_probability
    }


# ==================================================
# 4. TEST PREDICTION
# ==================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("TESTING PREDICTION")
    print("=" * 60)

    result = predict_flight_delay(
        month=8,
        day=15,
        day_of_week=6,
        airline="DL",
        origin_airport="ATL",
        destination_airport="JFK",
        scheduled_departure=1800,
        scheduled_time=150,
        distance=760,
        scheduled_arrival=2030,
        departure_hour=18,
        time_of_day="Evening"
    )

    print("\nPrediction Result:")
    print(f"Prediction: {result['result']}")
    print(
        f"Delay probability: "
        f"{result['delay_probability']:.2%}"
    )
    print(
        f"No-delay probability: "
        f"{result['no_delay_probability']:.2%}"
    )

    print("\n" + "=" * 60)
    print("PREDICTION TEST COMPLETE")
    print("=" * 60)