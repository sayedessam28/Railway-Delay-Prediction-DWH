import pandas as pd
import numpy as np
import datetime
import joblib
from random import choice, randint

# ======================================================
# 1) Generate Synthetic Future Data
# ======================================================

def generate_synthetic_data(num_rows=500) -> pd.DataFrame:
    """Creates a DataFrame with features required by the prediction model."""
    payment_methods = ['Credit Card', 'Debit Card', 'Contactless']
    ticket_types = ['Advance', 'Off-Peak', 'Anytime']
    railcards = ['None', 'Adult', 'Senior']
    ticket_classes = ['Standard', 'First Class']
    stations = ['London Paddington', 'Birmingham New Street', 'Manchester Piccadilly', 'Liverpool Lime Street', 'Leeds', 'York', 'Bristol Temple Meads', 'Reading']
    
    today = datetime.date.today()
    future_dates_journey = [today + datetime.timedelta(days=i) for i in range(1, 31)]
    future_dates_purchase = [today - datetime.timedelta(days=i) for i in range(1, 15)]

    data = {
        "Transaction ID": [f"T{i:04d}" for i in range(num_rows)],
        "Date of Purchase": [choice(future_dates_purchase) for _ in range(num_rows)],
        "Date of Journey": [choice(future_dates_journey) for _ in range(num_rows)],
        "Departure Time": [f"{randint(6, 22):02d}:{randint(0, 59):02d}:00" for _ in range(num_rows)],
        "Arrival Time": [f"{randint(8, 23):02d}:{randint(0, 59):02d}:00" for _ in range(num_rows)],
        "Payment Method": [choice(payment_methods) for _ in range(num_rows)],
        "Ticket Type": [choice(ticket_types) for _ in range(num_rows)],
        "Railcard": [choice(railcards) for _ in range(num_rows)],
        "Ticket Class": [choice(ticket_classes) for _ in range(num_rows)],
        "Departure Station": [choice(stations) for _ in range(num_rows)],
        "Arrival Destination": [choice(stations) for _ in range(num_rows)],
        # إضافة أعمدة وصفية لبيانات الرحلة التي قد تكون مطلوبة في Power BI
        "Purchase Type": [choice(['Online', 'Station']) for _ in range(num_rows)],
        "Time of Purchase": [f"{randint(0, 23):02d}:{randint(0, 59):02d}:00" for _ in range(num_rows)],
    }
    
    df = pd.DataFrame(data)
    df = df[df["Departure Station"] != df["Arrival Destination"]]
    return df.reset_index(drop=True)

df_new = generate_synthetic_data(num_rows=500)
print(f"Successfully generated {len(df_new)} rows of synthetic data.")

# ======================================================
# 2) Load Model and Define Constants
# ======================================================

MODEL_FILE = "railway_delay_model_final_balanced.pkl" 
OUTPUT_FILE = "synthetic_future_predictions.csv"

try:
    model = joblib.load(MODEL_FILE)
    print(f"Model '{MODEL_FILE}' loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_FILE}' not found. Please ensure the model file is in the same directory.")
    exit()

# ======================================================
# 3) Preprocessing & Feature Engineering (Identical to Training)
# ======================================================

def time_to_minutes(t):
    """Converts time string (HH:MM:SS) or (HH:MM) to total minutes from midnight."""
    if pd.isna(t):
        return np.nan
    try:
        t = str(t).strip().replace(".", ":")
        parts = t.split(":")
        h, m = 0, 0
        if len(parts) >= 2:
            h, m = int(parts[0]), int(parts[1])
        return h * 60 + m
    except ValueError:
        return np.nan

df_new["Date of Purchase"] = pd.to_datetime(df_new["Date of Purchase"])
df_new["Date of Journey"] = pd.to_datetime(df_new["Date of Journey"])

df_new["Purchase_Day"] = df_new["Date of Purchase"].dt.day
df_new["Purchase_Month"] = df_new["Date of Purchase"].dt.month
df_new["Journey_Day"] = df_new["Date of Journey"].dt.day
df_new["Journey_Month"] = df_new["Date of Journey"].dt.month

df_new["Departure_Min"] = df_new["Departure Time"].apply(time_to_minutes)
df_new["Arrival_Min"] = df_new["Arrival Time"].apply(time_to_minutes)

X_inference = df_new[[
    "Purchase_Day", "Purchase_Month",
    "Journey_Day", "Journey_Month",
    "Departure_Min", "Arrival_Min",
    "Payment Method", "Ticket Type",
    "Departure Station", "Arrival Destination",
    "Railcard", "Ticket Class" 
]].copy()

print("Preprocessing complete for inference.")

# ======================================================
# 4) Prediction / Inference
# ======================================================

df_new["Predicted_Delayed"] = model.predict(X_inference)
df_new["Prediction_Probability"] = model.predict_proba(X_inference)[:, 1].round(4)

print("Prediction complete. Results added to the dataframe.")

# ======================================================
# 5) Export Results for Power BI
# ======================================================

columns_to_keep = [
    'Transaction ID', 'Date of Purchase', 'Time of Purchase', 'Purchase Type', 
    'Date of Journey', 'Departure Time', 'Arrival Time', 
    'Payment Method', 'Railcard', 'Ticket Class', 'Ticket Type', 
    'Departure Station', 'Arrival Destination', 
    'Predicted_Delayed', 'Prediction_Probability'
]

df_new[columns_to_keep].to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

print(f"\n--- SUCCESS ---")
print(f"Synthetic data with predictions exported to '{OUTPUT_FILE}'.")