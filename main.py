# LSTM-based Smart Irrigation API with Supabase Integration and Extended Alert System (Flutter Compatible)
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import uvicorn
import os
import numpy as np
import requests
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
from supabase import create_client, Client
import joblib

app = FastAPI()

# Constants
SUPABASE_URL = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFmYW1kcnd5cXVtYnBxYW9rZGFuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM4NDAyODMsImV4cCI6MjA1OTQxNjI4M30.odsYjrOBcdiq4gPSOVm-8DpBsjLKR24yIvTaPktNo4o"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFmYW1kcnd5cXVtYnBxYW9rZGFuIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0Mzg0MDI4MywiZXhwIjoyMDU5NDE2MjgzfQ.Ty8ORU2qfTEZxm-pupbmWEToi1yW56ckD_Ha4ZGxWs8"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

LSTM_MODEL_FILE = "lstm_should_water.h5"
SEQUENCE_LENGTH = 6
FEATURES = ['temperature', 'humidity', 'moisture', 'ldr', 'pressure', 'rain']
WEATHER_API_KEY = "3370b2a0cc1a652422e836ec82daf9b2"
CITY = "Chengalpattu"
COUNTRY = "Tamil Nadu"

CONFIG = {
    "moisture_threshold": 30.0,
    "fire_alert_threshold": 1,
    "check_weather": True,
    "rain_duration_threshold_sec": 120,
    "rat_sensitivity": 1,
    "human_sensitivity": 1,
    "no_signal_threshold_sec": 1800,
    "manual_water": False,
    "manual_target_ml": 100
}

lstm_model = load_model(LSTM_MODEL_FILE)
scaler = joblib.load("feature_scaler.pkl")
sensor_sequence_buffer = []
rain_detected_start = None
last_signal_time = datetime.now()

@app.get("/")
def root():
    return {"message": "🌿 Smart Irrigation API with Supabase, LSTM, and Alerts - Flutter Ready"}

@app.post("/config")
def update_config(
    moisture_threshold: float = Form(...),
    rat_sensitivity: int = Form(...),
    human_sensitivity: int = Form(...),
    no_signal_threshold_sec: int = Form(...),
    manual_water: bool = Form(...),
    manual_target_ml: int = Form(...)
):
    CONFIG.update({
        "moisture_threshold": moisture_threshold,
        "rat_sensitivity": rat_sensitivity,
        "human_sensitivity": human_sensitivity,
        "no_signal_threshold_sec": no_signal_threshold_sec,
        "manual_water": manual_water,
        "manual_target_ml": manual_target_ml
    })
    return {"message": "Config updated", "config": CONFIG}

@app.post("/connectivity-alert")
def handle_wifi_failure():
    supabase.table("wifi_failures").insert({"timestamp": datetime.now().isoformat()}).execute()
    return JSONResponse(content={"alert": True, "message": "📡 Field unit WiFi connection failed."})

@app.get("/check-signal")
def check_for_missing_signals():
    if (datetime.now() - last_signal_time).total_seconds() > CONFIG["no_signal_threshold_sec"]:
        supabase.table("no_signal_alerts").insert({"timestamp": datetime.now().isoformat()}).execute()
        return {"alert": True, "message": "🚫 No signal from field unit in the past configured duration."}
    return {"alert": False, "message": "✅ Signal is up to date."}

@app.post("/sensor-data")
def process_sensor_data(
    temperature: float = Form(...),
    humidity: float = Form(...),
    moisture: float = Form(...),
    ldr: float = Form(...),
    pressure: float = Form(...),
    rain: int = Form(...),
    flame: int = Form(...),
    watered: int = Form(...),
    pir: int = Form(...),
    ultrasonic: int = Form(...)
):
    global rain_detected_start, last_signal_time
    last_signal_time = datetime.now()
    now_str = last_signal_time.isoformat()

    supabase.table("sensor_readings").insert({
        "timestamp": now_str,
        "temperature": temperature,
        "humidity": humidity,
        "moisture": moisture,
        "ldr": ldr,
        "pressure": pressure,
        "rain": rain
    }).execute()

    if flame >= CONFIG["fire_alert_threshold"]:
        supabase.table("fire_alerts").insert({"timestamp": now_str, "value": flame}).execute()
        return JSONResponse(content={"alert": True, "message": "🔥 Fire detected!"})

    if pir >= CONFIG["human_sensitivity"] and ultrasonic >= CONFIG["human_sensitivity"]:
        supabase.table("human_detection").insert({"timestamp": now_str}).execute()
        return JSONResponse(content={"alert": True, "message": "👤 Human detected in the farm!"})

    if pir >= CONFIG["rat_sensitivity"] and ultrasonic == 0:
        supabase.table("rat_detection").insert({"timestamp": now_str}).execute()

    if rain == 1:
        if rain_detected_start is None:
            rain_detected_start = datetime.now()
        elif (datetime.now() - rain_detected_start).total_seconds() >= CONFIG["rain_duration_threshold_sec"]:
            supabase.table("rain_alerts").insert({"timestamp": now_str}).execute()
            return JSONResponse(content={"alert": True, "message": "🌧️ Continuous rain detected for 2+ minutes."})
    else:
        rain_detected_start = None

    if CONFIG["check_weather"]:
        try:
            url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY},{COUNTRY}&appid={WEATHER_API_KEY}&units=metric"
            response = requests.get(url)
            data = response.json()
            today = datetime.now().date()
            tomorrow = today + timedelta(days=1)
            rain_expected, cyclone_alert = False, False
            for entry in data.get("list", []):
                dt_txt = entry.get("dt_txt", "")
                if dt_txt:
                    forecast_date = datetime.strptime(dt_txt, "%Y-%m-%d %H:%M:%S").date()
                    if forecast_date in [today, tomorrow]:
                        main = entry["weather"][0]["main"].lower()
                        if "rain" in main:
                            rain_expected = True
                        if "cyclone" in main or "storm" in main:
                            cyclone_alert = True
            if cyclone_alert:
                supabase.table("weather_alerts").insert({"timestamp": now_str, "alert": "cyclone"}).execute()
                return JSONResponse(content={"should_water": 0, "reason": "⛈️ Cyclone alert active"})
            if rain_expected:
                supabase.table("weather_alerts").insert({"timestamp": now_str, "alert": "rain"}).execute()
                return JSONResponse(content={"should_water": 0, "reason": "🌧️ Rain expected, skipping watering"})
        except Exception as e:
            print("Weather API error:", e)

    if CONFIG["manual_water"]:
        supabase.table("watering_logs").insert({"timestamp": now_str, "volume_ml": CONFIG["manual_target_ml"]}).execute()
        return JSONResponse(content={"should_water": 1, "reason": "🚿 Manual watering triggered by user", "target_ml": CONFIG["manual_target_ml"]})

    sensor_sequence_buffer.append([temperature, humidity, moisture, ldr, pressure, rain])
    if len(sensor_sequence_buffer) < SEQUENCE_LENGTH:
        return JSONResponse(content={"message": "📊 Gathering sensor data for model..."})
    elif len(sensor_sequence_buffer) > SEQUENCE_LENGTH:
        sensor_sequence_buffer.pop(0)

    input_seq = scaler.transform(sensor_sequence_buffer)
    input_seq = np.array(input_seq).reshape(1, SEQUENCE_LENGTH, len(FEATURES))
    prediction = lstm_model.predict(input_seq, verbose=0)
    should_water = int(prediction[0][0] > 0.5 and moisture < CONFIG["moisture_threshold"])

    supabase.table("watering_decisions").insert({
        "timestamp": now_str,
        "should_water": should_water,
        "confidence": float(prediction[0][0]),
        "moisture": moisture,
        "model_result": float(prediction[0][0])
    }).execute()

    if watered:
        supabase.table("watering_logs").insert({"timestamp": now_str, "volume_ml": 100}).execute()

    return JSONResponse(content={
        "should_water": should_water,
        "prediction_confidence": float(prediction[0][0]),
        "moisture": float(moisture),
        "message": "✅ Decision made using LSTM + Supabase logging"
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
