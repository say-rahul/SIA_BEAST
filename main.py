# Hybrid Smart Irrigation API with Threshold Logic, Live Model Learning, and Weather Awareness (Flutter Compatible)

from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import uvicorn
import os
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Constants
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
CITY = "Chengalpattu"
COUNTRY = "Tamil Nadu"

CONFIG = {
    "moisture_threshold": 30.0,
    "danger_threshold": 10.0,
    "fire_alert_threshold": 1,
    "check_weather": True,
    "rain_duration_threshold_sec": 120,
    "rat_sensitivity": 1,
    "human_sensitivity": 1,
    "no_signal_threshold_sec": 1800,
    "manual_water": False,
    "manual_target_ml": 100
}

# Live model + scaler
model = SGDClassifier(loss="log_loss")
scaler = StandardScaler()
X_buffer, y_buffer = [], []
model_initialized = False
last_signal_time = datetime.now()
rain_detected_start = None
last_moisture = None

@app.get("/")
def root():
    return {"message": "🌿 Hybrid Smart Irrigation API - Threshold + Live Learning + Weather AI"}

@app.post("/config")
def update_config(
    moisture_threshold: float = Form(...),
    danger_threshold: float = Form(...),
    rat_sensitivity: int = Form(...),
    human_sensitivity: int = Form(...),
    no_signal_threshold_sec: int = Form(...),
    manual_water: bool = Form(...),
    manual_target_ml: int = Form(...)
):
    CONFIG.update({
        "moisture_threshold": moisture_threshold,
        "danger_threshold": danger_threshold,
        "rat_sensitivity": rat_sensitivity,
        "human_sensitivity": human_sensitivity,
        "no_signal_threshold_sec": no_signal_threshold_sec,
        "manual_water": manual_water,
        "manual_target_ml": manual_target_ml
    })
    return {"message": "Config updated", "config": CONFIG}

@app.post("/sensor-data")
def sensor_data(
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
    global last_signal_time, rain_detected_start, last_moisture, model_initialized

    now = datetime.now()
    last_signal_time = now
    now_str = now.isoformat()

    # Fire alert
    if flame >= CONFIG["fire_alert_threshold"]:
        print(f"🔥 Fire detected at {now_str}")
        return JSONResponse({"alert": True, "message": "🔥 Fire detected!"})

    # PIR alerts
    if pir >= CONFIG["human_sensitivity"] and ultrasonic >= CONFIG["human_sensitivity"]:
        print(f"👤 Human detected at {now_str}")
    elif pir >= CONFIG["rat_sensitivity"] and ultrasonic == 0:
        print(f"🐀 Rat detected at {now_str}")

    # Rain duration check
    if rain == 1:
        if rain_detected_start is None:
            rain_detected_start = now
        elif (now - rain_detected_start).total_seconds() >= CONFIG["rain_duration_threshold_sec"]:
            print("🌧️ Continuous rain detected")
            return JSONResponse({"should_water": 0, "reason": "🌧️ Rain ongoing"})
    else:
        rain_detected_start = None

    # Manual watering
    if CONFIG["manual_water"]:
        print("🚿 Manual watering triggered")
        return JSONResponse({"should_water": 1, "reason": "Manual watering", "target_ml": CONFIG["manual_target_ml"]})

    # Danger zone check
    if moisture < CONFIG["danger_threshold"]:
        print("⚠️ Danger: Moisture critically low, watering now!")
        return JSONResponse({"should_water": 1, "reason": "⚠️ Critical moisture level", "danger": True})

    # Weather forecast check
    rain_expected = False
    if CONFIG["check_weather"]:
        try:
            url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY},{COUNTRY}&appid={WEATHER_API_KEY}&units=metric"
            data = requests.get(url).json()
            today, tomorrow = now.date(), now.date() + timedelta(days=1)
            for entry in data.get("list", []):
                dt_txt = entry.get("dt_txt", "")
                if dt_txt:
                    date = datetime.strptime(dt_txt, "%Y-%m-%d %H:%M:%S").date()
                    if date in [today, tomorrow] and "rain" in entry["weather"][0]["main"].lower():
                        rain_expected = True
                        break
        except Exception as e:
            print("Weather check failed", e)

    # Logic: combine moisture threshold and weather
    should_water = 0
    if moisture < CONFIG["moisture_threshold"] and not rain_expected:
        should_water = 1

    # Learn from data
    features = np.array([[temperature, humidity, moisture, ldr, pressure, rain]])
    X_buffer.append(features[0])
    y_buffer.append(should_water)

    if len(X_buffer) >= 10:
        X_np = np.array(X_buffer)
        y_np = np.array(y_buffer)
        X_scaled = scaler.fit_transform(X_np)
        if not model_initialized:
            model.partial_fit(X_scaled, y_np, classes=[0, 1])
            model_initialized = True
        else:
            model.partial_fit(X_scaled, y_np)
        X_buffer.clear()
        y_buffer.clear()

    if model_initialized:
        X_test_scaled = scaler.transform(features)
        model_pred = int(model.predict(X_test_scaled)[0])
        print(f"🤖 Model suggests watering: {model_pred}")

    # After watering check
    if watered:
        if last_moisture is not None:
            if moisture > last_moisture:
                print("✅ Moisture improved after watering")
            else:
                print("❌ Moisture did not improve after watering")
        last_moisture = moisture

    return JSONResponse({
        "should_water": should_water,
        "moisture": moisture,
        "model_active": model_initialized,
        "rain_expected": rain_expected
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
