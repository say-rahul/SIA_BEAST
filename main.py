from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from typing import Optional
import uvicorn
import os
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import joblib

app = FastAPI()

# === Environment Variables ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
CITY = "Chengalpattu"
COUNTRY = "Tamil Nadu"

# === Config ===
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

# === Live Learning Setup ===
sgd_model = SGDClassifier(loss="log_loss")
scaler = StandardScaler()
X_buffer, y_buffer = [], []
sgd_initialized = False

# === Load Pretrained Models ===
try:
    rf_model = joblib.load("rf_water_decision.pkl")
    anomaly_model = joblib.load("isolation_forest.pkl")
    models_loaded = True
except Exception as e:
    print("⚠️ Warning: Failed to load models:", e)
    rf_model, anomaly_model = None, None
    models_loaded = False

# === State Variables ===
last_signal_time = datetime.now()
rain_detected_start = None
last_moisture = None

@app.get("/")
def root():
    return {"message": "🌿 Smart Irrigation API - Weather + ML + Threshold + Anomaly"}

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
    return {"message": "✅ Config updated", "config": CONFIG}

@app.post("/sensor-data")
def sensor_data(
    temperature: float = Form(...),
    humidity: float = Form(...),
    moisture: float = Form(...),
    ldr: float = Form(...),
    pressure: Optional[float] = Form(None),  # Optional BMP280
    rain: int = Form(...),
    flame: int = Form(...),
    watered: int = Form(...),
    pir: int = Form(...),
    ultrasonic: int = Form(...)
):
    global last_signal_time, rain_detected_start, last_moisture, sgd_initialized

    now = datetime.now()
    last_signal_time = now
    now_str = now.isoformat()

    # === Fire Alert ===
    if flame >= CONFIG["fire_alert_threshold"]:
        print(f"🔥 Fire detected at {now_str}")
        return JSONResponse({"alert": True, "message": "🔥 Fire detected!"})

    # === PIR Alerts ===
    if pir >= CONFIG["human_sensitivity"] and ultrasonic >= CONFIG["human_sensitivity"]:
        print(f"👤 Human detected at {now_str}")
    elif pir >= CONFIG["rat_sensitivity"] and ultrasonic == 0:
        print(f"🐀 Rat detected at {now_str}")

    # === Rain Monitoring ===
    if rain == 1:
        if rain_detected_start is None:
            rain_detected_start = now
        elif (now - rain_detected_start).total_seconds() >= CONFIG["rain_duration_threshold_sec"]:
            print("🌧️ Continuous rain detected")
            return JSONResponse({
                "should_water": 0,
                "reason": "🌧️ Rain ongoing",
                "moisture": moisture,
                "rain_expected": True,
                "anomaly": False,
                "model_active": sgd_initialized
            })
    else:
        rain_detected_start = None

    # === Manual Watering ===
    if CONFIG["manual_water"]:
        print("🚿 Manual watering triggered")
        return JSONResponse({
            "should_water": 1,
            "reason": "Manual watering",
            "target_ml": CONFIG["manual_target_ml"],
            "moisture": moisture,
            "rain_expected": False,
            "anomaly": False,
            "model_active": sgd_initialized
        })

    # === Danger Moisture ===
    if moisture < CONFIG["danger_threshold"]:
        print("⚠️ Danger: Critically low moisture!")
        return JSONResponse({
            "should_water": 1,
            "reason": "⚠️ Critical moisture level",
            "danger": True,
            "moisture": moisture,
            "rain_expected": False,
            "anomaly": False,
            "model_active": sgd_initialized
        })

    # === Weather Forecast ===
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
            print("🌦️ Weather check failed:", e)

    # === Feature Vector (no BMP280) ===
    features = np.array([[temperature, humidity, moisture, ldr, rain]])

    # === Anomaly Detection ===
    anomaly_flag = False
    try:
        if models_loaded:
            anomaly_score = anomaly_model.decision_function(features)[0]
            if anomaly_score < -0.2:
                anomaly_flag = True
                print("⚠️ Anomaly detected in sensor data")
    except Exception as e:
        print("Anomaly detection failed:", e)

    # === Live Learning (SGDClassifier) ===
    should_water = 0
    if not anomaly_flag:
        label = 1 if (moisture < CONFIG["moisture_threshold"] and not rain_expected) else 0
        X_buffer.append(features[0])
        y_buffer.append(label)

        if len(X_buffer) >= 10:
            X_np = np.array(X_buffer)
            y_np = np.array(y_buffer)
            X_scaled = scaler.fit_transform(X_np)
            if not sgd_initialized:
                sgd_model.partial_fit(X_scaled, y_np, classes=[0, 1])
                sgd_initialized = True
            else:
                sgd_model.partial_fit(X_scaled, y_np)
            X_buffer.clear()
            y_buffer.clear()

        if sgd_initialized:
            X_test_scaled = scaler.transform(features)
            live_pred = int(sgd_model.predict(X_test_scaled)[0])
            print(f"📊 Live SGD model decision: {live_pred}")

    # === Final Watering Decision ===
    reason = "Default logic"
    if anomaly_flag:
        reason = "⚠️ Anomaly detected - watering disabled"
        should_water = 0
    elif moisture < CONFIG["moisture_threshold"] and not rain_expected:
        if rf_model:
            try:
                rf_pred = int(rf_model.predict(features)[0])
                should_water = rf_pred
                reason = "🤖 ML model decision"
            except Exception as e:
                print("Random Forest failed:", e)
                should_water = 1
                reason = "Fallback: threshold decision"
        else:
            should_water = 1
            reason = "Threshold-based decision"
    else:
        should_water = 0
        reason = "No watering needed"

    # === Moisture Feedback After Watering ===
    if watered and last_moisture is not None:
        if moisture > last_moisture:
            print("✅ Moisture improved after watering")
        else:
            print("❌ Moisture did not improve")
    last_moisture = moisture

    return JSONResponse({
        "should_water": should_water,
        "moisture": moisture,
        "model_active": sgd_initialized,
        "rain_expected": rain_expected,
        "anomaly": anomaly_flag,
        "reason": reason
    })

# === Run the server ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)