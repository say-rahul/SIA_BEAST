from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from typing import Optional
from supabase import create_client, Client
from datetime import datetime, timedelta
import uvicorn
import numpy as np
import os
import joblib
import requests
from typing import Union
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# === Telegram Notifier ===
def send_telegram_message(text: str):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("❌ Telegram error:", e)

# === Supabase Connection ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === Weather Settings ===
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
CITY = "Chengalpattu"
COUNTRY = "Tamil Nadu"

# === Device Metadata ===
ESP32_ID = "sector-a-01"
ZONE = "Sector A"

# === Config ===
CONFIG = {
    "moisture_threshold": 30.0,
    "danger_threshold": 20.0,
    "fire_alert_threshold": 1,
    "check_weather": True,
    "rain_duration_threshold_sec": 120,
    "rat_sensitivity": 1,
    "human_sensitivity": 1,
    "no_signal_threshold_sec": 1800,
    "manual_water": False,
    "manual_target_ml": 100
}

# === Models ===
sgd_model = SGDClassifier(loss="log_loss")
scaler = StandardScaler()
X_buffer, y_buffer = [], []
sgd_initialized = False

try:
    rf_model = joblib.load("rf_water_decision.pkl")
    anomaly_model = joblib.load("isolation_forest.pkl")
    models_loaded = True
except Exception as e:
    print("⚠️ Model loading failed:", e)
    send_telegram_message(f"⚠️ Model loading failed: {e}")
    rf_model = anomaly_model = None
    models_loaded = False

# === State Tracking ===
last_signal_time = datetime.now()
rain_detected_start = None
last_moisture = None

@app.post("/config")
def update_config(
    moisture_threshold: float = Form(...),
    danger_threshold: float = Form(...),
    rat_sensitivity: int = Form(...),
    human_sensitivity: int = Form(...),
    no_signal_threshold_sec: int = Form(...),
    manual_water: bool = Form(...),
    manual_target_ml: int = Form(...),
    enable_pir: bool = Form(...),
    enable_ultrasonic: bool = Form(...)
):
    CONFIG.update({
        "moisture_threshold": moisture_threshold,
        "danger_threshold": danger_threshold,
        "rat_sensitivity": rat_sensitivity,
        "human_sensitivity": human_sensitivity,
        "no_signal_threshold_sec": no_signal_threshold_sec,
        "manual_water": manual_water,
        "manual_target_ml": manual_target_ml,
        "enable_pir": enable_pir,
        "enable_ultrasonic": enable_ultrasonic
    })
    supabase.table("config_history").insert({
        "timestamp": datetime.now().isoformat(),
        "config": CONFIG,
        "updated_by": "server",
        "esp32_id": ESP32_ID,
        "zone": ZONE
    }).execute()
    return {"message": "✅ Config updated", "config": CONFIG}

@app.post("/log_command")
async def log_command(command: str = Form(...), source: str = Form(...), status: str = Form(...)):
    timestamp = datetime.utcnow().isoformat()
    supabase.table("command_logs").insert({
        "command": command,
        "source": source,
        "timestamp": timestamp,
        "status": status
    }).execute()
    return {"message": "Command logged"}

@app.post("/sensor-data")
def sensor_data(
    temperature: float = Form(...),
    humidity: float = Form(...),
    moisture: float = Form(...),
    ldr: float = Form(...),
    pressure: Union[str, None] = Form(None),
    rain: int = Form(...),
    flame: int = Form(...),
    watered: int = Form(...),
    pir: int = Form(...),
    ultrasonic: int = Form(...)
):
    global last_signal_time, rain_detected_start, last_moisture, sgd_initialized

    now = datetime.now()

    # ✅ FIXED INDENTATION HERE
    try:
        pressure_val = float(pressure) if pressure not in [None, ""] else None
    except ValueError:
        pressure_val = None


    # Log sensor readings
    supabase.table("sensor_readings").insert({
        "timestamp": now.isoformat(),
        "temperature": temperature,
        "humidity": humidity,
        "moisture": moisture,
        "ldr": ldr,
        "pressure": pressure_val,
        "rain": rain,
        "flame": flame,
        "pir": pir,
        "ultrasonic": ultrasonic,
        "esp32_id": ESP32_ID,
        "zone": ZONE
    }).execute()

    # === Fire Detection ===
    if flame < CONFIG["fire_alert_threshold"]:
        print("🔥 Fire detected!")
        send_telegram_message("🔥 Fire detected!")
        supabase.table("fire_alerts").insert({
            "timestamp": now.isoformat(),
            "flame_value": flame,
            "esp32_id": ESP32_ID,
            "zone": ZONE,
            "comment": "🔥 Fire sensor triggered"
        }).execute()
        return JSONResponse({"alert": True, "message": "🔥 Fire detected!"})

    # === Rat/Human Detection ===
    if pir >= CONFIG["human_sensitivity"] and ultrasonic >= CONFIG["human_sensitivity"]:
        print("👤 Human detected")
        send_telegram_message("👤 Human detected")
        supabase.table("human_detection").insert({
            "timestamp": now.isoformat(),
            "pir_value": pir,
            "ultrasonic_value": ultrasonic,
            "esp32_id": ESP32_ID,
            "zone": ZONE
        }).execute()
    elif pir >= CONFIG["rat_sensitivity"] and ultrasonic == 0:
        print("🐀 Rat detected")
        send_telegram_message("🐀 Rat detected")
        supabase.table("rat_detection").insert({
            "timestamp": now.isoformat(),
            "pir_value": pir,
            "ultrasonic_value": ultrasonic,
            "esp32_id": ESP32_ID,
            "zone": ZONE
        }).execute()

    # === Rain Detection ===
    rain_expected = False
    if rain == 1:
        if rain_detected_start is None:
            rain_detected_start = now
        elif (now - rain_detected_start).total_seconds() >= CONFIG["rain_duration_threshold_sec"]:
            print("🌧️ Continuous rain logged")
            send_telegram_message("🌧️ Continuous rain logged")
            supabase.table("rain_alerts").insert({
                "timestamp": now.isoformat(),
                "duration_sec": CONFIG["rain_duration_threshold_sec"],
                "rain_start_time": rain_detected_start.isoformat(),
                "rain_end_time": now.isoformat(),
                "esp32_id": ESP32_ID,
                "zone": ZONE
            }).execute()
            return JSONResponse({
                "should_water": 0,
                "reason": "🌧️ Continuous rain",
                "moisture": moisture,
                "rain_expected": True,
                "anomaly": False,
                "model_active": sgd_initialized
            })
    else:
        rain_detected_start = None

    # === Manual Watering ===
    if CONFIG["manual_water"]:
        print("🚿 Manual watering")
        send_telegram_message("🚿 Manual watering")
        supabase.table("watering_logs").insert({
            "timestamp": now.isoformat(),
            "target_ml": CONFIG["manual_target_ml"],
            "actual_ml": CONFIG["manual_target_ml"],
            "watering_duration_sec": 30,
            "watering_mode": "manual",
            "triggered_by": "app",
            "esp32_id": ESP32_ID,
            "zone": ZONE
        }).execute()
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
        print("⚠️ Danger: low moisture")
        send_telegram_message("⚠️ Danger: low moisture")
        return JSONResponse({
            "should_water": 1,
            "reason": "⚠️ Critical moisture level",
            "moisture": moisture,
            "rain_expected": False,
            "anomaly": False,
            "model_active": sgd_initialized
        })

    # === Weather Forecast ===
    try:
        if CONFIG["check_weather"]:
            url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY},{COUNTRY}&appid={WEATHER_API_KEY}&units=metric"
            data = requests.get(url).json()
            for entry in data["list"]:
                dt_txt = entry.get("dt_txt", "")
                if "rain" in entry["weather"][0]["main"].lower():
                    rain_expected = True
                    supabase.table("weather_alerts").insert({
                        "timestamp": now.isoformat(),
                        "alert_type": "rain",
                        "alert_details": entry["weather"][0]["description"],
                        "esp32_id": ESP32_ID,
                        "zone": ZONE
                    }).execute()
                    break
    except Exception as e:
        print("Weather check failed:", e)
        send_telegram_message(f"🌦️ Weather check failed: {e}")

    # === ML Features
    features = np.array([[temperature, humidity, moisture, ldr, rain]])
    anomaly_flag = False
    try:
        if models_loaded:
            if anomaly_model.decision_function(features)[0] < -0.2:
                anomaly_flag = True
                print("⚠️ Anomaly detected")
                send_telegram_message("⚠️ Anomaly detected in sensor readings")
    except:
        pass

    # === Live Learning
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
        X_scaled_test = scaler.transform(features)
        _ = int(sgd_model.predict(X_scaled_test)[0])

    # === Watering Decision
    should_water = 0
    reason = "No watering needed"
    model_type = "none"
    model_result = 0

    if not anomaly_flag and moisture < CONFIG["moisture_threshold"] and not rain_expected:
        if rf_model:
            model_result = int(rf_model.predict(features)[0])
            should_water = model_result
            model_type = "RF"
            reason = "🤖 ML model"
        else:
            should_water = 1
            reason = "Threshold"
            model_type = "threshold"

    supabase.table("watering_decisions").insert({
        "timestamp": now.isoformat(),
        "should_water": bool(should_water),
        "model_result": model_result,
        "model_type": model_type,
        "moisture": moisture,
        "temperature": temperature,
        "humidity": humidity,
        "rain": rain,
        "confidence": 0.9,
        "reason": reason,
        "anomaly_detected": anomaly_flag,
        "esp32_id": ESP32_ID,
        "zone": ZONE
    }).execute()

    if should_water:
        supabase.table("watering_logs").insert({
            "timestamp": now.isoformat(),
            "target_ml": 100,
            "actual_ml": 100,
            "watering_duration_sec": 30,
            "watering_mode": "auto",
            "triggered_by": "server",
            "esp32_id": ESP32_ID,
            "zone": ZONE
        }).execute()

    return JSONResponse({
        "should_water": should_water,
        "moisture": moisture,
        "model_active": sgd_initialized,
        "rain_expected": rain_expected,
        "anomaly": anomaly_flag,
        "reason": reason
    })

@app.get("/")
def root():
    return {"message": "🌿 Smart Irrigation API - Fully Integrated with Supabase"}


@app.get("/commands")
def get_commands():
    return {
        "manual_trigger": CONFIG["manual_water"],
        "target_ml": CONFIG["manual_target_ml"],
        "should_water": 1 if CONFIG["manual_water"] else 0,
        "enable_pir": CONFIG.get("enable_pir", True),
        "enable_ultrasonic": CONFIG.get("enable_ultrasonic", True)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
