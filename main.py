from fastapi import FastAPI, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, Union
from supabase import create_client, Client
from datetime import datetime, timedelta
import uvicorn
import numpy as np
import os
import joblib
import requests
from pydantic import BaseModel
from sklearn.neural_network import MLPClassifier
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
    "manual_target_ml": 100,
    "enable_pir": True,
    "enable_ultrasonic": True
}

# === Models ===
mlp_model = MLPClassifier(hidden_layer_sizes=(10, 10), warm_start=True, max_iter=1)
scaler = StandardScaler()
X_buffer, y_buffer = [], []
mlp_initialized = False

try:
    anomaly_model = joblib.load("isolation_forest.pkl")
    rf_model = joblib.load("rf_water_decision.pkl")
    models_loaded = True
except Exception as e:
    print("⚠️ Model loading failed:", e)
    send_telegram_message(f"⚠️ Model loading failed: {e}")
    rf_model = anomaly_model = None
    models_loaded = False
#LOG
class CommandLog(BaseModel):
    command: str
    issued_by: Optional[str] = "esp32"
    status: Optional[str] = "sent"
    response: Optional[str] = None

@app.post("/log_command")
def log_command(cmd: CommandLog):
    now = datetime.now().isoformat()
    try:
        log_to_supabase("command_logs", {
            "timestamp": now,
            "command": cmd.command,
            "issued_by": cmd.issued_by,
            "status": cmd.status,
            "response": cmd.response,
            "esp32_id": ESP32_ID,
            "zone": ZONE
        })
        return {"message": "✅ Command log saved"}
    except Exception as e:
        send_telegram_message(f"❌ Command log failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# === State Tracking ===
last_signal_time = datetime.now()
rain_detected_start = None
last_moisture = None

def log_to_supabase(table: str, data: dict):
    try:
        supabase.table(table).insert(data).execute()
    except Exception as e:
        print(f"❌ Failed to log to {table}: {e}")

@app.get("/waternow")
def water_now():
    CONFIG["manual_water"] = True
    return {"message": "✅ Manual watering command activated"}

@app.get("/stopwater")
def stop_water():
    CONFIG["manual_water"] = False
    return {"message": "🛑 Manual watering stopped"} 

@app.get("/enablepir")
def enable_pir():
    CONFIG["enable_pir"] = True
    return {"message": "✅ PIR sensor enabled"}

@app.get("/disablepir")
def disable_pir():
    CONFIG["enable_pir"] = False
    return {"message": "🛑 PIR sensor disabled"}

@app.get("/enableultrasonic")
def enable_ultrasonic():
    CONFIG["enable_ultrasonic"] = True
    return {"message": "✅ Ultrasonic sensor enabled"}

@app.get("/disableultrasonic")
def disable_ultrasonic():
    CONFIG["enable_ultrasonic"] = False
    return {"message": "🛑 Ultrasonic sensor disabled"}



@app.post("/sensor-data")
def sensor_data(
    background_tasks: BackgroundTasks,
    temperature: float = Form(...),
    humidity: float = Form(...),
    moisture: float = Form(...),
    ldr: float = Form(...),
    battery_voltage: Union[str, None] = Form(None),
    rain: int = Form(...),
    flame: int = Form(...),
    watered: int = Form(...),
    pir: int = Form(...),
    ultrasonic: int = Form(...)
):
    global last_signal_time, rain_detected_start, last_moisture, mlp_initialized
    now = datetime.now()

    try:
        battery_val = float(battery_voltage) if battery_voltage not in [None, ""] else None
    except ValueError:
        battery_val = None

    background_tasks.add_task(log_to_supabase, "sensor_readings", {
        "timestamp": now.isoformat(),
        "temperature": temperature,
        "humidity": humidity,
        "moisture": moisture,
        "ldr": ldr,
        "battery_voltage": battery_val,
        "rain": rain,
        "flame": flame,
        "pir": pir,
        "ultrasonic": ultrasonic,
        "esp32_id": ESP32_ID,
        "zone": ZONE
    })

    if flame < CONFIG["fire_alert_threshold"]:
        send_telegram_message("🔥 Fire detected!")
        background_tasks.add_task(log_to_supabase, "fire_alerts", {
            "timestamp": now.isoformat(),
            "flame_value": flame,
            "esp32_id": ESP32_ID,
            "zone": ZONE,
            "comment": "🔥 Fire sensor triggered"
        })
        return JSONResponse({"alert": True, "message": "🔥 Fire detected!"})

    if pir >= CONFIG["human_sensitivity"] and ultrasonic >= CONFIG["human_sensitivity"]:
        send_telegram_message("👤 Human detected")
        background_tasks.add_task(log_to_supabase, "human_detection", {
            "timestamp": now.isoformat(),
            "pir_value": pir,
            "ultrasonic_value": ultrasonic,
            "esp32_id": ESP32_ID,
            "zone": ZONE
        })
    elif pir >= CONFIG["rat_sensitivity"] and ultrasonic == 0:
        send_telegram_message("🐀 Rat detected")
        background_tasks.add_task(log_to_supabase, "rat_detection", {
            "timestamp": now.isoformat(),
            "pir_value": pir,
            "ultrasonic_value": ultrasonic,
            "esp32_id": ESP32_ID,
            "zone": ZONE
        })

    rain_expected = False
    if rain == 1:
        if rain_detected_start is None:
            rain_detected_start = now
        elif (now - rain_detected_start).total_seconds() >= CONFIG["rain_duration_threshold_sec"]:
            send_telegram_message("🌧️ Continuous rain logged")
            background_tasks.add_task(log_to_supabase, "rain_alerts", {
                "timestamp": now.isoformat(),
                "duration_sec": CONFIG["rain_duration_threshold_sec"],
                "rain_start_time": rain_detected_start.isoformat(),
                "rain_end_time": now.isoformat(),
                "esp32_id": ESP32_ID,
                "zone": ZONE
            })
            return JSONResponse({
                "should_water": 0,
                "reason": "🌧️ Continuous rain",
                "moisture": moisture,
                "rain_expected": True,
                "anomaly": False,
                "model_active": mlp_initialized
            })
    else:
        rain_detected_start = None

    if CONFIG["manual_water"]:
        send_telegram_message("🚿 Manual watering")
        background_tasks.add_task(log_to_supabase, "watering_logs", {
            "timestamp": now.isoformat(),
            "target_ml": CONFIG["manual_target_ml"],
            "actual_ml": CONFIG["manual_target_ml"],
            "watering_duration_sec": 30,
            "watering_mode": "manual",
            "triggered_by": "app",
            "esp32_id": ESP32_ID,
            "zone": ZONE
        })
        return JSONResponse({
            "should_water": 1,
            "reason": "Manual watering",
            "target_ml": CONFIG["manual_target_ml"],
            "moisture": moisture,
            "rain_expected": False,
            "anomaly": False,
            "model_active": mlp_initialized
        })

    if moisture < CONFIG["danger_threshold"]:
        send_telegram_message("⚠️ Danger: low moisture")
        return JSONResponse({
            "should_water": 1,
            "reason": "⚠️ Critical moisture level",
            "moisture": moisture,
            "rain_expected": False,
            "anomaly": False,
            "model_active": mlp_initialized
        })

    try:
        if CONFIG["check_weather"]:
            url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY},{COUNTRY}&appid={WEATHER_API_KEY}&units=metric"
            data = requests.get(url).json()
            for entry in data["list"]:
                if "rain" in entry["weather"][0]["main"].lower():
                    rain_expected = True
                    background_tasks.add_task(log_to_supabase, "weather_alerts", {
                        "timestamp": now.isoformat(),
                        "alert_type": "rain",
                        "alert_details": entry["weather"][0]["description"],
                        "esp32_id": ESP32_ID,
                        "zone": ZONE
                    })
                    break
    except Exception as e:
        send_telegram_message(f"🌦️ Weather check failed: {e}")

    features = np.array([[temperature, humidity, moisture, ldr, rain]])
    anomaly_flag = False
    try:
        if models_loaded:
            if anomaly_model.decision_function(features)[0] < -0.2:
                anomaly_flag = True
                send_telegram_message("⚠️ Anomaly detected in sensor readings")
    except:
        pass

    label = 1 if (moisture < CONFIG["moisture_threshold"] and not rain_expected) else 0
    X_buffer.append(features[0])
    y_buffer.append(label)

    if len(X_buffer) >= 10:
        X_np = np.array(X_buffer)
        y_np = np.array(y_buffer)
        X_scaled = scaler.fit_transform(X_np)
        if not mlp_initialized:
            mlp_model.partial_fit(X_scaled, y_np, classes=[0, 1])
            mlp_initialized = True
        else:
            mlp_model.partial_fit(X_scaled, y_np)
        X_buffer.clear()
        y_buffer.clear()

    if mlp_initialized:
        X_scaled_test = scaler.transform(features)
        mlp_result = int(mlp_model.predict(X_scaled_test)[0])
        rf_result = int(rf_model.predict(features)[0]) if rf_model else 0
    else:
        mlp_result = rf_result = 0

    should_water = 0
    reason = "No watering needed"
    model_type = "none"

    if not anomaly_flag and moisture < CONFIG["moisture_threshold"] and not rain_expected:
        if mlp_result == rf_result:
            should_water = mlp_result
            model_type = "MLP + RF"
            reason = "✅ Both models agree"
        else:
            should_water = mlp_result
            model_type = "MLP only"
            reason = "⚠️ Conflict: RF disagrees"

    background_tasks.add_task(log_to_supabase, "watering_decisions", {
        "timestamp": now.isoformat(),
        "should_water": bool(should_water),
        "model_result": mlp_result,
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
    })

    if should_water:
        background_tasks.add_task(log_to_supabase, "watering_logs", {
            "timestamp": now.isoformat(),
            "target_ml": 100,
            "actual_ml": 100,
            "watering_duration_sec": 30,
            "watering_mode": "auto",
            "triggered_by": "server",
            "esp32_id": ESP32_ID,
            "zone": ZONE
        })

    return JSONResponse({
        "should_water": should_water,
        "moisture": moisture,
        "model_active": mlp_initialized,
        "rain_expected": rain_expected,
        "anomaly": anomaly_flag,
        "reason": reason
    })

@app.get("/")
def root():
    return {"message": "🌿 Smart Irrigation API - MLP Enhanced"}

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
