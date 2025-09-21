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

def is_flame_detected(flame_val: int, ldr_val: float, temp: float) -> bool:
    return flame_val < CONFIG["fire_alert_threshold"] and ldr_val < 1000 and temp > 70.0


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

@app.post("/waternow") 
def water_now(): 
    try: 
        log_to_supabase("device_commands", { 
            "esp32_id": ESP32_ID, 
            "command": "START_WATERING", 
            "value": {"ml": 100} 
        }) 
        return {"message": "✅ Manual watering command logged"} 
    except Exception as e: 
        return JSONResponse(status_code=500, content={"error": str(e)}) 

@app.post("/stopwater") 
def stop_water(): 
    try: 
        log_to_supabase("device_commands", { 
            "esp32_id": ESP32_ID, 
            "command": "STOP_WATERING", 
            "value": {} 
        }) 
        return {"message": "🛑 Manual watering stopped"} 
    except Exception as e: 
        return JSONResponse(status_code=500, content={"error": str(e)}) 

@app.post("/enablepir") 
def enable_pir(): 
    try: 
        log_to_supabase("device_commands", { 
            "esp32_id": ESP32_ID, 
            "command": "ENABLE_PIR", 
            "value": {} 
        }) 
        return {"message": "✅ PIR sensor enabled"} 
    except Exception as e: 
        return JSONResponse(status_code=500, content={"error": str(e)}) 

@app.post("/disablepir") 
def disable_pir(): 
    try: 
        log_to_supabase("device_commands", { 
            "esp32_id": ESP32_ID, 
            "command": "DISABLE_PIR", 
            "value": {} 
        }) 
        return {"message": "🛑 PIR sensor disabled"} 
    except Exception as e: 
        return JSONResponse(status_code=500, content={"error": str(e)}) 

@app.post("/enableultrasonic") 
def enable_ultrasonic(): 
    try: 
        log_to_supabase("device_commands", { 
            "esp32_id": ESP32_ID, 
            "command": "ENABLE_ULTRASONIC", 
            "value": {} 
        }) 
        return {"message": "✅ Ultrasonic sensor enabled"} 
    except Exception as e: 
        return JSONResponse(status_code=500, content={"error": str(e)}) 

@app.post("/disableultrasonic") 
def disable_ultrasonic(): 
    try: 
        log_to_supabase("device_commands", { 
            "esp32_id": ESP32_ID, 
            "command": "DISABLE_ULTRASONIC", 
            "value": {} 
        }) 
        return {"message": "🛑 Ultrasonic sensor disabled"} 
    except Exception as e: 
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/commands")
def get_latest_command(last_id: Optional[int] = 0):
    try:
        # Fetch the single most recent command from the device_commands table
        # where the ID is greater than the last processed ID
        response = supabase.table("device_commands").select("id, command, value").eq("esp32_id", ESP32_ID).gt("id", last_id).order("id", desc=True).limit(1).execute()

        if not response.data:
            return {"command": "NO_COMMAND", "value": {}}

        latest_command = response.data[0]
        return {
            "id": latest_command["id"],
            "command": latest_command["command"],
            "value": latest_command["value"]
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})




@app.post("/sensor-data")
def sensor_data(
    background_tasks: BackgroundTasks,
    temperature: float = Form(...),
    humidity: float = Form(...),
    moisture: float = Form(...),
    ldr: float = Form(...),
    battery_voltage: float = Form(...),
    rain: int = Form(...),
    flame: int = Form(...),
    watered: int = Form(...),
    pir: int = Form(...),
    ultrasonic: int = Form(...)
):
    global last_signal_time, rain_detected_start, last_moisture, mlp_initialized
    now = datetime.now()


    background_tasks.add_task(log_to_supabase, "sensor_readings", {
        "timestamp": now.isoformat(),
        "temperature": temperature,
        "humidity": humidity,
        "moisture": moisture,
        "ldr": ldr,
        "battery_voltage": battery_voltage,
        "rain": rain,
        "flame": flame,
        "pir": pir,
        "ultrasonic": ultrasonic,
        "esp32_id": ESP32_ID,
        "zone": ZONE
    })

    if is_flame_detected(flame, ldr, temperature):  # ✅ fixed here
        send_telegram_message("🔥 Fire detected! (LDR + Temp Verified)")
        background_tasks.add_task(log_to_supabase, "fire_alerts", {
            "timestamp": now.isoformat(),
            "flame_value": flame,
            "ldr_value": ldr,
            "temperature": temperature,
            "esp32_id": ESP32_ID,
            "zone": ZONE,
            "comment": "🔥 Verified flame detection (LDR+Temp)"
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



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
