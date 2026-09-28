from flask import Flask, render_template, Response, jsonify
import cv2
import time
import math
import os
import PIL.Image
from google import genai
from google.genai import types
from ultralytics import YOLO
from twilio.rest import Client

app = Flask(__name__)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
VIDEO_SOURCE = 0 
model = YOLO("yolov8n.pt")

# Adjust these for your environment
SAFE_ZONE            = (50, 50, 590, 430)
DISTANCE_THRESHOLD   = 120
ALERT_COOLDOWN       = 20
INACTIVITY_THRESHOLD = 30
NIGHT_HOURS          = (22, 6)

DANGEROUS_CLASSES = ["dog", "knife", "car", "truck"]
FIRE_CLASSES      = ["fire", "smoke"]
ANIMAL_CLASSES    = ["cat", "dog", "bird", "snake", "horse", "cow", "elephant"]

# ─────────────────────────────────────────
# GEMINI 2026 CONFIG (Updated SDK)
# ─────────────────────────────────────────
GEMINI_API_KEY = ""#add your api
MODEL_NAME = "gemini-3-flash-preview"  # Most stable 2026 model
GEMINI_COOLDOWN = 15

# Initialize New GenAI Client
client = genai.Client(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────
# TWILIO CONFIG
# ─────────────────────────────────────────
TWILIO_SID   = ""
TWILIO_AUTH  = ""
TWILIO_FROM  = ""
PARENT_PHONE = "" # MUST BE VERIFIED in Twilio Trial Console

twilio_client = Client(TWILIO_SID, TWILIO_AUTH)

# ─────────────────────────────────────────
# GLOBAL STATE & FOLDERS
# ─────────────────────────────────────────
current_status = {
    "risk_level": 0,
    "status": "Safe",
    "ai_description": "",
    "timestamp": time.time(),
}

danger_active           = False
last_alert_time         = 0
last_gemini_time        = 0
previous_child_center_y = None
child_stationary_start  = None

os.makedirs("incidents", exist_ok=True)
camera = cv2.VideoCapture(VIDEO_SOURCE)

# ─────────────────────────────────────────
# PROMPTS & ANALYSIS
# ─────────────────────────────────────────

FALL_PROMPT = "A FALL DETECTED. Describe the person's clothing, position, exact location in scene, and severity. End with a recommended action."
INACTIVITY_PROMPT = "PERSON MOTIONLESS. Describe position and surroundings. Is it a normal rest or collapse? Action needed?"
ANIMAL_PROMPT = "DANGEROUS ANIMAL. Identify type/size, location, and risk to nearby persons."
FIRE_PROMPT = "FIRE/SMOKE. Identify location, intensity, and immediate danger level."
INTRUDER_PROMPT = "NIGHT INTRUSION. Describe clothing/build and location. Security action required."
DANGER_OBJECT_PROMPT = "DANGEROUS OBJECT NEAR PERSON. Describe distance and potential risk."

PROMPT_MAP = {
    "fall": FALL_PROMPT, "inactivity": INACTIVITY_PROMPT, "animal": ANIMAL_PROMPT,
    "fire": FIRE_PROMPT, "intruder": INTRUDER_PROMPT, "danger": DANGER_OBJECT_PROMPT
}

def describe_scene_with_gemini(frame, event_type="fall", extra_context=""):
    global last_gemini_time
    now = time.time()
    
    if (now - last_gemini_time) < GEMINI_COOLDOWN:
        return current_status.get("ai_description", "Analysis in progress...")

    last_gemini_time = now

    # Prep Image for New SDK
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = PIL.Image.fromarray(rgb_frame)

    prompt_text = PROMPT_MAP.get(event_type, FALL_PROMPT)
    if extra_context:
        prompt_text += f"\n\nContext: {extra_context}"

    try:
        print(f"[Gemini] Analyzing: {event_type}...")
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt_text, pil_img],
            config=types.GenerateContentConfig(temperature=0.2)
        )
        
        if response.text:
            print(f"[Gemini] Success: {response.text[:50]}...")
            return response.text.strip()
        return "[AI Analysis failed to generate text]"
    except Exception as e:
        print(f"[Gemini Error]: {e}")
        return f"[AI Error: {str(e)[:50]}]"

# ─────────────────────────────────────────
# ALERTING LOGIC
# ─────────────────────────────────────────

def fire_alert(status_text, frame, event_type="fall", extra_context="", use_whatsapp=False):
    global last_alert_time, danger_active
    now = time.time()

    if danger_active or (now - last_alert_time < ALERT_COOLDOWN):
        return
    
    last_alert_time = now
    danger_active = True

    # Save snapshot
    filename = f"incidents/alert_{int(now)}.jpg"
    cv2.imwrite(filename, frame)

    # Trigger AI
    ai_desc = describe_scene_with_gemini(frame, event_type, extra_context)
    current_status["ai_description"] = ai_desc

    # Notify via Twilio
    full_msg = f"🚨 {status_text}\n\nAI Analysis: {ai_desc[:140]}..."
    try:
        twilio_client.messages.create(body=full_msg, from_=TWILIO_FROM, to=PARENT_PHONE)
        print("📩 Alert Sent via Twilio")
    except Exception as e:
        print(f"Twilio Error (Check if {PARENT_PHONE} is verified): {e}")

# ─────────────────────────────────────────
# MAIN PROCESSING
# ─────────────────────────────────────────

def generate_frames():
    global current_status, previous_child_center_y, child_stationary_start, danger_active

    while True:
        success, frame = camera.read()
        if not success: break

        results = model(frame, verbose=False)
        result = results[0]

        child_box = None
        danger_now = False
        alert_reason = "fall"
        alert_extra = ""

        # Simplified Detection Loop
        for box in result.boxes:
            c = int(box.cls[0])
            name = result.names[c]
            if float(box.conf[0]) < 0.45: continue
            
            coords = [int(x) for x in box.xyxy[0]]
            if name == "person":
                child_box = coords
                cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (255, 80, 0), 2)
            elif name in FIRE_CLASSES:
                danger_now, alert_reason = True, "fire"

        # Fall/Inactivity Logic
        if child_box:
            x1, y1, x2, y2 = child_box
            w, h = x2-x1, y2-y1
            center_y = (y1 + y2) // 2

            if w > h * 1.2:
                danger_now, alert_reason = True, "fall"
                alert_extra = "Horizontal posture detected."
            
            previous_child_center_y = center_y

        # Handle UI and Alerts
        if danger_now:
            fire_alert(f"CRITICAL: {alert_reason.upper()}!", frame, alert_reason, alert_extra)
        else:
            danger_active = False

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

# ─────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────

@app.route("/")
def index(): return render_template("index1.html")

@app.route("/video_feed")
def video_feed(): return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/get_status")
def get_status(): return jsonify(current_status)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
