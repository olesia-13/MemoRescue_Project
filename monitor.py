from dotenv import load_dotenv
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import requests
import time
from collections import deque
from scipy.spatial.distance import cosine

load_dotenv()

# Вкажіть точні координати місця встановлення камери
CAMERA_LAT = 50.5186  
CAMERA_LON = 30.2397

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# ── ЗАГАЛЬНІ НАЛАШТУВАННЯ СИСТЕМИ ───────────────────────────
DATABASE_DIR = os.path.join(os.path.dirname(__file__), "database")
SIMILARITY_THRESHOLD = 0.85     
QUEUE_SIZE = 100                

ZIGZAG_WINDOW = 60              
ZIGZAG_SINUOSITY = 2.5          
STILLNESS_FRAMES = 90           
STILLNESS_VEL_THRESH = 0.003    

IDENTIFY_EVERY = 30             
ALERT_COOLDOWN = 60             
ANOMALY_CONFIRM_TIME = 2.0      # Час для підтвердження тривоги

if TELEGRAM_BOT_TOKEN is None:
    print("[ERROR] Токен не знайдено! Перевірте файл .env")
else:
    print(f"[INFO] Токен успішно завантажено: {TELEGRAM_BOT_TOKEN[:10]}...")
    
# Словники стану
last_alerts = {}
anomaly_start_time = None       

# ── MEDIAPIPE ІНІЦІАЛІЗАЦІЯ ─────────────────────────────────
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── ФУНКЦІЯ TELEGRAM ────────────────────────────────────────
def send_telegram_alert(user_name, phone, chat_id, alert_type):
    """Надсилає сповіщення з координатами камери та посиланням на карту."""
    if not chat_id:
        print(f"[WARN] Chat ID для {user_name} не знайдено.")
        return

    # Формуємо посилання на Google Maps
    maps_link = f"https://www.google.com/maps?q={49.8441550958368},{24.026250638148717}"
    
    emoji = "🚨" if alert_type == "ZIGZAG" else "⚠️"
    message = (
        f"{emoji} MemoRescue: ПІДТВЕРДЖЕНО ТРИВОГУ!!!\n"
        f"----------------------------------\n"
        f"👤 Особа: {user_name}\n"
        f"🔗 Карта: {maps_link}\n"
        f"⏰ Час: {time.strftime('%H:%M:%S')}"
    )
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        payload = {"chat_id": chat_id, "text": message}
        requests.post(url, data=payload)
        print(f"[SUCCESS] Тривогу з геолокацією надіслано для {user_name}")
    except Exception as e:
        print(f"[ERROR] Не вдалося надіслати повідомлення: {e}")

# ── ДОПОМІЖНІ ФУНКЦІЇ АНАЛІЗУ ──────────────────────────────
def _angle(a, b, c):
    va = np.array([a.x - b.x, a.y - b.y])
    vc = np.array([c.x - b.x, c.y - b.y])
    n_a, n_c = np.linalg.norm(va), np.linalg.norm(vc)
    if n_a < 1e-6 or n_c < 1e-6: return 0.0
    return float(np.arccos(np.clip(np.dot(va, vc) / (n_a * n_c), -1.0, 1.0)))

def extract_features(landmarks):
    try:
        lm = landmarks
        nose = lm[mp_pose.PoseLandmark.NOSE]
        l_an, r_an = lm[mp_pose.PoseLandmark.LEFT_ANKLE], lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
        l_hip, r_hip = lm[mp_pose.PoseLandmark.LEFT_HIP], lm[mp_pose.PoseLandmark.RIGHT_HIP]
        l_kn, r_kn = lm[mp_pose.PoseLandmark.LEFT_KNEE], lm[mp_pose.PoseLandmark.RIGHT_KNEE]
        l_sh, r_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        h = abs(nose.y - l_an.y)
        if h < 0.01: return None
        return [np.sqrt((l_an.x-r_an.x)**2+(l_an.y-r_an.y)**2)/h, _angle(l_hip,l_kn,l_an), _angle(r_hip,r_kn,r_an),
                _angle(l_sh,l_hip,l_kn), _angle(r_sh,r_hip,r_kn), np.sqrt((l_sh.x-r_sh.x)**2+(l_sh.y-r_sh.y)**2)/h, abs(l_an.y-r_an.y)/h]
    except: return None

def load_database():
    users = []
    if not os.path.exists(DATABASE_DIR): return users
    for fn in os.listdir(DATABASE_DIR):
        if fn.endswith(".json"):
            with open(os.path.join(DATABASE_DIR, fn), "r", encoding="utf-8") as f:
                users.append(json.load(f))
    return users

def identify_person(current_vec, users):
    best_u, best_s = None, -1.0
    curr = np.array(current_vec).flatten()
    for u in users:
        sig = np.array(u["gait_signature"]).flatten()
        if curr.shape == sig.shape:
            sim = 1.0 - cosine(curr, sig)
            if sim > best_s: best_s, best_u = sim, u
    return (best_u, best_s) if best_s >= SIMILARITY_THRESHOLD else (None, best_s)

def detect_zigzag(positions):
    if len(positions) < ZIGZAG_WINDOW: return False
    pts = list(positions)[-ZIGZAG_WINDOW:]
    disp = np.linalg.norm(pts[-1] - pts[0])
    if disp < 0.01: return False
    path = sum(np.linalg.norm(pts[i]-pts[i-1]) for i in range(1, len(pts)))
    return (path / disp) > ZIGZAG_SINUOSITY

def detect_stillness(positions):
    if len(positions) < STILLNESS_FRAMES: return False
    pts = list(positions)[-STILLNESS_FRAMES:]
    return all(np.linalg.norm(pts[i]-pts[i-1]) < STILLNESS_VEL_THRESH for i in range(1, len(pts)))

# ── ГОЛОВНИЙ ЦИКЛ ───────────────────────────────────────────
def run_monitor():
    global anomaly_start_time
    cap = cv2.VideoCapture(0)
    users = load_database()
    print(f"[INFO] Камера активована. Профілів у базі: {len(users)}")

    feature_buf = deque(maxlen=QUEUE_SIZE)
    pos_queue = deque(maxlen=QUEUE_SIZE)
    identified = None
    frame_idx = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        label, color = "Scanning...", (255, 255, 0)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark
            feats = extract_features(lm)
            if feats:
                feature_buf.append(feats)
                pos_queue.append(np.array([(lm[23].x+lm[24].x)/2, (lm[23].y+lm[24].y)/2]))

            # Ідентифікація
            if frame_idx % IDENTIFY_EVERY == 0 and len(feature_buf) >= 20:
                avg_vec = np.mean(list(feature_buf), axis=0).tolist()
                identified, sim = identify_person(avg_vec, users)

            if identified:
                label, color = f"OK: {identified['name']}", (0, 255, 0)
                
                # Аналіз аномалій
                active_anomaly = None
                if detect_zigzag(pos_queue): active_anomaly = "ZIGZAG"
                elif detect_stillness(pos_queue): active_anomaly = "STILLNESS"

                if active_anomaly:
                    if anomaly_start_time is None: 
                        anomaly_start_time = time.time()
                    
                    elapsed = time.time() - anomaly_start_time
                    label = f"CONFIRMING {active_anomaly}: {elapsed:.1f}s"
                    color = (0, 165, 255)

                    if elapsed >= ANOMALY_CONFIRM_TIME:
                        label = f"!!! ALARM {active_anomaly} !!!"
                        color = (0, 0, 255)
                        u_name = identified['name']
                        if time.time() - last_alerts.get(u_name, 0) > ALERT_COOLDOWN:
                            send_telegram_alert(u_name, identified['phone'], identified.get('chat_id'), active_anomaly)
                            last_alerts[u_name] = time.time()
                else:
                    anomaly_start_time = None
            else:
                label, color = "Unknown", (0, 165, 255)
        else:
            label, color = "No person", (128, 128, 128)
            anomaly_start_time = None

        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow(f"MemoRescue Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_monitor()

