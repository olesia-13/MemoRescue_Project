"""
MemoRescue — Крок 3-5: Моніторинг з камери
  • Виявлення скелета через MediaPipe
  • Ідентифікація людини (cosine similarity ≥ 0.85)
  • Аналіз аномалій: зигзаг центру мас / тривала нерухомість
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from collections import deque
from scipy.spatial.distance import cosine

# ── Налаштування ────────────────────────────────────────────
DATABASE_DIR = os.path.join(os.path.dirname(__file__), "database")
SIMILARITY_THRESHOLD = 0.85     # поріг cosine-схожості
QUEUE_SIZE = 100                # черга останніх кадрів

ZIGZAG_WINDOW = 60              # вікно для детекції зигзагу (кадри)
ZIGZAG_SMOOTH = 10              # ковзне середнє для згладжування природного хитання
ZIGZAG_SINUOSITY = 2.5          # поріг звивистості (path_length / displacement)

STILLNESS_VEL_THRESH = 0.003    # поріг швидкості (нормалізовані коорд.)
STILLNESS_FRAMES = 90           # скільки кадрів нерухомості = тривога (~3 сек)

IDENTIFY_EVERY = 30             # перерахунок ідентифікації кожні N кадрів
MIN_FEATURES_FOR_ID = 20        # мін. кадрів у буфері для порівняння

# ── MediaPipe ───────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ── Допоміжні функції ──────────────────────────────────────
def _angle(a, b, c):
    """Кут у точці b (рад.)."""
    va = np.array([a.x - b.x, a.y - b.y])
    vc = np.array([c.x - b.x, c.y - b.y])
    cos_a = np.dot(va, vc) / (np.linalg.norm(va) * np.linalg.norm(vc) + 1e-6)
    return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def extract_features(landmarks):
    """7-елементний вектор ознак з одного кадру (той самий, що і в registration)."""
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    l_kn = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    r_kn = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    l_an = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    r_an = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    height = abs(nose.y - l_an.y)
    if height < 0.01:
        return None

    ankle_dist = np.sqrt((l_an.x - r_an.x) ** 2 + (l_an.y - r_an.y) ** 2) / height
    l_knee_a = _angle(l_hip, l_kn, l_an)
    r_knee_a = _angle(r_hip, r_kn, r_an)
    l_hip_a = _angle(l_sh, l_hip, l_kn)
    r_hip_a = _angle(r_sh, r_hip, r_kn)
    shoulder_w = np.sqrt((l_sh.x - r_sh.x) ** 2 + (l_sh.y - r_sh.y) ** 2) / height
    step_h = abs(l_an.y - r_an.y) / height

    return [ankle_dist, l_knee_a, r_knee_a, l_hip_a, r_hip_a, shoulder_w, step_h]


def get_pelvis(landmarks):
    """Центр мас ≈ середина між лівим і правим стегном (таз)."""
    l = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    r = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    return np.array([(l.x + r.x) / 2, (l.y + r.y) / 2])


# ── База даних ─────────────────────────────────────────────
def load_database():
    """Завантажує всі JSON-профілі з папки database."""
    users = []
    if not os.path.exists(DATABASE_DIR):
        return users
    for fn in os.listdir(DATABASE_DIR):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(DATABASE_DIR, fn), "r", encoding="utf-8") as f:
            data = json.load(f)
        # пропускаємо старий формат (одне число)
        if not isinstance(data.get("gait_signature"), list):
            continue
        users.append(data)
    return users


# ── Ідентифікація ──────────────────────────────────────────
def identify_person(current_vec, users):
    """Повертає (user_dict, similarity) або (None, best_sim)."""
    best_user = None
    best_sim = -1.0
    for u in users:
        sig = u["gait_signature"]
        sim = 1.0 - cosine(current_vec, sig)
        if sim > best_sim:
            best_sim = sim
            best_user = u
    if best_sim >= SIMILARITY_THRESHOLD:
        return best_user, best_sim
    return None, best_sim


# ── Детекція аномалій ──────────────────────────────────────
def _smooth(pts, window):
    """Ковзне середнє для списку 2D-точок."""
    if len(pts) < window:
        return pts
    kernel = np.ones(window) / window
    xs = np.convolve([p[0] for p in pts], kernel, mode="valid")
    ys = np.convolve([p[1] for p in pts], kernel, mode="valid")
    return [np.array([x, y]) for x, y in zip(xs, ys)]


def detect_zigzag(positions):
    """
    Зигзаг = звивистість (sinuosity) згладженої траєкторії.
    sinuosity = сумарна довжина шляху / пряма відстань між початком і кінцем.
    Нормальна ходьба ≈ 1.0–2.0, зигзаг > ZIGZAG_SINUOSITY.
    Згладжування прибирає природне хитання тазу при кожному кроці.
    """
    if len(positions) < ZIGZAG_WINDOW:
        return False

    raw = list(positions)[-ZIGZAG_WINDOW:]
    pts = _smooth(raw, ZIGZAG_SMOOTH)

    if len(pts) < 3:
        return False

    # пряма відстань від початку до кінця
    displacement = np.linalg.norm(pts[-1] - pts[0])
    if displacement < 0.005:
        # людина майже не зрушила з місця — це не зигзаг, а можливо нерухомість
        return False

    # сумарна довжина шляху
    path_len = sum(np.linalg.norm(pts[i] - pts[i - 1]) for i in range(1, len(pts)))

    sinuosity = path_len / displacement
    return sinuosity > ZIGZAG_SINUOSITY


def detect_stillness(positions):
    """
    Нерухомість = швидкість < порогу впродовж STILLNESS_FRAMES кадрів (~3 с).
    """
    if len(positions) < STILLNESS_FRAMES:
        return False

    pts = list(positions)[-STILLNESS_FRAMES:]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[i - 1]) > STILLNESS_VEL_THRESH:
            return False
    return True


# ── Головний цикл моніторингу ──────────────────────────────
def run_monitor():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Не вдалося відкрити камеру!")
        return

    users = load_database()
    print(f"[INFO] Завантажено профілів: {len(users)}")
    if len(users) == 0:
        print("[WARN] Немає профілів з векторним gait_signature. "
              "Перереєструйте користувачів через registration.py")
    print("[INFO] Камера запущена. Натисніть 'q' для виходу.\n")

    feature_buf = deque(maxlen=QUEUE_SIZE)   # для ідентифікації
    pos_queue = deque(maxlen=QUEUE_SIZE)     # для аномалій

    identified = None
    alarm_type = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        label = "Scanning..."
        color = (255, 255, 0)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            lm = results.pose_landmarks.landmark

            # ── Збираємо ознаки ──
            feats = extract_features(lm)
            if feats is not None:
                feature_buf.append(feats)

            pelvis = get_pelvis(lm)
            pos_queue.append(pelvis)

            # ── Ідентифікація ──
            if (frame_idx % IDENTIFY_EVERY == 0
                    and len(feature_buf) >= MIN_FEATURES_FOR_ID
                    and len(users) > 0):
                sig = np.mean(list(feature_buf), axis=0).tolist()
                person, sim = identify_person(sig, users)
                if person:
                    identified = person
                    label = f"{person['name']}  (sim {sim:.2f})"
                    color = (0, 255, 0)
                else:
                    identified = None
                    label = f"Unknown (best {sim:.2f})"
                    color = (0, 165, 255)

            # ── Аномалії (тільки якщо людина ідентифікована) ──
            alarm_type = None
            if identified and len(pos_queue) >= ZIGZAG_WINDOW:
                if detect_zigzag(pos_queue):
                    alarm_type = "ZIGZAG"
                elif detect_stillness(pos_queue):
                    alarm_type = "STILLNESS"

            # ── Відображення ──
            if alarm_type:
                label = f"ALARM [{alarm_type}] — {identified['name']}"
                color = (0, 0, 255)
                cv2.putText(frame, f"Call: {identified['phone']}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                # червона рамка
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
            elif identified:
                label = f"OK: {identified['name']}"
                color = (0, 255, 0)
        else:
            label = "No person"
            color = (128, 128, 128)

        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Queue: {len(pos_queue)}/{QUEUE_SIZE}",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("MemoRescue Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Моніторинг завершено.")


if __name__ == "__main__":
    run_monitor()
