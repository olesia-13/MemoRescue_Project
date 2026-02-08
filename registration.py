import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
import tempfile
import base64

# --- Ініціалізація MediaPipe (Твій оригінал) ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)


def _calculate_angle(a, b, c):
    """Обчислює кут у точці b за трьома landmarks (у радіанах)"""
    va = np.array([a.x - b.x, a.y - b.y])
    vc = np.array([c.x - b.x, c.y - b.y])
    cos_angle = np.dot(va, vc) / (np.linalg.norm(va) * np.linalg.norm(vc) + 1e-6)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _extract_frame_features(landmarks):
    """Витягує вектор ознак з одного кадру (7 ознак)"""
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    # Використовуємо зріст як коефіцієнт нормалізації
    height = abs(nose.y - l_ankle.y)
    if height < 0.01:
        return None

    ankle_dist = np.sqrt((l_ankle.x - r_ankle.x) ** 2 + (l_ankle.y - r_ankle.y) ** 2) / height
    l_knee_angle = _calculate_angle(l_hip, l_knee, l_ankle)
    r_knee_angle = _calculate_angle(r_hip, r_knee, r_ankle)
    l_hip_angle = _calculate_angle(l_shoulder, l_hip, l_knee)
    r_hip_angle = _calculate_angle(r_shoulder, r_hip, r_knee)
    shoulder_w = np.sqrt((l_shoulder.x - r_shoulder.x) ** 2 + (l_shoulder.y - r_shoulder.y) ** 2) / height
    step_h = abs(l_ankle.y - r_ankle.y) / height

    return [ankle_dist, l_knee_angle, r_knee_angle, l_hip_angle, r_hip_angle, shoulder_w, step_h]


def calculate_gait_signature(video_path):
    """Аналізує відео та повертає усереднений вектор ходьби"""
    cap = cv2.VideoCapture(video_path)
    all_features = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 5 == 0:  # Обробляємо кожен 5-й кадр для швидкості
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if results.pose_landmarks:
                feats = _extract_frame_features(results.pose_landmarks.landmark)
                if feats is not None:
                    all_features.append(feats)
        frame_count += 1

    cap.release()
    if len(all_features) > 0:
        return np.mean(all_features, axis=0).tolist()
    return None


def record_from_camera(duration=10):
    """Записує відео з вебкамери"""
    # Спроба відкрити камеру з DirectShow для уникнення зависання
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Запасний варіант

    if not cap.isOpened():
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20.0

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

    start = time.time()
    # Контейнер для виводу відео, щоб сайт не "мерз"
    video_placeholder = st.empty()

    while time.time() - start < duration:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Recording...")

    writer.release()
    cap.release()
    video_placeholder.empty()
    return tmp_path


def _save_profile(name, phone, chat_id, video_path):
    """Аналізує відео та зберігає дані в базу (папка database)"""
    signature = calculate_gait_signature(video_path)

    if os.path.exists(video_path):
        os.remove(video_path)

    if signature is None:
        st.error("Не вдалося розпізнати людину на відео. Спробуйте інше відео.")
        return

    if not os.path.exists("database"):
        os.makedirs("database")

    user_data = {
        "name": name,
        "phone": phone,
        "chat_id": chat_id,
        "gait_signature": signature,
    }

    file_name = f"database/{name.replace(' ', '_')}.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(user_data, f, ensure_ascii=False, indent=4)

    st.success(f"Профіль для {name} успішно створено!")
    st.info(f"Сповіщення будуть надсилатися на Telegram ID: {chat_id}")


def get_base64(bin_file):
    """Конвертує зображення в base64 для CSS"""
    if not os.path.exists(bin_file): return ""
    with open(bin_file, 'rb') as f:
        return base64.b64encode(f.read()).decode()


# --- Дизайн (Твій стиль) ---
st.set_page_config(page_title="MemoRescue", layout="centered")
bg_path = os.path.join("assets", "background.png")
bin_str = get_base64(bg_path)

st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover; background-position: center; background-attachment: fixed;
    }}
    header, footer {{visibility: hidden !important;}}
    .st-emotion-cache-1y4p8pa {{
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 40px !important; padding: 40px 60px !important;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1) !important;
    }}
    h1, p, label {{ color: white !important; }}
    h1 {{ font-size: 42px !important; font-weight: 700 !important; }}
    div[data-baseweb="input"] {{
        background-color: rgba(255, 255, 255, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
    }}
    input {{ color: black !important; -webkit-text-fill-color: black !important; }}
    input::placeholder {{ color: #666666 !important; opacity: 1 !important; -webkit-text-fill-color: #666666 !important; }}
    .stButton button {{
        background: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
        border-radius: 14px !important; color: white !important; font-weight: 600 !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{ background: transparent !important; }}
    .stTabs [data-baseweb="tab"] {{ color: rgba(255, 255, 255, 0.6) !important; }}
    .stTabs [aria-selected="true"] {{ color: white !important; border-bottom: 2px solid white !important; }}
    </style>
""", unsafe_allow_html=True)

st.title("Registration in MemoRescue system")
st.write("Upload user data to create a digital walking profile.")

name = st.text_input("Username", placeholder="Enter name")
guardian_phone = st.text_input("Phone number of the trusted person", placeholder="+380...")
guardian_chat_id = st.text_input("Telegram Chat ID of the guardian", placeholder="12345678")

tab_upload, tab_record = st.tabs(["Upload video", "Record from camera"])

with tab_upload:
    video_file = st.file_uploader("Select file (mp4, mov, avi)", type=["mp4", "mov", "avi"])
    if st.button("Create a profile", key="btn_upload"):
        if name and guardian_phone and guardian_chat_id and video_file:
            with st.spinner("Analyzing..."):
                temp_path = "temp_video.mp4"
                with open(temp_path, "wb") as f: f.write(video_file.read())
                _save_profile(name, guardian_phone, guardian_chat_id, temp_path)
        else:
            st.warning("Будь ласка, заповніть всі поля.")

with tab_record:
    if st.button("Start camera session", key="btn_record"):
        if name and guardian_phone and guardian_chat_id:
            with st.spinner("Recording... Check your camera!"):
                rec_path = record_from_camera(10)
            if rec_path:
                with st.spinner("Analyzing gait signature..."):
                    _save_profile(name, guardian_phone, guardian_chat_id, rec_path)
            else:
                st.error("Камера не відповідає. Перевірте дозволи браузера або чи не зайнята вона іншою програмою.")
        else:
            st.warning("Будь ласка, заповніть всі поля.")