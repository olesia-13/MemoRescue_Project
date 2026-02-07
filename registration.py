import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
import tempfile

# --- Ініціалізація MediaPipe ---
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
    cap = cv2.VideoCapture(0)
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
    while time.time() - start < duration:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)

    writer.release()
    cap.release()
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

    # Формуємо дані профілю, включаючи Chat ID
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

# --- ІНТЕРФЕЙС STREAMLIT ---
st.set_page_config(page_title="MemoRescue Registration", page_icon="👤")

st.title("👤 Реєстрація в системі MemoRescue")
st.write("Заповніть дані та створіть цифровий профіль ходьби.")

# Поля введення
name = st.text_input("Ім'я користувача (підопічного)")
guardian_phone = st.text_input("Номер телефону опікуна (для відображення)")
guardian_chat_id = st.text_input("Telegram Chat ID опікуна (дізнайтеся через @userinfobot)")

# Вкладки для відео
tab_upload, tab_record = st.tabs(["Завантажити відео", "Записати з камери"])

with tab_upload:
    video_file = st.file_uploader("Виберіть файл (mp4, mov, avi)", type=["mp4", "mov", "avi"])
    if st.button("Створити профіль (завантаження)", key="btn_upload"):
        if not name or not guardian_phone or not guardian_chat_id:
            st.warning("Будь ласка, заповніть всі поля вище.")
        elif not video_file:
            st.warning("Будь ласка, спочатку завантажте відео.")
        else:
            with st.spinner("Аналіз ходьби... зачекайте."):
                temp_path = "temp_video.mp4"
                with open(temp_path, "wb") as f:
                    f.write(video_file.read())
                _save_profile(name, guardian_phone, guardian_chat_id, temp_path)

with tab_record:
    st.info("Після натискання кнопки камера записуватиме 10 секунд. Пройдіться перед нею природною ходою.")
    if st.button("Почати запис з камери", key="btn_record"):
        if not name or not guardian_phone or not guardian_chat_id:
            st.warning("Будь ласка, заповніть всі поля вище.")
        else:
            with st.spinner("Запис..."):
                rec_path = record_from_camera(10)
            if rec_path:
                with st.spinner("Аналіз ходьби..."):
                    _save_profile(name, guardian_phone, guardian_chat_id, rec_path)
            else:
                st.error("Камера не знайдена!")
