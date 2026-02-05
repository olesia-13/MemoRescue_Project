import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import os

# Ініціалізація MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)


def calculate_gait_signature(video_path):
    """Витягує вектор ознак ходьби з відео"""
    cap = cv2.VideoCapture(video_path)
    signatures = []

    # Тимчасово обробляємо кожні 5 кадрів для швидкості
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 5 == 0:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Приклад ознак (можна розширити):
                # 1. Відстань між щиколотками (точки 27 і 28)
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                dist = np.sqrt((left_ankle.x - right_ankle.x) ** 2 + (left_ankle.y - right_ankle.y) ** 2)

                # 2. Висота людини (для нормалізації) - від носа до щиколотки
                nose = landmarks[mp_pose.PoseLandmark.NOSE]
                height = np.abs(nose.y - left_ankle.y)

                # Зберігаємо відношення (нормалізована ознака)
                if height > 0:
                    signatures.append(dist / height)

        frame_count += 1

    cap.release()

    if len(signatures) > 0:
        # Повертаємо середнє значення як "відбиток" (спрощено для MVP)
        return float(np.mean(signatures))
    return None


# --- ІНТЕРФЕЙС STREAMLIT ---
st.set_page_config(page_title="Registration MemoRescue", page_icon="👤")

st.title("Registration in MemoRescue system")
st.write("Upload user data to create a digital walking profile.")

with st.form("registration_form"):
    name = st.text_input("Username")
    guardian_phone = st.text_input("Phone number of the trusted person")
    video_file = st.file_uploader("Upload a walking video (baseline)", type=['mp4', 'mov', 'avi'])

    submit = st.form_submit_button("Create a profile")

if submit:
    if name and guardian_phone and video_file:
        with st.spinner("Analyzing walking... wait."):
            # Зберігаємо відео тимчасово
            temp_path = "temp_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(video_file.read())

            # Крок 1 і 2: Отримуємо відбиток
            signature = calculate_gait_signature(temp_path)

            if signature:
                # Зберігаємо в базу даних (папка database має бути створена)
                if not os.path.exists("database"):
                    os.makedirs("database")

                user_data = {
                    "name": name,
                    "phone": guardian_phone,
                    "gait_signature": signature
                }

                with open(f"database/{name.replace(' ', '_')}.json", "w", encoding='utf-8') as f:
                    json.dump(user_data, f, ensure_ascii=False, indent=4)

                st.success(f"Profile for {name} has been successfully created!")
                st.metric("Digital signature of walking", round(signature, 4))
                os.remove(temp_path)  # видаляємо тимчасовий файл
            else:
                st.error("The skeleton in the video could not be recognized. Try a different video.")
    else:
        st.warning("Please fill in all fields and upload the video.")