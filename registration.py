import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
import tempfile

# Ініціалізація MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)


def _calculate_angle(a, b, c):
    """Обчислює кут у точці b за трьома landmarks"""
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
    """Витягує вектор ознак ходьби з відео (7-елементний вектор)"""
    cap = cv2.VideoCapture(video_path)
    all_features = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 5 == 0:
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


RECORD_SECONDS = 10  # тривалість запису з камери


def record_from_camera(duration=RECORD_SECONDS):
    """Записує відео з вебкамери протягом duration секунд, повертає шлях до файлу."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

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


def _save_profile(name, phone, video_path):
    """Спільна логіка: аналіз відео → збереження профілю."""
    signature = calculate_gait_signature(video_path)

    # прибираємо тимчасовий файл
    if os.path.exists(video_path):
        os.remove(video_path)

    if signature is None:
        st.error("The skeleton in the video could not be recognized. Try a different video.")
        return

    if not os.path.exists("database"):
        os.makedirs("database")

    user_data = {
        "name": name,
        "phone": phone,
        "gait_signature": signature,
    }
    with open(f"database/{name.replace(' ', '_')}.json", "w", encoding="utf-8") as f:
        json.dump(user_data, f, ensure_ascii=False, indent=4)

    st.success(f"Profile for {name} has been successfully created!")
    st.metric("Gait vector dimension", len(signature))


# --- ІНТЕРФЕЙС STREAMLIT ---
st.set_page_config(page_title="Registration MemoRescue", page_icon="👤")

st.title("Registration in MemoRescue system")
st.write("Upload user data to create a digital walking profile.")

name = st.text_input("Username")
guardian_phone = st.text_input("Phone number of the trusted person")

tab_upload, tab_record = st.tabs(["Upload video", "Record from camera"])

# ── Вкладка 1: завантаження файлу ──
with tab_upload:
    video_file = st.file_uploader(
        "Upload a walking video (baseline)", type=["mp4", "mov", "avi"]
    )
    upload_btn = st.button("Create profile (upload)", key="btn_upload")

    if upload_btn:
        if not name or not guardian_phone:
            st.warning("Please fill in the name and phone fields above.")
        elif not video_file:
            st.warning("Please upload a video first.")
        else:
            with st.spinner("Analyzing walking... wait."):
                temp_path = "temp_video.mp4"
                with open(temp_path, "wb") as f:
                    f.write(video_file.read())
                _save_profile(name, guardian_phone, temp_path)

# ── Вкладка 2: запис з камери ──
with tab_record:
    st.info(f"Click the button below — the camera will record for {RECORD_SECONDS} seconds. "
            "Walk naturally in front of the camera.")
    record_btn = st.button("Start recording", key="btn_record")

    if record_btn:
        if not name or not guardian_phone:
            st.warning("Please fill in the name and phone fields above.")
        else:
            with st.spinner(f"Recording {RECORD_SECONDS}s from camera..."):
                rec_path = record_from_camera(RECORD_SECONDS)
            if rec_path is None:
                st.error("Cannot open camera! Check that the webcam is connected.")
            else:
                with st.spinner("Analyzing walking... wait."):
                    _save_profile(name, guardian_phone, rec_path)