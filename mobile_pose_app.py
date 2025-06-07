import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tempfile

# === MediaPipe Setup ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

ANGLE_WEIGHTS = np.array([0.4, 0.4, 0.1, 0.1])

# === Utility Functions ===
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def extract_angles(landmarks):
    indices = mp_pose.PoseLandmark
    try:
        return [
            calculate_angle(landmarks[indices.LEFT_SHOULDER.value],
                            landmarks[indices.LEFT_ELBOW.value],
                            landmarks[indices.LEFT_WRIST.value]),
            calculate_angle(landmarks[indices.RIGHT_SHOULDER.value],
                            landmarks[indices.RIGHT_ELBOW.value],
                            landmarks[indices.RIGHT_WRIST.value]),
            calculate_angle(landmarks[indices.LEFT_HIP.value],
                            landmarks[indices.LEFT_KNEE.value],
                            landmarks[indices.LEFT_ANKLE.value]),
            calculate_angle(landmarks[indices.RIGHT_HIP.value],
                            landmarks[indices.RIGHT_KNEE.value],
                            landmarks[indices.RIGHT_ANKLE.value])
        ]
    except:
        return None

def weighted_distance(a1, a2, weights=ANGLE_WEIGHTS):
    diff = np.array(a1) - np.array(a2)
    return np.sqrt(np.sum((diff ** 2) * weights))

def feedback_label(score, threshold):
    if score < threshold:
        return "✅ Good", (0, 255, 0)
    elif score < threshold * 2:
        return "⚠️ Okay", (0, 255, 255)
    else:
        return "❌ Poor", (0, 0, 255)

def plot_live_chart(scores, threshold):
    fig, ax = plt.subplots()
    ax.plot(scores, label='Deviation', color='red')
    ax.axhline(y=threshold, color='green', linestyle='--', label='Good')
    ax.axhline(y=threshold * 2, color='orange', linestyle='--', label='Okay')
    ax.axhline(y=threshold * 3, color='red', linestyle='--', label='Poor')
    ax.set_ylim(0, max(threshold * 3.5, max(scores) if scores else 1))
    ax.set_title("Live Pose Deviation")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Deviation Score")
    ax.legend()
    st.pyplot(fig)

# === Streamlit UI ===
st.set_page_config(page_title="Pose Analysis", layout="wide", initial_sidebar_state="collapsed")

# Mobile-friendly layout styling
st.markdown("""
    <style>
        .block-container {
            max-width: 480px;
            margin: auto;
            padding: 1rem;
        }
        .phone-frame {
            position: relative;
            width: 360px;
            height: 640px;
            margin: auto;
            border: 16px solid black;
            border-radius: 36px;
            background-color: #000;
            box-shadow: 0 0 0 8px #888;
            overflow: hidden;
        }
        .phone-screen {
            width: 100%;
            height: 100%;
            background-color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📱 Pose Analysis in Phone Frame")
st.write("Track your posture in real time using your camera. Optionally upload a reference video for comparison.")

# Reference pose (optional)
ref_upload = st.file_uploader("Upload a reference pose video (optional)", type=["mp4"])
ref_sequence = []
if ref_upload:
    temp_ref = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_ref.write(ref_upload.read())
    cap = cv2.VideoCapture(temp_ref.name)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks:
            lm = [(l.x, l.y, l.z) for l in result.pose_landmarks.landmark]
            angles = extract_angles(lm)
            if angles:
                ref_sequence.append(angles)
    cap.release()

# Threshold calculation
if ref_sequence:
    diffs = [
        weighted_distance(ref_sequence[i+1], ref_sequence[i])
        for i in range(len(ref_sequence)-1)
    ]
    threshold = np.mean(diffs) + np.std(diffs)
else:
    threshold = 20  # fallback

# Start camera
if st.button("📷 Start Camera Analysis"):
    st.markdown('<div class="phone-frame"><div class="phone-screen">', unsafe_allow_html=True)
    stframe = st.empty()
    st.markdown('</div></div>', unsafe_allow_html=True)

    scores = deque(maxlen=50)
    cap = cv2.VideoCapture(0)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        feedback = "No Pose"
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = [(l.x, l.y, l.z) for l in results.pose_landmarks.landmark]
            test_angles = extract_angles(lm)
            if test_angles:
                if ref_sequence:
                    ref_idx = min(frame_idx, len(ref_sequence) - 1)
                    score = weighted_distance(test_angles, ref_sequence[ref_idx])
                else:
                    score = sum(test_angles) / len(test_angles)

                label, color = feedback_label(score, threshold)
                feedback = f"{label} | Score: {score:.1f}"
                scores.append(score)

        # Draw feedback text
        cv2.putText(frame, feedback, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color if 'color' in locals() else (0,0,0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

        frame_idx += 1
        if len(scores) > 3:
            st.subheader("📊 Live Deviation Score Chart")
            plot_live_chart(list(scores), threshold)

    cap.release()
