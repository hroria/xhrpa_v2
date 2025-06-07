import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tempfile

# === Config ===
st.set_page_config(page_title="Mobile Pose Analyzer", layout="centered")

# === Style ===
st.markdown("""
<style>
.block-container {
    max-width: 480px;
    margin: auto;
}
.phone-frame {
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
    display: flex;
    align-items: center;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

st.title("📱 Mobile Pose Analysis")
st.markdown("Capture a photo and get pose feedback. Optionally compare to a reference pose.")

# === MediaPipe Setup ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# === Helper Functions ===
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

def weighted_distance(a1, a2):
    weights = np.array([0.4, 0.4, 0.1, 0.1])
    diff = np.array(a1) - np.array(a2)
    return np.sqrt(np.sum((diff ** 2) * weights))

def feedback_label(score, threshold=20):
    if score < threshold:
        return "✅ Good", (0, 255, 0)
    elif score < threshold * 2:
        return "⚠️ Okay", (0, 255, 255)
    else:
        return "❌ Poor", (255, 0, 0)

# === Reference Pose ===
ref_angles = None
ref_img = st.file_uploader("Upload a reference pose image (optional)", type=["jpg", "jpeg", "png"])
if ref_img:
    img = Image.open(ref_img).convert("RGB")
    frame = np.array(img)
    result = pose.process(frame)
    if result.pose_landmarks:
        lm = [(l.x, l.y, l.z) for l in result.pose_landmarks.landmark]
        ref_angles = extract_angles(lm)
        st.image(img, caption="Reference Pose", use_column_width=True)

# === Camera Input (Mobile-Compatible) ===
img_file = st.camera_input("Capture a pose photo")

if img_file:
    # Convert to OpenCV image
    img = Image.open(img_file).convert("RGB")
    frame = np.array(img)
    result = pose.process(frame)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = [(l.x, l.y, l.z) for l in result.pose_landmarks.landmark]
        angles = extract_angles(lm)

        if angles:
            if ref_angles:
                score = weighted_distance(angles, ref_angles)
            else:
                score = np.std(angles)  # fallback scoring

            label, color = feedback_label(score)
            cv2.putText(frame, f"{label} | Score: {score:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(frame, "Pose not complete", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Display inside phone frame
        st.markdown('<div class="phone-frame"><div class="phone-screen">', unsafe_allow_html=True)
        st.image(frame, channels="RGB", use_column_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    else:
        st.warning("No pose detected. Try a clearer image.")