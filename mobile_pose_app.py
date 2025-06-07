import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# === Setup ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# === Angle Calculation Utilities ===
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def extract_joint_angles(landmarks):
    try:
        indices = mp_pose.PoseLandmark
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
    except Exception as e:
        return None

# === Streamlit UI ===
st.set_page_config(page_title="📱 Mobile Pose Analyzer", layout="centered")

st.title("📷 Mobile Pose Analyzer")
st.write("Take a photo with your phone camera to analyze your posture.")

# === Camera Input ===
image_data = st.camera_input("Capture your pose")

if image_data:
    image = Image.open(image_data)
    frame = np.array(image)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    if results.pose_landmarks:
        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        st.image(frame, channels="RGB", caption="Detected Pose with Landmarks")

        # Extract and display angles
        lm = [(l.x, l.y, l.z) for l in results.pose_landmarks.landmark]
        joint_angles = extract_joint_angles(lm)

        if joint_angles:
            st.success("🎯 Detected Joint Angles")
            st.write(f"**Left Arm Angle**: {joint_angles[0]:.1f}°")
            st.write(f"**Right Arm Angle**: {joint_angles[1]:.1f}°")
            st.write(f"**Left Leg Angle**: {joint_angles[2]:.1f}°")
            st.write(f"**Right Leg Angle**: {joint_angles[3]:.1f}°")

            if max(joint_angles) - min(joint_angles) < 30:
                st.info("✅ Symmetrical form")
            else:
                st.warning("⚠️ Check for asymmetry between sides.")
        else:
            st.warning("Could not calculate joint angles.")
    else:
        st.error("No pose detected. Try improving lighting or body visibility.")
