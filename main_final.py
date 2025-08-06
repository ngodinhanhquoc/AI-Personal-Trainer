import streamlit as st
import tempfile
import os
import shutil
from run import analyze_video_and_return_data, ExerciseDetector, extract_keypoints_for_sequence, analyze_webcam

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tensorflow as tf
import mediapipe as mp
import numpy as np
import cv2
from collections import deque
from ultralytics import YOLO
import torch
from utils_overlay import draw_feedback_overlay
from run2 import analyze_video_and_return_data_yolo, analyze_webcam_yolo



# --- CẤU HÌNH MODEL ---
MODEL_MAP = {
    "Model A": "model/best_model_2307.keras",
    "Model B": "YOLO"  # dùng chuỗi "YOLO" để biết là dùng mô hình best.pt
}


DETAIL_MODEL_PATHS = {
    'bicep_model': "model/bicep_KNN_model.pkl", 'bicep_scaler': "model/bicep_input_scaler.pkl",
    'plank_model': "model/plank_LR_model.pkl", 'plank_scaler': "model/plank_input_scaler.pkl",
    'squat_model': "model/squat_LR_model.pkl",
    'lunge_stage_model': "model/lunge_stage_SVC_model.pkl",
    'lunge_error_model': "model/lunge_err_LR_model.pkl",
    'lunge_scaler': "model/lunge_input_scaler.pkl"
}

# --- VIDEO PROCESSOR CHO STREAMLIT-WEBRTC ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self, model_path):
        self.sequence = deque(maxlen=30)
        self.frame_counter = 0
        self.last_action = "DETECTING..."
        self.last_confidence = 0.0
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.actions = np.array(['curl', 'lunge', 'plank', 'situp', 'squat'])

        self.detector = ExerciseDetector(DETAIL_MODEL_PATHS)

        if model_path == "YOLO":
            # Dùng YOLO thay vì model keras
            self.use_yolo = True
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.yolo_model = YOLO("model/best.pt")
            self.yolo_model.to(self.device)
            self.yolo_model.model.fuse()
        else:
            # Dùng model keras như cũ
            import tensorflow as tf
            self.use_yolo = False
            self.model = tf.keras.models.load_model(model_path)

        self.threshold = 0.4

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            keypoints = extract_keypoints_for_sequence(results)
            self.sequence.append(keypoints)

            if self.use_yolo:
    # Nếu đang dùng YOLO (Model B)
                if self.frame_counter % 5 == 0:
                    yolo_result = self.yolo_model.predict(image, device=self.device, conf=0.25, verbose=False)[0]
                    if yolo_result.boxes:
                        best_box = max(yolo_result.boxes, key=lambda box: float(box.conf[0]))
                        class_id = int(best_box.cls[0])
                        class_name = self.yolo_model.names[class_id]

                        exercise_map = {
                            'bicep': 'Bicep Curl',
                            'lunge': 'Lunge',
                            'plank': 'Plank',
                            'situp': 'Situp',
                            'squat': 'Squat'
                        }
                        action = exercise_map.get(class_name, class_name.title())
                        if self.last_action != action:
                            self.detector.set_exercise_type(action)
                        self.last_action = action
                        self.last_confidence = float(best_box.conf[0])
            else:
                # Dùng model keras như cũ
                if len(self.sequence) == 30 and self.frame_counter % 5 == 0:
                    res = self.model.predict(np.expand_dims(list(self.sequence), axis=0), verbose=0)[0]
                    confidence = np.max(res)
                    if confidence > self.threshold:
                        action = self.actions[np.argmax(res)]
                        if self.last_action != action:
                            self.detector.set_exercise_type(action.replace("curl", "Bicep Curl").title())
                        self.last_action = action
                        self.last_confidence = confidence
                    else:
                        self.last_action = "DETECTING..."
                        self.last_confidence = confidence

            result = self.detector.analyze_exercise(
                results,
                foot_shoulder_thresholds=[1.2, 2.8],
                knee_foot_thresholds={"up": [0.5, 1.0], "down": [0.7, 1.1]},
                visibility_threshold=0.6
            )

            

            # 🔁 Overlay tối ưu + dựng dọc góc trái + nhỏ gọn + feedback xuống dòng riêng

            image = draw_feedback_overlay(image, self.detector, result, self.last_action, self.last_confidence)

            self.frame_counter += 1
            image = cv2.resize(image, (1080, 720))
            return image


# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Fitness Coach", layout="centered")
st.title("🏋️ AI Personal Trainer")

mode = st.radio("🎬 Chọn chế độ hoạt động:", ["📹 Webcam Realtime", "📤 Upload Video"])
selected_model_name = st.selectbox("🧠 Chọn mô hình:", list(MODEL_MAP.keys()))
selected_model_path = MODEL_MAP[selected_model_name]

# --- TẠO THƯ MỤC KẾT QUẢ THEO MODEL ---
if selected_model_name == "Model A":
    output_dir = "result_model_a"
else:
    output_dir = "result_model_b"

os.makedirs(output_dir, exist_ok=True)


# --- XỬ LÝ VIDEO UPLOAD ---
from datetime import datetime  # Thêm import này nếu chưa có

if mode == "📤 Upload Video":
    uploaded_file = st.file_uploader("🎥 Tải video bài tập (MP4)", type=["mp4"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

        st.video(video_path)
        st.info("⏳ Đang phân tích video...")

        # 🔸 Tạo tên file có timestamp để không bị ghi đè
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"output_{timestamp}.webm")

        # 🔸 Phân tích và lưu video đầu ra
        if selected_model_name == "Model A":
            result_path = analyze_video_and_return_data(video_path, output_path=output_path)
        else:
            result_path = analyze_video_and_return_data_yolo(video_path, output_path=output_path)


        if result_path and os.path.exists(result_path):
            st.success("✅ Phân tích xong! Xem video kết quả bên dưới.")
            st.video(result_path)
            with open(result_path, 'rb') as f:
                st.download_button("📥 Tải video kết quả", f, file_name=os.path.basename(result_path))
        else:
            st.error("❌ Có lỗi xảy ra khi phân tích video.")

# --- XỬ LÝ WEBCAM REALTIME ---
elif mode == "📹 Webcam Realtime":
    st.warning("⚠️ Hãy cho phép trình duyệt sử dụng webcam.")
    webrtc_streamer(
        key=f"realtime-{selected_model_name.lower().replace(' ', '-')}",
        video_processor_factory=lambda: VideoProcessor(selected_model_path),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# --- HIỂN THỊ TOÀN BỘ KẾT QUẢ VIDEO THEO MODEL ---

st.markdown("---")
st.subheader("📁 Kết quả đã lưu theo từng model")

def show_results_section(model_label, folder):
    st.markdown(f"### 📂 {model_label}")
    if not os.path.exists(folder):
        st.info("Chưa có kết quả nào.")
        return

    videos = sorted([f for f in os.listdir(folder) if f.endswith(".webm")], reverse=True)
    if not videos:
        st.info("Chưa có kết quả nào.")
    else:
        for vid in videos:
            st.video(os.path.join(folder, vid))

# ✅ Gọi cho từng model
show_results_section("Model A", "result_model_a")
show_results_section("Model B", "result_model_b")



