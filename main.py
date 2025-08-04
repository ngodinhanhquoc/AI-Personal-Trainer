import streamlit as st
import tempfile
import os
import shutil
from run import analyze_video_and_return_data, ExerciseDetector, extract_keypoints_for_sequence

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tensorflow as tf
import mediapipe as mp
import numpy as np
import cv2
from collections import deque

# --- CẤU HÌNH MODEL ---
MODEL_MAP = {
    "Model A": "model/best_model_2307.keras",
    "Model B": "model/best_model_fixed.keras"
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
        self.model = tf.keras.models.load_model(model_path)
        self.detector = ExerciseDetector(DETAIL_MODEL_PATHS)
        self.sequence = deque(maxlen=30)
        self.actions = np.array(['curl', 'lunge', 'plank', 'situp', 'squat'])
        self.threshold = 0.4
        self.frame_counter = 0
        self.last_action = "DETECTING..."
        self.last_confidence = 0.0
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

            # GIAO DIỆN HIỂN THỊ (từ run.py)
            # Vẽ nền overlay phía trên
            # 🔁 Overlay TỐI ƯU: nhanh hơn, nhẹ hơn, không che người

            # 🔁 Overlay tối ưu + dựng dọc góc trái + nhỏ gọn + feedback xuống dòng riêng

            h, w = image.shape[:2]

            # Giảm chiều rộng khung overlay
            box_width = 165  # <-- bạn có thể chỉnh dòng này để đổi chiều dài sau
            box_height = 200
            font_scale_title = 0.4
            font_scale_body = 0.35
            line_height = 18

            # Nền dọc bên trái
            cv2.rectangle(image, (0, 0), (box_width, box_height), (30, 30, 30), -1)

            # Dòng tiêu đề
            cv2.putText(image, f"MODE: {self.detector.current_exercise}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, (255, 255, 255), 1)
            cv2.putText(image, f"ACTION: {self.last_action.upper()} ({self.last_confidence:.2f})", (10, 20 + line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (200, 200, 200), 1)

            if self.detector.current_exercise == "Bicep Curl" and result:
                l, r = result['bicep_left_analyzer'], result['bicep_right_analyzer']
                cv2.putText(image, f"L: REP={l.counter} STG={l.stage}", (10, 20 + line_height*2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (255, 255, 255), 1)
                cv2.putText(image, f"   {l.feedback}", (10, 20 + line_height*3),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (0,255,0) if l.feedback=="GOOD" else (0,0,255), 1)

                cv2.putText(image, f"R: REP={r.counter} STG={r.stage}", (10, 20 + line_height*4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (255, 255, 255), 1)
                cv2.putText(image, f"   {r.feedback}", (10, 20 + line_height*5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (0,255,0) if r.feedback=="GOOD" else (0,0,255), 1)

            elif self.detector.current_exercise == "Plank" and result:
                status = result['status']
                conf = result['confidence']
                color = (0, 255, 0) if status == "Correct" else (0, 0, 255)
                cv2.putText(image, f"STATUS: {status} ({conf:.2f})", (10, 20 + line_height*2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, color, 1)

            elif self.detector.current_exercise == "Squat" and result:
                cv2.putText(image, f"REP={result['counter']} STG={result['stage']}", (10, 20 + line_height*2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (255,255,255), 1)
                cv2.putText(image, f"FOOT={result['foot_placement']}", (10, 20 + line_height*3),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (0,255,0) if result['foot_placement']=="Correct" else (0,0,255), 1)
                cv2.putText(image, f"KNEE={result['knee_placement']}", (10, 20 + line_height*4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (0,255,0) if result['knee_placement']=="Correct" else (0,0,255), 1)

            elif self.detector.current_exercise == "Lunge" and result:
                cv2.putText(image, f"REP={result['counter']} STG={result['stage']}({result['stage_confidence']:.2f})", (10, 20 + line_height*2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (255,255,255), 1)
                cv2.putText(image, f"KNEE: {'OK' if not result['knee_angle_error'] else 'BAD'}", (10, 20 + line_height*3),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (0,255,0) if not result['knee_angle_error'] else (0,0,255), 1)
                cv2.putText(image, f"TOE: {'OK' if not result['knee_over_toe_error'] else 'OVER'}", (10, 20 + line_height*4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (0,255,0) if not result['knee_over_toe_error'] else (0,0,255), 1)
                cv2.putText(image, f"BACK: {'OK' if not result['back_posture_error'] else 'BAD'}", (10, 20 + line_height*5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (0,255,0) if not result['back_posture_error'] else (0,0,255), 1)

            elif self.detector.current_exercise == "Situp" and result:
                msg = self.detector.situp_analysis.get_feedback_message()
                fb_color = (0, 255, 0) if msg == "Good form" else (0, 255, 255)
                cv2.putText(image, f"REP={result['counter']} STG={result['stage'].upper()}", (10, 20 + line_height*2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (255,255,255), 1)
                cv2.putText(image, f"BACK={'OK' if not result['back_angle_error'] else 'LOW'}", (10, 20 + line_height*3),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (0,255,0) if not result['back_angle_error'] else (0,0,255), 1)
                cv2.putText(image, f"LEG={'STABLE' if not result['leg_stability_error'] else 'UNSTABLE'}", (10, 20 + line_height*4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, (0,255,0) if not result['leg_stability_error'] else (0,0,255), 1)
                cv2.putText(image, f"FB: {msg}", (10, 20 + line_height*5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_body, fb_color, 1)



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
        result_path = analyze_video_and_return_data(video_path, output_path=output_path)

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



