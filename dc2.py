import cv2
import torch
import mediapipe as mp
import numpy as np
import pickle
import os
from ultralytics import YOLO
import ultralytics.utils.loss as loss_mod
import torch.nn as nn
from run import (
    BicepPoseAnalysis, PlankPoseAnalysis, SquatPoseAnalysis,
    LungePoseAnalysis, SitupPoseAnalysis
)

# === Cấu hình nguồn video: 0 (webcam) hoặc "path/to/video.mp4" ===
VIDEO_SOURCE = '1607.mp4'  # hoặc 0 để dùng webcam

# === Khởi tạo MediaPipe pose 1 lần ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# === Load các model & scaler ===
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

model_dir = "model"
plank_model = load_model(os.path.join(model_dir, "plank_LR_model.pkl"))
plank_scaler = load_model(os.path.join(model_dir, "plank_input_scaler.pkl"))
squat_model = load_model(os.path.join(model_dir, "squat_LR_model.pkl"))
lunge_stage_model = load_model(os.path.join(model_dir, "lunge_stage_SVC_model.pkl"))
lunge_error_model = load_model(os.path.join(model_dir, "lunge_err_LR_model.pkl"))
lunge_scaler = load_model(os.path.join(model_dir, "lunge_input_scaler.pkl"))

# === Khởi tạo các class phân tích ===
bicep_left = BicepPoseAnalysis("left", 120, 90, 60, 45, 0.65)
bicep_right = BicepPoseAnalysis("right", 120, 90, 60, 45, 0.65)
plank_analysis = PlankPoseAnalysis(plank_model, plank_scaler)
squat_analysis = SquatPoseAnalysis(squat_model)
lunge_analysis = LungePoseAnalysis(lunge_stage_model, lunge_error_model, lunge_scaler)
situp_analysis = SitupPoseAnalysis()

# === Load YOLO ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from ultralytics import YOLO
model = YOLO('model/best.pt')
model.model.fuse()
model.to(device)
if device == 'cuda':
    model.model.fuse()
model.to(device)
if device == 'cuda':
    model.model.half()

# === Capture video hoặc webcam ===
is_video_file = isinstance(VIDEO_SOURCE, str)
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Lỗi: Không thể mở nguồn video")
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0.0: fps = 25

if is_video_file:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('inference_output.mp4', fourcc, fps, (w, h))
else:
    out = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (w, h))
    image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    result_pose = pose.process(image_rgb)

    # === YOLO detect động tác ===
    results = model(frame_resized, device=device, half=(device=='cuda'), verbose=False)[0]
    labels = results.boxes
    if len(labels) > 0:
        cls_id = int(labels[0].cls[0])
        action = model.names[cls_id].lower()
        conf = float(labels[0].conf[0])
        cv2.putText(frame_resized, f"Detected: {action} ({conf:.2f})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # === MediaPipe landmarks ===
        if result_pose and result_pose.pose_landmarks:
            mp_drawing.draw_landmarks(frame_resized, result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if action == "curl":
                bicep_left.analyze_pose(result_pose.pose_landmarks.landmark)
                bicep_right.analyze_pose(result_pose.pose_landmarks.landmark)
                cv2.putText(frame_resized, f"L: {bicep_left.counter} {bicep_left.stage} {bicep_left.feedback}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame_resized, f"R: {bicep_right.counter} {bicep_right.stage} {bicep_right.feedback}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            elif action == "plank":
                status, confidence = plank_analysis.analyze_pose(result_pose)
                color = (0,255,0) if status == "Correct" else (0,0,255)
                cv2.putText(frame_resized, f"PLANK: {status} ({confidence})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            elif action == "situp":
                res = situp_analysis.analyze_pose(result_pose)
                cv2.putText(frame_resized, f"SITUP: {res['stage']} {res['counter']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame_resized, situp_analysis.get_feedback_message(), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            elif action == "squat":
                res = squat_analysis.analyze_pose(result_pose, [1.2, 2.8], {"up":[0.5,1.0], "down":[0.7,1.1]}, 0.6)
                if res:
                    cv2.putText(frame_resized, f"SQUAT: {res['stage']} {res['counter']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(frame_resized, f"FOOT: {res['foot_placement']}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if res['foot_placement']=="Correct" else (0,0,255), 2)

            elif action == "lunge":
                res = lunge_analysis.analyze_pose(result_pose)
                if res:
                    cv2.putText(frame_resized, f"LUNGE: {res['stage']} {res['counter']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(frame_resized, f"BACK: {'OK' if not res['back_posture_error'] else 'BAD'}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if not res['back_posture_error'] else (0,0,255), 2)

    # Ghi lại video nếu là file input
    if out:
        out.write(frame_resized)

    cv2.imshow("Exercise Detection", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
pose.close()

class YoloPoseEstimator:
    def __init__(self, model_path='best.pt'):
        from ultralytics import YOLO
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.model.to(self.device)
        if self.device == 'cuda':
            self.model.model.half()
        self.pose = mp.solutions.pose.Pose(static_image_mode=False,
                                           min_detection_confidence=0.5,
                                           min_tracking_confidence=0.5)

    def infer_and_draw(self, frame):
        frame_resized = cv2.resize(frame, (int(frame.shape[1]*0.5), int(frame.shape[0]*0.5)))
        image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        result_pose = self.pose.process(image_rgb)

        results = self.model(frame_resized, device=self.device, half=(self.device == 'cuda'), verbose=False)[0]
        labels = results.boxes
        if len(labels) > 0:
            cls_id = int(labels[0].cls[0])
            action = self.model.names[cls_id].lower()
            conf = float(labels[0].conf[0])
            cv2.putText(frame_resized,"Detected: {action} ({conf:.2f})", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            if result_pose and result_pose.pose_landmarks:
                mp_drawing.draw_landmarks(frame_resized, result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                if action == "curl":
                    bicep_left.analyze_pose(result_pose.pose_landmarks.landmark)
                    bicep_right.analyze_pose(result_pose.pose_landmarks.landmark)
                    cv2.putText(frame_resized, "L: {bicep_left.counter} {bicep_left.stage} {bicep_left.feedback}",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(frame_resized, "R: {bicep_right.counter} {bicep_right.stage} {bicep_right.feedback}",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                elif action == "plank":
                    status, confidence = plank_analysis.analyze_pose(result_pose)
                    color = (0,255,0) if status == "Correct" else (0,0,255)
                    cv2.putText(frame_resized, "PLANK: {status} ({confidence:.2f})", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                elif action == "situp":
                    res = situp_analysis.analyze_pose(result_pose)
                    cv2.putText(frame_resized, "SITUP: {res['stage']} {res['counter']}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(frame_resized, situp_analysis.get_feedback_message(), (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                elif action == "squat":
                    res = squat_analysis.analyze_pose(result_pose, [1.2, 2.8], {"up":[0.5,1.0], "down":[0.7,1.1]}, 0.6)
                    if res:
                        cv2.putText(frame_resized, f"SQUAT: {res['stage']} {res['counter']}", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        cv2.putText(frame_resized, "FOOT: {res['foot_placement']}", (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0,255,0) if res['foot_placement']=="Correct" else (0,0,255), 2)

                elif action == "lunge":
                    res = lunge_analysis.analyze_pose(result_pose)
                    if res:
                        cv2.putText(frame_resized, "LUNGE: {res['stage']} {res['counter']}", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        cv2.putText(frame_resized, "BACK: {'OK' if not res['back_posture_error'] else 'BAD'}", (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0,255,0) if not res['back_posture_error'] else (0,0,255), 2)

        return frame_resized

    