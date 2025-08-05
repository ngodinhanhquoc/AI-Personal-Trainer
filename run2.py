import cv2
import numpy as np
import pandas as pd
import pickle
import warnings
import math
import os
import mediapipe as mp
import time
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.utils.loss as loss_mod
from utils_overlay import draw_feedback_overlay

warnings.filterwarnings('ignore')


class DFLoss(nn.Module):
    """Lớp DFLoss giả để tương thích ngược."""
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, preds, targets, *args, **kwargs):
        return torch.tensor(0.0, device=preds.device)
loss_mod.DFLoss = DFLoss
    
# --- CÁC THÀNH PHẦN CHUNG VÀ HÀM HỖ TRỢ ---

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

BICEP_IMPORTANT_LMS = ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "RIGHT_ELBOW", "LEFT_ELBOW", "RIGHT_WRIST", "LEFT_WRIST", "LEFT_HIP", "RIGHT_HIP"]
PLANK_IMPORTANT_LMS = ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"]
SQUAT_IMPORTANT_LMS = ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]
LUNGE_IMPORTANT_LMS = ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"]

def calculate_angle(point1, point2, point3):
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)
    angleInRad = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
    angleInDeg = np.abs(angleInRad * 180.0 / np.pi)
    return angleInDeg if angleInDeg <= 180 else 360 - angleInDeg

def calculate_distance(point1, point2):
    """Tính khoảng cách Euclide giữa 2 điểm."""
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def extract_keypoints_for_details(results, important_landmarks: list):
    landmarks = results.pose_landmarks.landmark
    data = []
    for lm in important_landmarks:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    return np.array(data).flatten().tolist()

def calculate_vector_angle(vector1, vector2):
    """Tính góc giữa hai vector (tính bằng độ)."""
    dot_product = np.dot(vector1, vector2)
    norms = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cos_angle = dot_product / norms
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# --- CÁC LỚP PHÂN TÍCH CHI TIẾT CHO TỪNG BÀI TẬP ---

class BicepPoseAnalysis:
    """Phân tích tư thế Bicep Curl và cung cấp feedback với tên lỗi cụ thể."""
    def __init__(self, side: str, stage_down_threshold: float, stage_up_threshold: float,
                 peak_contraction_threshold: float, loose_upper_arm_angle_threshold: float,
                 visibility_threshold: float):
        self.side = side
        self.stage_down_threshold = stage_down_threshold
        self.stage_up_threshold = stage_up_threshold
        self.peak_contraction_threshold = peak_contraction_threshold
        self.loose_upper_arm_angle_threshold = loose_upper_arm_angle_threshold
        self.visibility_threshold = visibility_threshold
        
        self.counter = 0
        self.stage = "down"
        self.peak_contraction_angle = 1000
        self.feedback = "GOOD" 

        self.is_visible = True
        self.shoulder = None
        self.elbow = None
        self.wrist = None

    def get_joints(self, landmarks):
        side = self.side.upper()
        shoulder_lm = landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value]
        elbow_lm = landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value]
        wrist_lm = landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value]
        joints_visibility = [shoulder_lm.visibility, elbow_lm.visibility, wrist_lm.visibility]
        self.is_visible = all(vis > self.visibility_threshold for vis in joints_visibility)
        if self.is_visible:
            self.shoulder = [shoulder_lm.x, shoulder_lm.y]
            self.elbow = [elbow_lm.x, elbow_lm.y]
            self.wrist = [wrist_lm.x, wrist_lm.y]
        return self.is_visible

    def analyze_pose(self, landmarks):
        if not self.get_joints(landmarks):
            return 
        
        self.feedback = "GOOD"
        bicep_curl_angle = int(calculate_angle(self.shoulder, self.elbow, self.wrist))
        shoulder_projection = [self.shoulder[0], 1]
        ground_upper_arm_angle = int(calculate_angle(self.elbow, self.shoulder, shoulder_projection))

        if bicep_curl_angle > self.stage_down_threshold:
            if self.stage == "up":
                if self.peak_contraction_angle >= self.peak_contraction_threshold:
                    self.feedback = "Peak contraction" 
                self.peak_contraction_angle = 1000
            self.stage = "down"
        elif bicep_curl_angle < self.stage_up_threshold and self.stage == "down":
            self.stage = "up"
            self.counter += 1
        
        if ground_upper_arm_angle > self.loose_upper_arm_angle_threshold:
            self.feedback = "Fix elbow"

        if self.stage == "up" and bicep_curl_angle < self.peak_contraction_angle:
            self.peak_contraction_angle = bicep_curl_angle

class PlankPoseAnalysis:
    """Phân tích tư thế Plank sử dụng model."""
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.current_status = "Model not loaded"
        self.confidence = 0.0

    def analyze_pose(self, results):
        if not self.model or not self.scaler: return self.current_status, self.confidence
        row = extract_keypoints_for_details(results, PLANK_IMPORTANT_LMS)
        X = np.array([row])
        X_scaled = self.scaler.transform(X)
        predicted_class = self.model.predict(X_scaled)[0]
        prediction_probability = self.model.predict_proba(X_scaled)[0]
        class_map = {0: "Correct", 1: "High back", 2: "Low back"}
        self.current_status = class_map.get(predicted_class, "Unknown")
        self.confidence = round(np.max(prediction_probability), 2)
        return self.current_status, self.confidence

class SquatPoseAnalysis:
    """Phân tích tư thế Squat, bao gồm đếm reps và kiểm tra vị trí chân/đầu gối."""
    def __init__(self, model, prob_threshold_down=0.5, prob_threshold_up=0.5):
        self.model, self.prob_threshold_down, self.prob_threshold_up = model, prob_threshold_down, prob_threshold_up
        self.counter, self.stage, self.class_indices = 0, "up", None
        self.foot_placement, self.knee_placement = "Unknown", "Unknown"

    def _get_class_indices(self):
        if self.class_indices is None and self.model:
            try:
                down_idx, up_idx = np.where(self.model.classes_ == "down")[0][0], np.where(self.model.classes_ == "up")[0][0]
                self.class_indices = {"down": down_idx, "up": up_idx}
            except Exception: self.class_indices = {"down": 0, "up": 1}

    def _analyze_counting(self, results):
        self._get_class_indices()
        row = extract_keypoints_for_details(results, SQUAT_IMPORTANT_LMS)
        X = np.array([row])
        try: prediction_probabilities = self.model.predict_proba(X)[0]
        except Exception: prediction_probabilities = self.model.predict_proba(pd.DataFrame(X))[0]
        down_prob, up_prob = prediction_probabilities[self.class_indices["down"]], prediction_probabilities[self.class_indices["up"]]
        if self.stage == "up" and down_prob > self.prob_threshold_down: self.stage = "down"
        elif self.stage == "down" and up_prob > self.prob_threshold_up:
            self.stage, self.counter = "up", self.counter + 1

    def _calculate_distance(self, p1, p2): return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def _analyze_placement(self, results, f_thresholds, k_thresholds, v_threshold):
        landmarks = results.pose_landmarks.landmark
        req_lms = [mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if any(landmarks[lm.value].visibility < v_threshold for lm in req_lms):
            self.foot_placement, self.knee_placement = "UNK", "UNK"; return
        l_s, r_s = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y], [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        s_w = self._calculate_distance(l_s, r_s)
        l_f, r_f = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y], [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
        f_w = self._calculate_distance(l_f, r_f)
        f_s_ratio = round(f_w / s_w, 1)
        min_f, max_f = f_thresholds
        if min_f <= f_s_ratio <= max_f: self.foot_placement = "Correct"
        elif f_s_ratio < min_f: self.foot_placement = "Too tight"
        else: self.foot_placement = "Too wide"
        l_k, r_k = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y], [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        k_w = self._calculate_distance(l_k, r_k)
        k_f_ratio = round(k_w / f_w, 1) if f_w > 0 else 0
        stage_thresholds = k_thresholds.get(self.stage, k_thresholds.get("up"))
        min_k, max_k = stage_thresholds
        if min_k <= k_f_ratio <= max_k: self.knee_placement = "Correct"
        elif k_f_ratio < min_k: self.knee_placement = "Too tight"
        else: self.knee_placement = "Too wide"

    def analyze_pose(self, results, foot_shoulder_thresholds, knee_foot_thresholds, visibility_threshold):
        if not self.model: return None
        self._analyze_counting(results)
        self._analyze_placement(results, foot_shoulder_thresholds, knee_foot_thresholds, visibility_threshold)
        return {"counter": self.counter, "stage": self.stage, "foot_placement": self.foot_placement, "knee_placement": self.knee_placement}

class LungePoseAnalysis:
    """Phân tích tư thế Lunge sử dụng 2 model: stage và error detection."""
    def __init__(self, stage_model, error_model, scaler, 
                 angle_thresholds=[60, 135], 
                 back_angle_threshold=15,
                 prob_threshold=0.8):
        self.stage_model = stage_model
        self.error_model = error_model
        self.scaler = scaler
        
        self.angle_thresholds = angle_thresholds
        self.back_angle_threshold = back_angle_threshold
        self.prob_threshold = prob_threshold
        
        self.counter = 0
        self.current_stage = ""
        
        self.knee_angle_error = False
        self.knee_over_toe_error = False
        self.back_posture_error = False
        self.error_model_confidence = 0.0

    def analyze_pose(self, results):
        if not all([self.stage_model, self.error_model, self.scaler]) or not results.pose_landmarks:
            return None
        
        self.knee_angle_error = False
        self.knee_over_toe_error = False
        self.back_posture_error = False
        self.error_model_confidence = 0.0

        row = extract_keypoints_for_details(results, LUNGE_IMPORTANT_LMS)
        X = np.array([row])
        X_scaled = self.scaler.transform(X)

        stage_predicted_class = self.stage_model.predict(X_scaled)[0]
        stage_prediction_probabilities = self.stage_model.predict_proba(X_scaled)[0]
        stage_confidence = round(stage_prediction_probabilities.max(), 2)

        if stage_confidence >= self.prob_threshold:
            if stage_predicted_class == "I":
                self.current_stage = "init"
            elif stage_predicted_class == "M":
                self.current_stage = "mid"
            elif stage_predicted_class == "D":
                if self.current_stage in ["mid", "init"]:
                    self.counter += 1
                self.current_stage = "down"

        if self.current_stage == "down":
            self.knee_angle_error = self._check_knee_angles(results)
            self.back_posture_error = self._check_back_posture(results)
            
            error_predicted_class = self.error_model.predict(X_scaled)[0]
            error_prediction_probabilities = self.error_model.predict_proba(X_scaled)[0]
            self.error_model_confidence = round(error_prediction_probabilities.max(), 2)
            
            if error_predicted_class == "Error": 
                self.knee_over_toe_error = True

        return {
            "counter": self.counter,
            "stage": self.current_stage,
            "stage_confidence": stage_confidence,
            "knee_angle_error": self.knee_angle_error,
            "knee_over_toe_error": self.knee_over_toe_error,
            "back_posture_error": self.back_posture_error,
            "kot_confidence": self.error_model_confidence,
        }

    def _check_knee_angles(self, results):
        landmarks = results.pose_landmarks.landmark
        
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        right_angle = calculate_angle(right_hip, right_knee, right_ankle)

        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        left_angle = calculate_angle(left_hip, left_knee, left_ankle)

        right_error = not (self.angle_thresholds[0] <= right_angle <= self.angle_thresholds[1])
        left_error = not (self.angle_thresholds[0] <= left_angle <= self.angle_thresholds[1])
        
        return right_error or left_error

    def _check_back_posture(self, results):
        landmarks = results.pose_landmarks.landmark
        
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        
        shoulder_mid = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]
        hip_mid = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
        
        back_vector = [shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1]]
        vertical_vector = [0, -1]
        
        angle_deviation = calculate_vector_angle(back_vector, vertical_vector)
        
        return angle_deviation > self.back_angle_threshold

# --- BẮT ĐẦU TÍCH HỢP CLASS SITUPPOSEANALYSIS ---
class SitupPoseAnalysis:
    """
    Phân tích tư thế Situp với các tính năng:
    - Đếm số lần thực hiện (count)
    - Phát hiện 2 lỗi sai:
      1. Lỗi góc lưng (back_angle_error): Không nâng người lên đủ cao
      2. Lỗi chân không ổn định (leg_stability_error): Chân di chuyển quá nhiều
    - **Cải tiến**: Tự động xác định bên cơ thể rõ hơn để tính toán,
      hoạt động tốt với video quay nghiêng.
    """
    
    def __init__(self, 
                 up_threshold=90,
                 down_threshold=120,
                 back_angle_max_on_up=85,
                 leg_movement_threshold=0.05, 
                 visibility_threshold=0.7):
        
        # Thông số cấu hình
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.back_angle_max_on_up = back_angle_max_on_up
        self.leg_movement_threshold = leg_movement_threshold
        self.visibility_threshold = visibility_threshold
        
        # Trạng thái đếm
        self.counter = 0
        self.stage = "down"
        
        # Trạng thái lỗi
        self.back_angle_error = False
        self.leg_stability_error = False
        
        # Lưu trữ vị trí chân
        self.previous_knee_positions = []
        self.knee_position_history_size = 10
        
        # Trạng thái visibility
        self.is_visible = False
        
    def _check_leg_stability_error(self, current_knee_pos):
        """Kiểm tra lỗi chân không ổn định."""
        if not current_knee_pos: return False
        
        self.previous_knee_positions.append(current_knee_pos)
        
        if len(self.previous_knee_positions) > self.knee_position_history_size:
            self.previous_knee_positions.pop(0)
        
        if len(self.previous_knee_positions) < 3: return False
        
        max_movement = 0
        for i in range(1, len(self.previous_knee_positions)):
            movement = calculate_distance(self.previous_knee_positions[i-1], self.previous_knee_positions[i])
            max_movement = max(max_movement, movement)
            
        return max_movement > self.leg_movement_threshold

    def analyze_pose(self, results):
        """
        Phân tích tư thế situp từ results của MediaPipe.
        """
        self.is_visible = False
        body_angle = 0
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 1. Xác định bên nào của cơ thể rõ hơn
            left_shoulder_vis = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
            left_hip_vis = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility
            left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
            
            right_shoulder_vis = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
            right_hip_vis = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
            right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

            left_vis_score = (left_shoulder_vis + left_hip_vis + left_knee_vis) / 3
            right_vis_score = (right_shoulder_vis + right_hip_vis + right_knee_vis) / 3
            
            shoulder, hip, knee = None, None, None
            
            if left_vis_score > self.visibility_threshold and left_vis_score >= right_vis_score:
                self.is_visible = True
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            elif right_vis_score > self.visibility_threshold:
                self.is_visible = True
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            # 2. Nếu một bên đủ rõ, tiến hành tính toán
            if self.is_visible:
                body_angle = calculate_angle(shoulder, hip, knee)
                
                if body_angle < self.up_threshold and self.stage == "down":
                    self.stage = "up"
                    self.counter += 1
                elif body_angle > self.down_threshold and self.stage == "up":
                    self.stage = "down"
                
                if self.stage == "up" and body_angle > self.back_angle_max_on_up:
                    self.back_angle_error = True
                else:
                    self.back_angle_error = False
                
                self.leg_stability_error = self._check_leg_stability_error(knee)
            else:
                 self.back_angle_error = False
                 self.leg_stability_error = False

        return {
            "counter": self.counter,
            "stage": self.stage,
            "back_angle_error": self.back_angle_error,
            "leg_stability_error": self.leg_stability_error,
            "is_visible": self.is_visible,
            "body_angle": round(body_angle, 1)
        }
    
    def reset_counter(self):
        """Reset bộ đếm về 0."""
        self.counter = 0
        self.stage = "down"
        self.previous_knee_positions = []
    
    def get_feedback_message(self):
        """Trả về thông báo feedback dựa trên lỗi hiện tại."""
        if self.back_angle_error and self.leg_stability_error:
            return "Sit up higher & Keep legs stable"
        elif self.back_angle_error:
            return "Sit up higher"
        elif self.leg_stability_error:
            return "Keep legs stable"
        elif self.is_visible:
            return "Good form"
        else:
            return "Body not fully visible"
# --- KẾT THÚC TÍCH HỢP CLASS SITUPPOSEANALYSIS ---


# --- LỚP ĐIỀU PHỐI CHÍNH ---

class ExerciseDetector:
    def __init__(self, model_paths):
        self.models, self.scalers = self._load_models_and_scalers(model_paths)
        self.current_exercise = "Detecting..."
        self.bicep_left = BicepPoseAnalysis(side="left", stage_down_threshold=120, stage_up_threshold=90, peak_contraction_threshold=60, loose_upper_arm_angle_threshold=45, visibility_threshold=0.65)
        self.bicep_right = BicepPoseAnalysis(side="right", stage_down_threshold=120, stage_up_threshold=90, peak_contraction_threshold=60, loose_upper_arm_angle_threshold=45, visibility_threshold=0.65)
        self.plank_analysis = PlankPoseAnalysis(self.models.get('plank'), self.scalers.get('plank'))
        self.squat_analysis = SquatPoseAnalysis(self.models.get('squat'), prob_threshold_down=0.5, prob_threshold_up=0.5)
        self.lunge_analysis = LungePoseAnalysis(stage_model=self.models.get('lunge_stage'), error_model=self.models.get('lunge_error'), scaler=self.scalers.get('lunge'), back_angle_threshold=10)
        self.situp_analysis = SitupPoseAnalysis(
            up_threshold=90,
            down_threshold=120,
            back_angle_max_on_up=80,
            leg_movement_threshold=0.05,
            visibility_threshold=0.6
        )

    def _load_models_and_scalers(self, paths):
        models, scalers = {}, {}
        for key, path in paths.items():
            try:
                with open(path, "rb") as f:
                    if 'model' in key: models[key.replace('_model', '')] = pickle.load(f)
                    elif 'scaler' in key: scalers[key.replace('_scaler', '')] = pickle.load(f)
            except FileNotFoundError: print(f"Warning: File not found at {path}"); models[key.replace('_model', '')], scalers[key.replace('_scaler', '')] = None, None
        return models, scalers

    def set_exercise_type(self, exercise_type: str):
        if self.current_exercise != exercise_type:
            print(f"Switching to: {exercise_type}"); self.current_exercise = exercise_type

    def analyze_exercise(self, results, **kwargs):
        if self.current_exercise == "Bicep Curl": return self._analyze_bicep(results)
        elif self.current_exercise == "Plank": return self._analyze_plank(results)
        elif self.current_exercise == "Squat": return self._analyze_squat(results, **kwargs)
        elif self.current_exercise == "Lunge": return self._analyze_lunge(results)  
        elif self.current_exercise == "Situp": return self._analyze_situp(results) 
        else: return None
    
    def _analyze_lunge(self, results):
        return self.lunge_analysis.analyze_pose(results)
    
    def _analyze_situp(self, results):
        return self.situp_analysis.analyze_pose(results)

    def _analyze_bicep(self, results):
        if not results.pose_landmarks:
            return None
        landmarks = results.pose_landmarks.landmark
        self.bicep_left.analyze_pose(landmarks)
        self.bicep_right.analyze_pose(landmarks)
        return {"bicep_left_analyzer": self.bicep_left, "bicep_right_analyzer": self.bicep_right}

    def _analyze_plank(self, results):
        status, confidence = self.plank_analysis.analyze_pose(results)
        return {"status": status, "confidence": confidence}

    def _analyze_squat(self, results, **kwargs):
        return self.squat_analysis.analyze_pose(results, **kwargs)

def main():
    VIDEO_PATH = '1607.mp4'  
    
    YOLO_MODEL_PATH = 'model/best.pt'
    CONFIDENCE_THRESHOLD = 0.25
    
    DETAIL_MODEL_PATHS = {
        'bicep_model': "model/bicep_KNN_model.pkl", 'bicep_scaler': "model/bicep_input_scaler.pkl",
        'plank_model': "model/plank_LR_model.pkl", 'plank_scaler': "model/plank_input_scaler.pkl",
        'squat_model': "model/squat_LR_model.pkl",
        'lunge_stage_model': "model/lunge_stage_SVC_model.pkl",
        'lunge_error_model': "model/lunge_err_LR_model.pkl",
        'lunge_scaler': "model/lunge_input_scaler.pkl"
    }
    
    # THIẾT LẬP YOLO (thêm sau các định nghĩa path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng thiết bị: {device}")
    
    yolo_model = YOLO(YOLO_MODEL_PATH)
    yolo_model.model.fuse()
    yolo_model.to(device)
    
    use_half = (device == 'cuda')
    if use_half:
        yolo_model.model.half()

    yolo_colors = [
        (255, 0, 0),    # Lớp 0 - Đỏ
        (0, 255, 0),    # Lớp 1 - Xanh lá
        (0, 0, 255),    # Lớp 2 - Xanh dương
        (255, 255, 0),  # Lớp 3 - Vàng
        (0, 255, 255),  # Lớp 4 - Cyan
    ]
    
    #####
    VISIBILITY_THRESHOLD = 0.6
    FOOT_SHOULDER_RATIO_THRESHOLDS, KNEE_FOOT_RATIO_THRESHOLDS = [1.2, 2.8], {"up": [0.5, 1.0], "down": [0.7, 1.1]}
    
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Lỗi: Không tìm thấy file YOLO model '{YOLO_MODEL_PATH}'.")
        return
    
    frame_counter = 0
    current_detected_exercise = "Detecting..."
    detection_confidence = 0.0

    detector = ExerciseDetector(DETAIL_MODEL_PATHS)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): 
        print(f"Lỗi: Không thể mở video/webcam tại '{VIDEO_PATH}'")
        return
    prev_time = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # SỬA VÒNG LẶP WHILE - THÊM YOLO DETECTION
        with torch.no_grad():  # Thêm context manager cho YOLO
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: 
                    print("Hết video hoặc không thể đọc frame."); break
                    
                
                
                # THÊM YOLO DETECTION TRƯỚC POSE DETECTION
                yolo_results = yolo_model.predict(
                    source=frame,
                    device=device,
                    half=use_half,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False
                )[0]
                
                if yolo_results.boxes:
                    # Lấy detection có confidence cao nhất
                    best_detection = None
                    best_conf = 0
                    
                    for box in yolo_results.boxes:
                        conf = float(box.conf[0])
                        if conf > best_conf:
                            best_conf = conf
                            cls_id = int(box.cls[0])
                            class_name = yolo_model.names[cls_id]
                            best_detection = class_name
                            print("YOLO model classes:", yolo_model.names)
                            
                    
                    if best_detection:
                        # Map class names từ YOLO sang exercise names
                        exercise_mapping = {
                            'bicep': 'Bicep Curl',
                            'lunge': 'Lunge', 
                            'plank': 'Plank',
                            'situp': 'Situp',
                            'squat': 'Squat'
                        }
                        
                        mapped_exercise = exercise_mapping.get(best_detection, best_detection.title())
                        if current_detected_exercise != mapped_exercise:
                            detector.set_exercise_type(mapped_exercise)
                            current_detected_exercise = mapped_exercise
                        detection_confidence = best_conf
                        
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))
                    
                    action = current_detected_exercise
                    confidence = detection_confidence
                    result = detector.analyze_exercise(results, 
                                                              foot_shoulder_thresholds=FOOT_SHOULDER_RATIO_THRESHOLDS, 
                                                              knee_foot_thresholds=KNEE_FOOT_RATIO_THRESHOLDS, 
                                                              visibility_threshold=VISIBILITY_THRESHOLD)

                    # --- GIAO DIỆN HIỂN THỊ ---
                    frame = draw_feedback_overlay(frame, detector, result, action, confidence)
                curr_time = time.time()
                if prev_time > 0: fps = 1 / (curr_time - prev_time); cv2.putText(image, f"FPS: {int(fps)}", (image.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                prev_time = curr_time
                
                # image = cv2.resize(image, (1080, 720))
                cv2.imshow('AI Exercise Coach', image)
                frame_counter += 1

                if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

def analyze_video_and_return_data_yolo(video_path, output_path="result/output_yolo.webm"):
    if not os.path.exists("result"):
        os.makedirs("result")  
    
    YOLO_MODEL_PATH = 'model/best.pt'
    CONFIDENCE_THRESHOLD = 0.25
    
    DETAIL_MODEL_PATHS = {
        'bicep_model': "model/bicep_KNN_model.pkl", 'bicep_scaler': "model/bicep_input_scaler.pkl",
        'plank_model': "model/plank_LR_model.pkl", 'plank_scaler': "model/plank_input_scaler.pkl",
        'squat_model': "model/squat_LR_model.pkl",
        'lunge_stage_model': "model/lunge_stage_SVC_model.pkl",
        'lunge_error_model': "model/lunge_err_LR_model.pkl",
        'lunge_scaler': "model/lunge_input_scaler.pkl"
    }
    
    # THIẾT LẬP YOLO (thêm sau các định nghĩa path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng thiết bị: {device}")
    
    yolo_model = YOLO(YOLO_MODEL_PATH)
    yolo_model.model.fuse()
    yolo_model.to(device)
    
    use_half = (device == 'cuda')
    if use_half:
        yolo_model.model.half()

    yolo_colors = [
        (255, 0, 0),    # Lớp 0 - Đỏ
        (0, 255, 0),    # Lớp 1 - Xanh lá
        (0, 0, 255),    # Lớp 2 - Xanh dương
        (255, 255, 0),  # Lớp 3 - Vàng
        (0, 255, 255),  # Lớp 4 - Cyan
    ]
    
    #####
    VISIBILITY_THRESHOLD = 0.6
    FOOT_SHOULDER_RATIO_THRESHOLDS, KNEE_FOOT_RATIO_THRESHOLDS = [1.2, 2.8], {"up": [0.5, 1.0], "down": [0.7, 1.1]}
    
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Lỗi: Không tìm thấy file YOLO model '{YOLO_MODEL_PATH}'.")
        return
    
    frame_counter = 0
    current_detected_exercise = "Detecting..."
    detection_confidence = 0.0
    result = None
    confidence = 0.0
    action = "DETECTING..."

    detector = ExerciseDetector(DETAIL_MODEL_PATHS)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print(f"Lỗi: Không thể mở video/webcam tại '{video_path}'")
        return
    # Ghi video dạng .webm cho Streamlit hiển thị
    output_path = output_path.replace(".mp4", ".webm")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'VP80'), fps, (1080, 720))

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    prev_time = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # SỬA VÒNG LẶP WHILE - THÊM YOLO DETECTION
        with torch.no_grad():  # Thêm context manager cho YOLO
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: 
                    print("Hết video hoặc không thể đọc frame."); break
                    
                
                
                # THÊM YOLO DETECTION TRƯỚC POSE DETECTION
                yolo_results = yolo_model.predict(
                    source=frame,
                    device=device,
                    half=use_half,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False
                )[0]
                
                if yolo_results.boxes:
                    # Lấy detection có confidence cao nhất
                    best_detection = None
                    best_conf = 0
                    
                    for box in yolo_results.boxes:
                        conf = float(box.conf[0])
                        if conf > best_conf:
                            best_conf = conf
                            cls_id = int(box.cls[0])
                            class_name = yolo_model.names[cls_id]
                            best_detection = class_name
                            print("YOLO model classes:", yolo_model.names)
                            
                    
                    if best_detection:
                        # Map class names từ YOLO sang exercise names
                        exercise_mapping = {
                            'bicep': 'Bicep Curl',
                            'lunge': 'Lunge', 
                            'plank': 'Plank',
                            'situp': 'Situp',
                            'squat': 'Squat'
                        }
                        
                        mapped_exercise = exercise_mapping.get(best_detection, best_detection.title())
                        if current_detected_exercise != mapped_exercise:
                            detector.set_exercise_type(mapped_exercise)
                            current_detected_exercise = mapped_exercise
                        detection_confidence = best_conf
                        
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))
                    
                    action = current_detected_exercise
                    confidence = detection_confidence
                    result = detector.analyze_exercise(results, 
                                                              foot_shoulder_thresholds=FOOT_SHOULDER_RATIO_THRESHOLDS, 
                                                              knee_foot_thresholds=KNEE_FOOT_RATIO_THRESHOLDS, 
                                                              visibility_threshold=VISIBILITY_THRESHOLD)

                    # --- GIAO DIỆN HIỂN THỊ ---
                try:
                    image = draw_feedback_overlay(image, detector, result, action, detection_confidence)
                except KeyError:
                    pass  
                curr_time = time.time()
                if prev_time > 0: fps = 1 / (curr_time - prev_time); cv2.putText(image, f"FPS: {int(fps)}", (image.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                prev_time = curr_time

                image = cv2.resize(image, (1080, 720))
                out.write(image)
                frame_counter += 1

    cap.release()
    out.release()
    return output_path
def analyze_webcam_yolo(video_path, output_path="result/output_yolo.webm"):
    if not os.path.exists("result"):
        os.makedirs("result")  
    
    YOLO_MODEL_PATH = 'model/best.pt'
    CONFIDENCE_THRESHOLD = 0.25
    
    DETAIL_MODEL_PATHS = {
        'bicep_model': "model/bicep_KNN_model.pkl", 'bicep_scaler': "model/bicep_input_scaler.pkl",
        'plank_model': "model/plank_LR_model.pkl", 'plank_scaler': "model/plank_input_scaler.pkl",
        'squat_model': "model/squat_LR_model.pkl",
        'lunge_stage_model': "model/lunge_stage_SVC_model.pkl",
        'lunge_error_model': "model/lunge_err_LR_model.pkl",
        'lunge_scaler': "model/lunge_input_scaler.pkl"
    }
    
    # THIẾT LẬP YOLO (thêm sau các định nghĩa path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng thiết bị: {device}")
    
    yolo_model = YOLO(YOLO_MODEL_PATH)
    yolo_model.model.fuse()
    yolo_model.to(device)
    
    use_half = (device == 'cuda')
    if use_half:
        yolo_model.model.half()

    yolo_colors = [
        (255, 0, 0),    # Lớp 0 - Đỏ
        (0, 255, 0),    # Lớp 1 - Xanh lá
        (0, 0, 255),    # Lớp 2 - Xanh dương
        (255, 255, 0),  # Lớp 3 - Vàng
        (0, 255, 255),  # Lớp 4 - Cyan
    ]
    
    #####
    VISIBILITY_THRESHOLD = 0.6
    FOOT_SHOULDER_RATIO_THRESHOLDS, KNEE_FOOT_RATIO_THRESHOLDS = [1.2, 2.8], {"up": [0.5, 1.0], "down": [0.7, 1.1]}
    
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Lỗi: Không tìm thấy file YOLO model '{YOLO_MODEL_PATH}'.")
        return
    
    frame_counter = 0
    current_detected_exercise = "Detecting..."
    detection_confidence = 0.0

    detector = ExerciseDetector(DETAIL_MODEL_PATHS)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print(f"Lỗi: Không thể mở video/webcam tại '{video_path}'")
        return
    # Ghi video dạng .webm cho Streamlit hiển thị
    output_path = output_path.replace(".mp4", ".webm")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'VP80'), fps, (1080, 720))

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    prev_time = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # SỬA VÒNG LẶP WHILE - THÊM YOLO DETECTION
        with torch.no_grad():  # Thêm context manager cho YOLO
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: 
                    print("Hết video hoặc không thể đọc frame."); break
                    
                
                
                # THÊM YOLO DETECTION TRƯỚC POSE DETECTION
                yolo_results = yolo_model.predict(
                    source=frame,
                    device=device,
                    half=use_half,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False
                )[0]
                
                if yolo_results.boxes:
                    # Lấy detection có confidence cao nhất
                    best_detection = None
                    best_conf = 0
                    
                    for box in yolo_results.boxes:
                        conf = float(box.conf[0])
                        if conf > best_conf:
                            best_conf = conf
                            cls_id = int(box.cls[0])
                            class_name = yolo_model.names[cls_id]
                            best_detection = class_name
                            print("YOLO model classes:", yolo_model.names)
                            
                    
                    if best_detection:
                        # Map class names từ YOLO sang exercise names
                        exercise_mapping = {
                            'bicep': 'Bicep Curl',
                            'lunge': 'Lunge', 
                            'plank': 'Plank',
                            'situp': 'Situp',
                            'squat': 'Squat'
                        }
                        
                        mapped_exercise = exercise_mapping.get(best_detection, best_detection.title())
                        if current_detected_exercise != mapped_exercise:
                            detector.set_exercise_type(mapped_exercise)
                            current_detected_exercise = mapped_exercise
                        detection_confidence = best_conf
                        
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))
                    
                    action = current_detected_exercise
                    confidence = detection_confidence
                    result = detector.analyze_exercise(results, 
                                                              foot_shoulder_thresholds=FOOT_SHOULDER_RATIO_THRESHOLDS, 
                                                              knee_foot_thresholds=KNEE_FOOT_RATIO_THRESHOLDS, 
                                                              visibility_threshold=VISIBILITY_THRESHOLD)

                    # --- GIAO DIỆN HIỂN THỊ ---
                frame = draw_feedback_overlay(frame, detector, result, action, confidence)
                curr_time = time.time()
                if prev_time > 0: fps = 1 / (curr_time - prev_time); cv2.putText(image, f"FPS: {int(fps)}", (image.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                prev_time = curr_time

                image = cv2.resize(image, (1080, 720))
                out.write(image)
                frame_counter += 1
        return frame



if __name__ == "__main__":
    main()