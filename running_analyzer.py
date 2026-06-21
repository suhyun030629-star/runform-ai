from __future__ import annotations

from pathlib import Path
from typing import Callable
import math

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


BEST_CONFIG = {
    "static_image_mode": False,
    "model_complexity": 2,
    "smooth_landmarks": True,
    "enable_segmentation": False,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}
MP_POSE = mp.solutions.pose
POSE_INSTANCE = None


def midpoint(a, b):
    result = {"x": (a["x"] + b["x"]) / 2, "y": (a["y"] + b["y"]) / 2}
    if "z" in a and "z" in b:
        result["z"] = (a["z"] + b["z"]) / 2
    return result


def initialize_pose_model():
    """macOS에서는 서버 시작 메인 스레드에서 한 번만 초기화합니다."""
    global POSE_INSTANCE
    if POSE_INSTANCE is None:
        POSE_INSTANCE = MP_POSE.Pose(**BEST_CONFIG)
    return POSE_INSTANCE


def mp_landmarks_to_points(result, width, height):
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks.landmark

    def point(name):
        item = lm[getattr(MP_POSE.PoseLandmark, name).value]
        return {"x": item.x * width, "y": item.y * height, "visibility": item.visibility}

    world_lm = result.pose_world_landmarks.landmark if result.pose_world_landmarks else None

    def world_point(name):
        item = world_lm[getattr(MP_POSE.PoseLandmark, name).value]
        return {"x": item.x, "y": item.y, "z": item.z}

    names = [
        "RIGHT_ANKLE", "RIGHT_HEEL", "RIGHT_FOOT_INDEX", "RIGHT_KNEE", "RIGHT_HIP",
        "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", "LEFT_HEEL", "LEFT_FOOT_INDEX",
        "RIGHT_WRIST", "RIGHT_ELBOW", "RIGHT_SHOULDER", "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
    ]
    points = {name.lower(): point(name) for name in names}
    points["pelvis"] = midpoint(points["right_hip"], points["left_hip"])
    points["thorax"] = midpoint(points["right_shoulder"], points["left_shoulder"])
    points["head_top"] = point("NOSE")
    if world_lm:
        world = {name.lower(): world_point(name) for name in names}
        world["pelvis"] = midpoint(world["right_hip"], world["left_hip"])
        world["thorax"] = midpoint(world["right_shoulder"], world["left_shoulder"])
        points["world"] = world
    return points


def run_pose_on_video(video_path: str | Path, progress: Callable[[float], None] | None = None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("영상을 열 수 없습니다. mp4 형식의 파일인지 확인하세요.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    rows = []
    pose = initialize_pose_model()
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = pose.process(rgb)
        h, w = frame.shape[:2]
        points = mp_landmarks_to_points(result, w, h)
        rows.append({"frame_no": frame_no, "time_sec": frame_no / fps, "detected": points is not None, "points": points})
        frame_no += 1
        if progress and total_frames:
            progress(frame_no / total_frames * 0.65)
    cap.release()
    return pd.DataFrame(rows), fps


def point_xy(points, name):
    return points[name]["x"], points[name]["y"]


def joint_coordinates(points, name):
    """관절각은 화면 2D 좌표보다 카메라 각도에 강한 3D 월드 좌표를 우선 사용합니다."""
    if points.get("world") and name in points["world"]:
        point = points["world"][name]
        return point["x"], point["y"], point["z"]
    return point_xy(points, name)


def angle(a, b, c):
    v1, v2 = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denominator == 0:
        return np.nan
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / denominator, -1, 1)))


def knee_flexion(points, side):
    raw = angle(joint_coordinates(points, f"{side}_hip"), joint_coordinates(points, f"{side}_knee"), joint_coordinates(points, f"{side}_ankle"))
    return 180 - raw


def elbow_angle(points, side):
    return angle(joint_coordinates(points, f"{side}_shoulder"), joint_coordinates(points, f"{side}_elbow"), joint_coordinates(points, f"{side}_wrist"))


def trunk_lean(points):
    if points.get("world"):
        pelvis, thorax = joint_coordinates(points, "pelvis"), joint_coordinates(points, "thorax")
        horizontal = math.hypot(thorax[0] - pelvis[0], thorax[2] - pelvis[2])
        vertical = abs(thorax[1] - pelvis[1])
        return math.degrees(math.atan2(horizontal, vertical)) if vertical else np.nan
    pelvis, thorax = point_xy(points, "pelvis"), point_xy(points, "thorax")
    return abs(math.degrees(math.atan2(thorax[0] - pelvis[0], pelvis[1] - thorax[1])))


def body_height_px(points):
    head, ra, la = point_xy(points, "head_top"), point_xy(points, "right_ankle"), point_xy(points, "left_ankle")
    return math.dist(head, ((ra[0] + la[0]) / 2, (ra[1] + la[1]) / 2))


def build_feature_df(pred_df):
    rows = []
    for _, item in pred_df[pred_df["detected"]].iterrows():
        pts = item["points"]
        rows.append({
            "frame_no": item["frame_no"], "time_sec": item["time_sec"],
            "trunk_lean": trunk_lean(pts),
            "right_knee_flexion": knee_flexion(pts, "right"), "left_knee_flexion": knee_flexion(pts, "left"),
            "right_elbow_angle": elbow_angle(pts, "right"), "left_elbow_angle": elbow_angle(pts, "left"),
            "pelvis_x": pts["pelvis"]["x"], "pelvis_y": pts["pelvis"]["y"],
            "right_ankle_x": pts["right_ankle"]["x"], "right_ankle_y": pts["right_ankle"]["y"],
            "left_ankle_x": pts["left_ankle"]["x"], "left_ankle_y": pts["left_ankle"]["y"],
            "right_heel_y": pts["right_heel"]["y"], "right_foot_index_y": pts["right_foot_index"]["y"],
            "left_heel_y": pts["left_heel"]["y"], "left_foot_index_y": pts["left_foot_index"]["y"],
            "ankle_mid_y": (pts["right_ankle"]["y"] + pts["left_ankle"]["y"]) / 2,
            "body_height_px": body_height_px(pts),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("영상에서 자세를 인식하지 못했습니다. 전신이 나오고 밝은 영상을 사용하세요.")
    df["pelvis_to_ankle_ratio"] = (df["ankle_mid_y"] - df["pelvis_y"]) / df["body_height_px"]
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ("frame_no", "time_sec"):
            df[col] = df[col].rolling(5, center=True, min_periods=1).median()
    return df


def local_max_indices(values, min_gap):
    peaks, last = [], -10**9
    for i in range(1, len(values) - 1):
        if values[i] >= values[i - 1] and values[i] > values[i + 1]:
            if i - last >= min_gap:
                peaks.append(i); last = i
            elif peaks and values[i] > values[peaks[-1]]:
                peaks[-1] = i; last = i
    return peaks


def infer_motion_geometry(df):
    window = max(3, len(df) // 10)
    start, end = df[["pelvis_x", "pelvis_y"]].iloc[:window].median().to_numpy(), df[["pelvis_x", "pelvis_y"]].iloc[-window:].median().to_numpy()
    vector, motion_px = end - start, float(np.linalg.norm(end - start))
    start_h, end_h = df["body_height_px"].iloc[:window].median(), df["body_height_px"].iloc[-window:].median()
    scale_change = (end_h - start_h) / start_h if start_h else np.nan
    if motion_px < max(20, df["body_height_px"].median() * 0.10):
        return {"valid": False, "unit": None, "reliability": "unknown", "scale_change_pct": scale_change * 100}
    unit = vector / motion_px
    horizontal = abs(unit[0])
    reliability = "high" if horizontal >= .80 and abs(scale_change) <= .10 else "medium" if horizontal >= .50 and abs(scale_change) <= .25 else "low"
    return {"valid": True, "unit": unit, "reliability": reliability, "scale_change_pct": scale_change * 100}


FOOT_STRIKE_LABELS = {"forefoot": "전족 착지", "midfoot": "중족 착지", "rearfoot": "후족 착지"}


def classify_foot_strike(row, side):
    """착지 프레임에서 발끝과 뒤꿈치의 화면상 높이를 비교합니다.

    화면 y축은 아래쪽으로 갈수록 커집니다. 발끝이 더 낮으면 전족,
    뒤꿈치가 더 낮으면 후족, 차이가 작으면 중족으로 표시합니다.
    단일 2D 영상의 참고용 분류이므로 임상적 판정에는 사용하지 않습니다.
    """
    foot_pitch = (row[f"{side}_foot_index_y"] - row[f"{side}_heel_y"]) / row["body_height_px"]
    threshold = 0.015
    if foot_pitch > threshold:
        return "forefoot", foot_pitch
    if foot_pitch < -threshold:
        return "rearfoot", foot_pitch
    return "midfoot", foot_pitch


def landing_stats(df, right_peaks, left_peaks, user_height_cm, geometry):
    steps = []
    for side, peaks in (("right", right_peaks), ("left", left_peaks)):
        for idx in peaks:
            row = df.iloc[idx]
            if geometry["unit"] is None:
                distance_px = np.nan
            else:
                relative = np.array([row[f"{side}_ankle_x"] - row["pelvis_x"], row[f"{side}_ankle_y"] - row["pelvis_y"]])
                distance_px = max(0, float(np.dot(relative, geometry["unit"])))
            ratio = distance_px / row["body_height_px"] if row["body_height_px"] else np.nan
            strike_type, foot_pitch = classify_foot_strike(row, side)
            steps.append({"time_sec": row["time_sec"], "foot": side, "landing_knee_flexion": row[f"{side}_knee_flexion"], "overstride_ratio": ratio, "overstride_distance_cm": ratio * user_height_cm, "foot_strike": strike_type, "foot_pitch_ratio": foot_pitch})
    landing = pd.DataFrame(steps).sort_values("time_sec").reset_index(drop=True) if steps else pd.DataFrame()
    if landing.empty:
        return landing, {"landing_knee_flexion_mean": np.nan, "overstride_ratio_mean": np.nan, "overstride_distance_cm_mean": np.nan, "landing_knee_asymmetry": np.nan, "foot_strike_counts": {key: 0 for key in FOOT_STRIKE_LABELS}, "foot_strike_total": 0, "dominant_foot_strike": None, "foot_strike_reliable": False}
    right = landing.loc[landing.foot == "right", "landing_knee_flexion"].mean()
    left = landing.loc[landing.foot == "left", "landing_knee_flexion"].mean()
    counts = landing["foot_strike"].value_counts().reindex(FOOT_STRIKE_LABELS, fill_value=0).to_dict()
    dominant = max(counts, key=counts.get)
    return landing, {"landing_knee_flexion_mean": landing.landing_knee_flexion.mean(), "overstride_ratio_mean": landing.overstride_ratio.mean(), "overstride_distance_cm_mean": landing.overstride_distance_cm.mean(), "landing_knee_asymmetry": abs(right - left), "foot_strike_counts": counts, "foot_strike_total": len(landing), "dominant_foot_strike": dominant, "foot_strike_reliable": geometry["reliability"] in ("high", "medium") and len(landing) >= 4}


def make_feedback(s):
    feedback = []
    trunk = s["trunk_lean_mean"]
    if trunk < 5:
        feedback.append(("상체 기울기", "주의", f"평균 {trunk:.1f}도: 상체가 너무 수직입니다. 허리만 접지 말고, 발목에서부터 몸 전체를 5~10도 정도 살짝 앞으로 기울여 보세요."))
    elif trunk <= 10:
        feedback.append(("상체 기울기", "정상", f"평균 {trunk:.1f}도: 목표 범위 5~10도입니다. 시선은 전방에 두고 가슴을 편 상태를 유지하세요."))
    else:
        feedback.append(("상체 기울기", "주의", f"평균 {trunk:.1f}도: 상체가 다소 많이 숙여졌습니다. 복부에 힘을 주고 가슴을 살짝 펴서 골반 위에 몸통을 쌓아 보세요."))
    for side, label in (("right", "오른팔"), ("left", "왼팔")):
        value = s[f"{side}_elbow_mean"]
        if value < 80:
            message = f"평균 {value:.1f}도: 팔꿈치가 너무 많이 접혔습니다. 손이 몸통 옆을 자연스럽게 앞뒤로 지나가도록 팔을 조금 더 편하게 펴세요."
            verdict = "주의"
        elif value > 100:
            message = f"평균 {value:.1f}도: 팔꿈치가 너무 펴졌습니다. 팔꿈치를 약 90도로 유지하며 뒤로 가볍게 당기세요."
            verdict = "주의"
        else:
            message = f"평균 {value:.1f}도: 목표 범위 80~100도입니다. 어깨 힘은 빼고 팔만 리듬 있게 움직이세요."
            verdict = "정상"
        feedback.append((f"{label} 팔꿈치", verdict, message))
    ratio, distance = s["overstride_ratio_mean"], s["overstride_distance_cm_mean"]
    if not s["motion_detected"]:
        feedback.append(("오버스트라이드", "확인 필요", "골반의 이동 방향을 충분히 판별하지 못했습니다. 트레드밀·카메라 추적 영상은 참고용입니다."))
    elif s["motion_reliability"] == "low":
        feedback.append(("오버스트라이드", "참고용", f"대각선 이동 또는 사람 크기 변화가 {s['scale_change_pct']:.1f}% 감지되었습니다. 이동 벡터 기준 앞쪽 거리는 평균 {distance:.1f}cm입니다."))
    elif np.isnan(ratio):
        feedback.append(("오버스트라이드", "확인 필요", "착지 후보 프레임을 충분히 찾지 못했습니다. 전신이 보이는 영상을 사용하세요."))
    elif ratio >= .12:
        feedback.append(("오버스트라이드", "주의", f"착지 발이 골반보다 평균 {distance:.1f}cm 앞에 있습니다. 보폭을 조금 줄이고 발이 골반 바로 아래에 닿도록 케이던스를 살짝 높여 보세요."))
    else:
        feedback.append(("오버스트라이드", "정상", f"착지 발-골반 앞쪽 거리가 평균 {distance:.1f}cm로 크지 않습니다. 현재처럼 발을 몸 아래에 가깝게 착지하세요."))
    knee = s["landing_knee_flexion_mean"]
    if np.isnan(knee):
        feedback.append(("착지 무릎 굽힘각", "확인 필요", "착지 후보 프레임을 충분히 찾지 못했습니다."))
    else:
        if knee < 15:
            feedback.append(("착지 무릎 굽힘각", "주의", f"평균 {knee:.1f}도: 무릎이 너무 펴진 채 착지할 수 있습니다. 보폭을 줄이고 발을 골반 가까이에 디디며 무릎을 부드럽게 굽혀 충격을 흡수하세요."))
        elif knee > 25:
            feedback.append(("착지 무릎 굽힘각", "관찰", f"평균 {knee:.1f}도: 착지 때 무릎 굽힘이 큰 편입니다. 엉덩이가 과하게 내려가지 않도록 몸통을 안정적으로 유지하세요."))
        else:
            feedback.append(("착지 무릎 굽힘각", "정상", f"평균 {knee:.1f}도: 목표 범위 15~25도입니다. 현재처럼 부드럽게 충격을 흡수하세요."))
    strike = s["dominant_foot_strike"]
    if not strike:
        feedback.append(("착지 유형", "확인 필요", "전족·중족·후족을 분류할 착지 후보 프레임을 충분히 찾지 못했습니다."))
    else:
        counts = s["foot_strike_counts"]
        detail = " · ".join(f"{FOOT_STRIKE_LABELS[key]} {counts[key]}회" for key in FOOT_STRIKE_LABELS)
        reliability_note = "측면 고정 촬영이 아니거나 착지 후보가 적어 정확도가 낮을 수 있습니다. " if not s["foot_strike_reliable"] else ""
        feedback.append(("착지 유형", "참고용", f"주된 경향은 {FOOT_STRIKE_LABELS[strike]}입니다. {detail}. {reliability_note}착지 유형에는 절대적인 우열이 없으며, 통증·속도·훈련 목적과 함께 해석하세요."))
    vertical = s["pelvis_vertical_cm"]
    if not s["vertical_amplitude_reliable"]:
        feedback.append(("수직 진폭", "확인 필요", f"{vertical:.1f}cm로 과하게 계산되었거나 대각선·원근 영향이 큽니다. 카메라를 고정한 측면 촬영에서 더 정확합니다."))
    else:
        if vertical < 5:
            feedback.append(("수직 진폭", "관찰", f"{vertical:.1f}cm: 상하 움직임이 작습니다. 보폭이 너무 짧아 추진력이 줄지 않는지만 확인하세요."))
        elif vertical <= 10:
            feedback.append(("수직 진폭", "정상", f"{vertical:.1f}cm: 목표 범위 5~10cm입니다. 위아래로 튀지 않고 앞으로 나아가는 흐름이 좋습니다."))
        else:
            feedback.append(("수직 진폭", "주의", f"{vertical:.1f}cm: 위아래 움직임이 큰 편입니다. 발을 몸 가까이에 디디고 시선·골반 높이를 안정적으로 유지하세요."))
    cadence = s["cadence_spm"]
    if cadence < 170:
        feedback.append(("케이던스", "주의", f"{cadence:.1f} SPM: 목표보다 낮습니다. 보폭을 무리하게 늘리지 말고, 짧고 가벼운 스텝으로 리듬을 조금 빠르게 해 보세요."))
    elif cadence > 180:
        feedback.append(("케이던스", "관찰", f"{cadence:.1f} SPM: 목표보다 높습니다. 속도를 유지할 수 있는 편안한 리듬인지 호흡과 피로도를 함께 확인하세요."))
    else:
        feedback.append(("케이던스", "정상", f"{cadence:.1f} SPM: 목표 범위입니다. 현재 리듬을 유지하세요."))
    asymmetry = s["landing_knee_asymmetry"]
    if np.isnan(asymmetry):
        feedback.append(("좌우 무릎 대칭", "확인 필요", "양쪽 착지 후보를 충분히 찾지 못했습니다."))
    else:
        if asymmetry < 5:
            feedback.append(("좌우 무릎 대칭", "정상", f"양쪽 착지 시 평균 차이 {asymmetry:.1f}도입니다. 좌우 움직임이 균형적입니다."))
        else:
            feedback.append(("좌우 무릎 대칭", "주의", f"양쪽 착지 시 평균 차이 {asymmetry:.1f}도입니다. 한쪽 다리에 더 실리는 느낌이 있는지 확인하고, 피로한 쪽의 가동성을 점검해 보세요."))
    return pd.DataFrame(feedback, columns=["분석 요소", "판정", "피드백"])


def analyze_video(video_path, user_height_cm, progress=None):
    pred_df, fps = run_pose_on_video(video_path, progress)
    detection_rate = float(pred_df.detected.mean() * 100)
    if detection_rate < 50:
        raise ValueError(f"자세 인식률이 {detection_rate:.1f}%입니다. 전신이 나오도록 다시 촬영해 주세요.")
    if progress: progress(.72)
    df = build_feature_df(pred_df)
    geometry = infer_motion_geometry(df)
    right_peaks = local_max_indices(df.right_ankle_y.to_numpy(), max(2, int(fps * .20)))
    left_peaks = local_max_indices(df.left_ankle_y.to_numpy(), max(2, int(fps * .20)))
    landing_df, step = landing_stats(df, right_peaks, left_peaks, user_height_cm, geometry)
    vertical_cm = (np.nanpercentile(df.pelvis_to_ankle_ratio, 90) - np.nanpercentile(df.pelvis_to_ankle_ratio, 10)) * user_height_cm
    duration = df.time_sec.max() - df.time_sec.min()
    summary = {"detection_rate": detection_rate, "analyzed_frames": len(pred_df), "duration_sec": duration, "trunk_lean_mean": df.trunk_lean.mean(), "right_elbow_mean": df.right_elbow_angle.mean(), "left_elbow_mean": df.left_elbow_angle.mean(), "pelvis_vertical_cm": vertical_cm, "vertical_amplitude_reliable": bool(np.isfinite(vertical_cm) and vertical_cm <= 15 and geometry["reliability"] != "low"), "cadence_spm": (len(right_peaks) + len(left_peaks)) / duration * 60 if duration else np.nan, "motion_detected": geometry["valid"], "motion_reliability": geometry["reliability"], "scale_change_pct": geometry["scale_change_pct"]}
    summary.update(step)
    if progress: progress(1.0)
    return {"summary": summary, "feedback": make_feedback(summary), "features": df, "landings": landing_df, "predictions": pred_df}


def create_annotated_video(video_path, pred_df, output_path, progress=None):
    """관절 점과 연결선을 겹친 결과 영상을 저장합니다."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    lookup = pred_df.set_index("frame_no")["points"].to_dict()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    links = [("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("right_shoulder", "right_hip"), ("left_shoulder", "left_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("right_hip", "left_hip"), ("right_shoulder", "left_shoulder")]
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        points = lookup.get(frame_no)
        if points:
            for start, end in links:
                a, b = points[start], points[end]
                cv2.line(frame, (int(a["x"]), int(a["y"])), (int(b["x"]), int(b["y"])), (76, 217, 100), 3)
            for point in points.values():
                if isinstance(point, dict) and "visibility" in point:
                    cv2.circle(frame, (int(point["x"]), int(point["y"])), 4, (42, 102, 255), -1)
        cv2.putText(frame, "RunForm AI", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        frame_no += 1
        if progress and total:
            progress(frame_no / total)
    cap.release()
    writer.release()
    return Path(output_path)
