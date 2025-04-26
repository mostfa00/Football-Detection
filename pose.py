import cv2
from ultralytics import YOLO
import numpy as np
import pickle
import os

# ------------ Utility Functions ------------ #

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox):
    x1, _, x2, _ = bbox
    return int(x2 - x1)

# ------------ Pose Estimation for Player ID 5 ------------ #

class PlayerPoseEstimator:
    def __init__(self, pose_model_path="yolov8n-pose.pt"):
        self.pose_model = YOLO(pose_model_path)

    def estimate_pose(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        cropped = frame[y1:y2, x1:x2]

        results = self.pose_model.predict(cropped, conf=0.25, verbose=False)
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy[0].cpu().numpy()  # Shape: (17, 2)
                for x, y in keypoints:
                    cv2.circle(frame, (int(x + x1), int(y + y1)), 3, (0, 255, 0), -1)

        return frame

# ------------ Drawing Functions ------------ #

def draw_ellipse(frame, bbox, color, track_id):
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(int(width), int(0.35 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4
    )

    # ID Box
    if track_id is not None:
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = y2 + 15 - rectangle_height // 2
        y2_rect = y2 + 15 + rectangle_height // 2

        cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)
        x1_text = x1_rect + 12
        cv2.putText(frame, f"{track_id}", (x1_text, y1_rect + 15), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

    return frame

def draw_triangle(frame, bbox, color):
    y = int(bbox[1])
    x, _ = get_center_of_bbox(bbox)

    triangle = np.array([
        [x, y],
        [x - 10, y - 20],
        [x + 10, y - 20],
    ])
    cv2.drawContours(frame, [triangle], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle], 0, (0, 0, 0), 2)

    return frame

# ------------ Annotator Function ------------ #

def annotate_video_with_pose(video_frames, tracks, pose_estimator):
    output_frames = []

    for frame_num, frame in enumerate(video_frames):
        frame = frame.copy()
        players = tracks["players"][frame_num]
        referees = tracks["referees"][frame_num]
        ball = tracks["ball"][frame_num]

        for track_id, player in players.items():
            frame = draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id)

            if track_id == 5:
                frame = pose_estimator.estimate_pose(frame, player["bbox"])

        for _, ref in referees.items():
            frame = draw_ellipse(frame, ref["bbox"], (0, 255, 255), None)

        for _, ball_obj in ball.items():
            frame = draw_triangle(frame, ball_obj["bbox"], (0, 255, 0))

        output_frames.append(frame)

    return output_frames