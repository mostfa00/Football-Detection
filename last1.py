import cv2
import os
import pickle
import numpy as np
from ultralytics import YOLO
import supervision as sv
import csv

from utils.bbox_utils import get_center_of_bbox, get_bbox_width


# كلاس التتبع
class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.track(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {"players": [],
                  "referees": [],
                  'ball': []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                #if cls_id == cls_names_inv["referees"]:
                 #   tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks


# تحميل الفريمات من الفيديو
def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


# حفظ إحداثيات لاعب معين في CSV
def save_player_coordinates_to_csv(tracks, player_id, output_path="player_5_coordinates.csv"):
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "x1", "y1", "x2", "y2"])  # Header

        for frame_num, frame_data in enumerate(tracks["players"]):
            if player_id in frame_data:
                bbox = frame_data[player_id]["bbox"]
                writer.writerow([frame_num] + bbox)


# -------------- التشغيل الرئيسي ------------------

if __name__ == "__main__":
    video_path = "match.mp4"  # ← حط اسم ملف الفيديو هنا
    model_path = "models/best.pt"    # ← حط اسم الموديل هنا
    player_id = 5             # ← رقم اللاعب اللي عايز تحفظ إحداثياته

    print("⏳ جاري تحميل الفيديو...")
    frames = load_video_frames(video_path)

    print("✅ تحميل الفريمات تم، عددها:", len(frames))

    print("🚀 تشغيل الموديل وتتبع اللاعبين...")
    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(frames)

    print(f"💾 جاري حفظ إحداثيات اللاعب {player_id} في ملف CSV...")
    save_player_coordinates_to_csv(tracks, player_id)

    print("🎉 تم الحفظ بنجاح في player_5_coordinates.csv")
