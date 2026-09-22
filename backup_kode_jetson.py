import cv2
import numpy as np
import pygame
from ultralytics import YOLO
from datetime import datetime
import Jetson.GPIO as GPIO
import time

# ----------------------------- Konfigurasi GPIO ----------------------------- #
RED_PIN = 11
YELLOW_PIN = 13
GREEN_PIN = 15

GPIO.setmode(GPIO.BOARD)
GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.setup(YELLOW_PIN, GPIO.OUT)
GPIO.setup(GREEN_PIN, GPIO.OUT)

def set_lampu(red=False, yellow=False, green=False):
    GPIO.output(RED_PIN, red)
    GPIO.output(YELLOW_PIN, yellow)
    GPIO.output(GREEN_PIN, green)

# ----------------------------- Inisialisasi ----------------------------- #
pygame.mixer.init()
def play_audio(file_path):
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

model = YOLO('yolo11n-pose.pt')
cap = cv2.VideoCapture(0)

region_audio_pts = [(50, 100), (300, 100), (300, 400), (50, 400)]
region_spc_pts = [(350, 100), (600, 100), (600, 400), (350, 400)]

trajectory_history = {}
tracking_log = {}

def update_trajectory(track_id, point):
    pt = (int(point[0]), int(point[1]))
    trajectory_history.setdefault(track_id, []).append(pt)
    if len(trajectory_history[track_id]) > 5:
        trajectory_history[track_id].pop(0)

def detect_direction(track_id):
    history = trajectory_history.get(track_id, [])
    if len(history) >= 2:
        dx = history[-1][0] - history[0][0]
        dy = history[-1][1] - history[0][1]
        return "Mendekat" if dy > 0 else "Menjauh"
    return "Diam"

def is_inside_region(xy, region):
    pt = (int(xy[0]), int(xy[1]))
    return cv2.pointPolygonTest(np.array(region, dtype=np.int32), pt, False) >= 0

# ----------------------------- Loop Utama ----------------------------- #
try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(source=frame, stream=False, verbose=False)[0]
        if results.boxes.id is None:
            continue

        ids = results.boxes.id.cpu().numpy().astype(int)
        keypoints_all = results.keypoints.xy.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()

        status_flags = {"RED": False, "YELLOW": False, "GREEN": False}

        for i, track_id in enumerate(ids):
            keypoints = keypoints_all[i]
            box = boxes[i].astype(int)
            center_bottom = ((box[0] + box[2]) // 2, box[3])
            nose = keypoints[0]
            update_trajectory(track_id, (nose[0], nose[1]))
            arah = detect_direction(track_id)

            if track_id not in tracking_log:
                tracking_log[track_id] = {
                    'detected_time': datetime.now(),
                    'spc_complete': False,
                    'inside_spc': False,
                    'has_exited': False,
                    'violated': False,
                    'wrist_outside_time': None,
                    'last_wrist_x': None,
                    'last_direction': None,
                    'audio_played': False,
                }

            log = tracking_log[track_id]

            if is_inside_region(center_bottom, region_audio_pts):
                if not log['audio_played'] and arah == "Mendekat":
                    play_audio("SPC.mp3")
                    log['audio_played'] = True

            if arah == "Mendekat":
                inside_spc = is_inside_region(center_bottom, region_spc_pts)
                if inside_spc:
                    log['inside_spc'] = True
                    r_wrist_x = keypoints[10][0]
                    l_shoulder_x = keypoints[5][0]
                    r_shoulder_x = keypoints[6][0]
                    min_x = min(l_shoulder_x, r_shoulder_x)
                    max_x = max(l_shoulder_x, r_shoulder_x)

                    velocity = abs(r_wrist_x - log['last_wrist_x']) if log['last_wrist_x'] else 0
                    direction = 'right' if r_wrist_x > r_shoulder_x else 'left'

                    if (r_wrist_x < min_x or r_wrist_x > max_x) and velocity > 10:
                        if not log['wrist_outside_time']:
                            log['wrist_outside_time'] = datetime.now()
                            log['last_direction'] = direction

                    elif (log['wrist_outside_time'] and
                          (datetime.now() - log['wrist_outside_time']).total_seconds() > 0.5 and
                          (min_x < r_wrist_x < max_x) and
                          direction != log['last_direction']):
                        log['spc_complete'] = True
                        log['wrist_outside_time'] = None

                    log['last_wrist_x'] = r_wrist_x

                elif log['inside_spc'] and not log['has_exited']:
                    log['has_exited'] = True
                    if not log['spc_complete']:
                        log['violated'] = True
                        print(f"[!] Pelanggaran oleh ID {track_id}")

            # Status untuk lampu
            if log['has_exited']:
                if log['violated']:
                    status_flags["RED"] = True
                elif log['spc_complete']:
                    status_flags["GREEN"] = True
            elif log['inside_spc']:
                if not log['spc_complete']:
                    status_flags["YELLOW"] = True

        # ----------------------------- Lampu Output ----------------------------- #
        set_lampu(
            red=status_flags["RED"],
            yellow=status_flags["YELLOW"],
            green=status_flags["GREEN"]
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    GPIO.cleanup()
    pygame.mixer.quit()
