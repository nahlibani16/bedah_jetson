import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os
import torch
import serial
import time
from playsound import playsound

print(torch.cuda.is_available())
print(torch.cuda.current_device())

# Inisialisasi koneksi serial ke Arduino
try:
    arduino = serial.Serial('COM13', 9600, timeout=1)
    time.sleep(2)
    print("Microcontroller Terhubung di port COM13")
except:
    arduino = None
    print("[WARNING] Arduino tidak terdeteksi. Output fisik tidak akan bekerja.")

model = YOLO('yolo11n-pose.pt')

region_audio_pts = []
region_spc_pts = []
drawing_state = 'audio'

def draw_region(event, x, y, flags, param):
    global drawing_state
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing_state == 'audio':
            region_audio_pts.append((x, y))
        elif drawing_state == 'spc':
            region_spc_pts.append((x, y))

cv2.namedWindow("Draw Region")
cv2.setMouseCallback("Draw Region", draw_region)

sumber = 1
cap = cv2.VideoCapture(sumber)
ret, first_frame = cap.read()
if ret:
    while True:
        temp_frame = first_frame.copy()
        if drawing_state == 'audio':
            for pt in region_audio_pts:
                cv2.circle(temp_frame, pt, 5, (255, 255, 0), -1)
            if len(region_audio_pts) >= 3:
                cv2.polylines(temp_frame, [np.array(region_audio_pts)], isClosed=True, color=(255, 255, 0), thickness=1)
        elif drawing_state == 'spc':
            for pt in region_spc_pts:
                cv2.circle(temp_frame, pt, 5, (0, 255, 0), -1)
            if len(region_spc_pts) >= 3:
                cv2.polylines(temp_frame, [np.array(region_spc_pts)], isClosed=True, color=(0, 255, 0), thickness=1)

        cv2.imshow("Draw Region", temp_frame)
        key = cv2.waitKey(1)
        if key == 13:
            if drawing_state == 'audio':
                drawing_state = 'spc'
            elif drawing_state == 'spc':
                break

cv2.destroyWindow("Draw Region")
cap.release()

def is_inside_region(xy, region):
    pt = (int(xy[0]), int(xy[1]))
    return cv2.pointPolygonTest(np.array(region, dtype=np.int32), pt, False) >= 0

tracking_log = {}
offset = 30  # offset garis ungu dari shoulder

results = model.track(source=sumber, stream=True, show=True)

for frame_num, result in enumerate(results):
    frame = result.orig_img.copy()
    if result.boxes.id is None:
        continue

    ids = result.boxes.id.cpu().numpy().astype(int)
    keypoints_all = result.keypoints.xy.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()

    status_flags = {"RED": False, "YELLOW": False, "GREEN": False}
    audio_triggered = False

    for i, track_id in enumerate(ids):
        keypoints = keypoints_all[i]
        box = boxes[i].astype(int)

        center_bottom = ((box[0] + box[2]) // 2, box[3])
        center_middle = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

        if track_id not in tracking_log:
            tracking_log[track_id] = {
                'detected_time': datetime.now(),
                'spc_complete': False,
                'inside_spc': False,
                'has_exited': False,
                'violated': False,
                'audio_played': False,
                'crossed_left': False,
                'crossed_right': False
            }

        log = tracking_log[track_id]

        if is_inside_region(center_middle, region_audio_pts):
            if not log['audio_played']:
                playsound(r"C:\\Users\\acer\\Downloads\\TMMIN project\\SPC detector\\SPC.mp3", block=False)
                log['audio_played'] = True
                print(f"[AUDIO] Suara diputar untuk ID {track_id}")

        inside_spc = is_inside_region(center_bottom, region_spc_pts)
        if inside_spc:
            log['inside_spc'] = True

            l_shoulder = keypoints[5]
            r_wrist = keypoints[10]
            l_wrist = keypoints[9]

            x_center = int(l_shoulder[0])
            y_shoulder = int(l_shoulder[1])
            x_left_line = x_center - offset
            x_right_line = x_center + offset
            
            # Gambar garis referensi ungu
            cv2.line(frame, (x_left_line, y_shoulder - 100), (x_left_line, y_shoulder + 100), (255, 0, 255), 2)
            cv2.line(frame, (x_right_line, y_shoulder - 100), (x_right_line, y_shoulder + 100), (255, 0, 255), 2)

            for wrist in [l_wrist, r_wrist]:
                x_wrist = wrist[0]
                if x_wrist < x_left_line:
                    log['crossed_left'] = True
                if x_wrist > x_right_line:
                    log['crossed_right'] = True

            if log['crossed_left'] and log['crossed_right']:
                log['spc_complete'] = True

        elif log['inside_spc'] and not log['has_exited']:
            log['has_exited'] = True
            if not log['spc_complete']:
                log['violated'] = True
                print(f"[!] Pelanggaran oleh ID {track_id}")

        if log['has_exited']:
            if log['violated']:
                status_flags["RED"] = True
            elif log['spc_complete']:
                status_flags["GREEN"] = True
        elif log['inside_spc']:
            if not log['spc_complete']:
                status_flags["YELLOW"] = True

        status_label = ""
        color = (0, 255, 0)
        if log['has_exited']:
            if log['violated']:
                status_label = "Pelanggaran!"
                color = (0, 0, 255)
            else:
                status_label = "SPC Lengkap"
        elif log['inside_spc']:
            if not log['spc_complete']:
                status_label = "Menunggu... (Belum Lengkap)"
                color = (0, 255, 255)
            else:
                status_label = "SPC Lengkap (Tunggu keluar)"
                color = (0, 255, 0)

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, f"ID: {track_id}", (box[0], box[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, status_label, (box[0], box[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if arduino is not None:
        if not any(status_flags.values()):
            arduino.write(b'ALL_OFF\n')
        else:
            flags = []
            if status_flags['RED']:
                flags.append('RED')
            if status_flags['YELLOW']:
                flags.append('YELLOW')
            if status_flags['GREEN']:
                flags.append('GREEN')
            signal = '_'.join(flags)
            arduino.write((signal + '\n').encode())

    cv2.imshow("SPC Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

if arduino is not None:
    arduino.write(b'ALL_OFF\n')
    time.sleep(0.5)

cv2.destroyAllWindows()
