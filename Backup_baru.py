import time
from datetime import datetime
import cv2
import numpy as np
import pygame
import serial
from ultralytics import YOLO

VIDEO_SRC = "hasil_resized2.mp4"
MODEL_PATH = r"C:\Users\acer\Downloads\TMMIN project\SPC detector\model\yolo11n-pose.engine"
AUDIO_PATH = r"C:\Users\acer\Downloads\TMMIN project\SPC detector\audio\SPC.mp3"

ser = serial.Serial("COM13", 9600, timeout=1)
time.sleep(2)

pygame.mixer.init()
sound = pygame.mixer.Sound(AUDIO_PATH)

def send_signal(flags):
    active = [k for k, v in flags.items() if v]
    msg = ("_".join(active) if active else "ALL_OFF") + "\n"
    ser.write(msg.encode())

def in_poly(pt, poly):
    if len(poly) < 3:
        return False
    return cv2.pointPolygonTest(np.array(poly, np.int32), (int(pt[0]), int(pt[1])), False) >= 0

def draw_regions(src):
    cap = cv2.VideoCapture(src)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return [], []

    pts_a, pts_s = [], []
    curr, col = pts_a, (255, 255, 0)

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            curr.append((x, y))

    cv2.namedWindow("Draw")
    cv2.setMouseCallback("Draw", mouse_cb)

    while True:
        img = frame.copy()
        for p in curr:
            cv2.circle(img, p, 5, col, -1)
        if len(curr) >= 3:
            cv2.polylines(img, [np.array(curr)], True, col, 2)
        cv2.imshow("Draw", img)
        if cv2.waitKey(1) == 13:
            if curr is pts_a:
                curr, col = pts_s, (0, 255, 0)
            else:
                break
    cv2.destroyWindow("Draw")
    return pts_a, pts_s

if __name__ == "__main__":
    reg_audio, reg_spc = draw_regions(VIDEO_SRC)
    model = YOLO(MODEL_PATH)
    history, logs = {}, {}

    for res in model.track(source=VIDEO_SRC, stream=True, show=False):
        frame = res.plot()
        if res.boxes.id is None:
            continue

        ids = res.boxes.id.cpu().numpy().astype(int)
        kpts = res.keypoints.xy.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        flags = {"RED": False, "YELLOW": False, "GREEN": False}

        for i, tid in enumerate(ids):
            kp, box = kpts[i], boxes[i]
            bot = ((box[0] + box[2]) // 2, box[3])
            nose = kp[0]

            history.setdefault(tid, []).append((int(nose[0]), int(nose[1])))
            if len(history[tid]) > 5:
                history[tid].pop(0)

            h = history[tid]
            arah = "Diam"
            if len(h) >= 2:
                dx, dy = h[-1][0] - h[0][0], h[-1][1] - h[0][1]
                arah = "Mendekat" if (dx > 0 if abs(dx) > abs(dy) else dy > 0) else "Menjauh"

            log = logs.setdefault(tid, {
                "spc_complete": False,
                "inside_spc": False,
                "has_exited": False,
                "violated": False,
                "wrist_outside_time": None,
                "last_wrist_x": None,
                "last_direction": None,
                "audio_played": False,
            })

            if in_poly(bot, reg_audio) and not log["audio_played"] and arah == "Mendekat":
                if not pygame.mixer.get_busy():
                    sound.play()
                log["audio_played"] = True

            if arah == "Mendekat":
                if in_poly(bot, reg_spc):
                    log["inside_spc"] = True
                    r_wx, l_sx, r_sx = kp[10][0], kp[5][0], kp[6][0]
                    min_x, max_x = min(l_sx, r_sx), max(l_sx, r_sx)
                    vel = abs(r_wx - log["last_wrist_x"]) if log["last_wrist_x"] is not None else 0
                    direct = "right" if r_wx > r_sx else "left"

                    if (r_wx < min_x or r_wx > max_x) and vel > 10:
                        if not log["wrist_outside_time"]:
                            log["wrist_outside_time"] = datetime.now()
                            log["last_direction"] = direct
                    elif (
                        log["wrist_outside_time"]
                        and (datetime.now() - log["wrist_outside_time"]).total_seconds() > 0.5
                        and min_x < r_wx < max_x
                        and direct != log["last_direction"]
                    ):
                        log["spc_complete"] = True
                        log["wrist_outside_time"] = None
                    log["last_wrist_x"] = r_wx

                elif log["inside_spc"] and not log["has_exited"]:
                    log["has_exited"] = True
                    if not log["spc_complete"]:
                        log["violated"] = True

            if log["has_exited"]:
                if log["violated"]:
                    flags["RED"] = True
                elif log["spc_complete"]:
                    flags["GREEN"] = True
            elif log["inside_spc"] and not log["spc_complete"]:
                flags["YELLOW"] = True

            lbl = "SPC Complete" if log["spc_complete"] else "Waiting..."
            col = (0, 0, 255) if log["violated"] else (0, 255, 0)
            cv2.putText(frame, f"ID:{tid} | {lbl}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        send_signal(flags)
        cv2.imshow("SPC Detection Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    ser.write(b"ALL_OFF\n")
    ser.close()
    cv2.destroyAllWindows()
    