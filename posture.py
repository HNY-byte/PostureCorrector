import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os
import time
import threading
import winsound

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'pose_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading model... please wait")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        model_path
    )
    print("Model downloaded!")

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

def beep_alert():
    winsound.Beep(1000, 500)

def draw_guide(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (w-280, 10), (w-10, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, "Correct Posture Guide",
               (w-275, 35), cv2.FONT_HERSHEY_SIMPLEX,
               0.55, (0, 255, 255), 1)
    tips = [
        "1. Sit up straight",
        "2. Head up, chin level",
        "3. Shoulders relaxed",
        "4. Back against chair",
        "5. Feet flat on floor",
    ]
    for i, tip in enumerate(tips):
        cv2.putText(frame, tip,
                   (w-275, 65 + i*28),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1)

def draw_score(frame, angle):
    h, w = frame.shape[:2]
    if angle >= 160:
        score = min(100, int(angle - 100))
        color = (0, 255, 0)
        status = "GOOD"
    else:
        score = max(0, int(angle - 100))
        color = (0, 0, 255)
        status = "BAD"
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, h-80), (220, h-10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, f"Posture Score: {score}%",
               (15, h-50), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, color, 2)
    cv2.putText(frame, f"Status: {status}",
               (15, h-25), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, color, 2)

def draw_timer(frame, start_time, bad_time):
    h, w = frame.shape[:2]
    elapsed = int(time.time() - start_time)
    mins = elapsed // 60
    secs = elapsed % 60
    bad_mins = int(bad_time) // 60
    bad_secs = int(bad_time) % 60

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"Session : {mins:02d}:{secs:02d}",
               (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Bad Time: {bad_mins:02d}:{bad_secs:02d}",
               (15, 65), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, (0, 100, 255), 2)

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

cap = cv2.VideoCapture(0)
print("Camera started! Press Q to quit.")

start_time = time.time()
bad_time = 0
last_beep = 0
bad_posture_start = None

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect(mp_image)

        draw_guide(frame)

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            lm = results.pose_landmarks[0]

            shoulder = [lm[11].x, lm[11].y]
            ear      = [lm[7].x,  lm[7].y]
            hip      = [lm[23].x, lm[23].y]

            angle = calculate_angle(ear, shoulder, hip)

            draw_score(frame, angle)

            if angle < 160:
                # Bad posture
                if bad_posture_start is None:
                    bad_posture_start = time.time()
                else:
                    bad_time += time.time() - bad_posture_start
                    bad_posture_start = time.time()

                # Beep every 3 seconds
                if time.time() - last_beep > 3:
                    threading.Thread(target=beep_alert).start()
                    last_beep = time.time()

                cv2.putText(frame, "BAD POSTURE! Sit Straight!",
                           (30, 130), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 0, 255), 3)
                cv2.rectangle(frame, (0, 0),
                             (frame.shape[1], frame.shape[0]),
                             (0, 0, 255), 8)
            else:
                bad_posture_start = None
                cv2.putText(frame, "Good Posture :)",
                           (30, 130), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 3)
                cv2.rectangle(frame, (0, 0),
                             (frame.shape[1], frame.shape[0]),
                             (0, 255, 0), 8)

            cv2.putText(frame, f"Angle: {int(angle)}",
                       (30, 170), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (255, 255, 0), 2)

        draw_timer(frame, start_time, bad_time)

        cv2.imshow("Posture Corrector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

total = int(time.time() - start_time)
print(f"\n--- Session Summary ---")
print(f"Total Time   : {total//60:02d}:{total%60:02d}")
print(f"Bad Posture  : {int(bad_time)//60:02d}:{int(bad_time)%60:02d}")
print(f"Good Posture : {(total-int(bad_time))//60:02d}:{(total-int(bad_time))%60:02d}")