import cv2
import numpy as np
import time
import os
import argparse

# Parse arguments (webcam index or video file path)
parser = argparse.ArgumentParser(description="Simple thermal blob detector")
parser.add_argument('--video', default=0, help='camera index (0) or path to video file')
parser.add_argument('--outdir', default='detections', help='folder to save detections')
args = parser.parse_args()

# Create output folder for detections
os.makedirs(args.outdir, exist_ok=True)

cap = cv2.VideoCapture(int(args.video) if str(args.video).isdigit() else args.video)
if not cap.isOpened():
    print("❌ Cannot open camera or video:", args.video)
    exit(1)

frame_id = 0
print("▶ Detector started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale (most thermal cameras output grayscale images)
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Enhance contrast and reduce noise
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Threshold to highlight hot regions
    _, th = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # Find contours (hot objects)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:  # filter out very small noise blobs
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        found = True

    # Show video frames
    cv2.imshow("thermal", frame)
    cv2.imshow("threshold", th)

    # Save frame if detection found
    if found:
        ts = int(time.time())
        filename = os.path.join(args.outdir, f"detection_{ts}_{frame_id}.png")
        cv2.imwrite(filename, frame)
        print("✅ Detection saved:", filename)

    frame_id += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
