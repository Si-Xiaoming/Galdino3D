import cv2
import os
import numpy as np
def extract_frames(video_path, output_dir, resize=(384, 384)):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        # Resize frame
        if resize and frame is not None:
            frame = cv2.resize(frame, resize)
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:05d}.png"), frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames.")
