import cv2
import os


video_path = "shark.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video")
    exit()

ret, frame = cap.read()
if not ret:
    print("Cannot read first frame")
    exit()

bbox = cv2.selectROI("Select Object", frame, showCrosshair=True)
cv2.destroyWindow("Select Object")

trackers = {
    "CSRT": cv2.legacy.TrackerCSRT_create(),
    "KCF": cv2.legacy.TrackerKCF_create()
}

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

for tracker_name, tracker in trackers.items():
    cap.release()
    cap = cv2.VideoCapture(video_path)  
    ret, frame = cap.read()             
    tracker.init(frame, bbox)

    out_path = f"{output_dir}/{tracker_name}_{video_path}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, frame_size)

    print(f"Running {tracker_name} tracker...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, box = tracker.update(frame)

        if success:
            x, y, w, h = map(int, box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, tracker_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking Lost", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            
        out.write(frame)

    out.release()
    print(f"Saved video: {out_path}")

cap.release()
print("All done.")
