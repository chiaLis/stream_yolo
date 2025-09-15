import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.engine")

# 影片來源：攝影機 0 或檔案 'input.mp4'
src = "videoSample_rec_20250909_175530.mp4"   # 或改成 0
cap = cv2.VideoCapture(src)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open source: {src}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# 若 ffmpeg/avc1 可用，用 H.264；不行就改 'mp4v' 或 MJPG
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 或 'mp4v'
out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

frame_id = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(frame)           # 每幀推論
    annotated = results[0].plot()    # 畫框

    out.write(annotated)             # 寫進影片
    frame_id += 1
    if frame_id % 50 == 0:
        print(f"[info] processed {frame_id} frames")

cap.release()
out.release()
print("done -> output.mp4")


