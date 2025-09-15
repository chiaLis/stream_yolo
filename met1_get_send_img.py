import socket, struct, cv2, numpy as np
from ultralytics import YOLO

HOST, PORT = "0.0.0.0", 9000   # 對外開放
model = YOLO("yolo11n.engine") # 換成你的 engine 檔路徑

def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"[INFO] Listening on {HOST}:{PORT}")
    while True:
        conn, addr = s.accept()
        with conn:
            while True:
                hdr = recv_exact(conn, 4)
                if not hdr:
                    break
                (n,) = struct.unpack(">I", hdr)
                data = recv_exact(conn, n)
                if not data:
                    break

                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                res = model(img)
                annot = res[0].plot()
                ok, buf = cv2.imencode(".jpg", annot)
                if not ok:
                    conn.sendall(struct.pack(">I", 0))
                    continue

                conn.sendall(struct.pack(">I", len(buf)))
                conn.sendall(buf.tobytes())

