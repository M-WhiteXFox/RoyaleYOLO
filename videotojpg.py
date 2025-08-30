import cv2
import os
import math

video_path = "gzx_2.mp4"  # 这里改成你的绝对路径
output_folder = "frames2_3s"
interval_seconds = 3

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"无法打开视频：{video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    raise ValueError("无法获取视频帧率，请检查视频文件")

frame_interval = math.floor(fps * interval_seconds)

count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % frame_interval == 0:
        frame_path = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        saved_count += 1

    count += 1

cap.release()
print(f"每隔 {interval_seconds} 秒截取一帧，已保存 {saved_count} 张图片到 '{output_folder}' 文件夹")
