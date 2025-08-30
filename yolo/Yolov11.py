from ultralytics import YOLO

# 加载默认模型（nano）
model = YOLO("yolo11n.pt")

# 跑一张图片
results = model.predict(
    source=1,  # 输入图片或文件夹
    show=True
)