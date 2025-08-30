from ultralytics import YOLO

# 加载你训练好的模型
model = YOLO(r"E:\hszz\yolo\model\train_2\weights\best.pt")

# 预测一张图片
results = model.predict(
    source=r"E:\hszz\yolo\video\gzx_2.mp4",  # 输入图片或文件夹
    save=False,                     # 保存带框图片
    project=r"E:\hszz\yolo\results",  # 指定保存目录
    show=True
)
# 输出预测框信息
for r in results:
    print(r.boxes)
