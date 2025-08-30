import os
from ultralytics import YOLO

# ---------------- 配置区域 ----------------
DATA_YAML = r"D:/Github/labelme2yolo/YOLODataset/dataset.yaml"  # 数据集
PROJECT_DIR = r"E:\hszz\yolo\model"               # 输出主目录
LOCAL_MODEL_PATH = r"C:\Users\White\yolo11n.pt"  # 本地权重

EPOCHS = 100      # 训练轮数
IMGSZ = 640       # 输入图片大小
DEVICE = "0"      # GPU id, CPU 可改 "cpu"
# ----------------------------------------

def get_next_train_name(base_dir, prefix="train"):
    """生成下一个训练文件夹名，避免覆盖"""
    existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix)]
    numbers = [int(d.replace(prefix+"_","")) for d in existing if "_" in d and d.split("_")[-1].isdigit()]
    next_num = max(numbers)+1 if numbers else 1
    return f"{prefix}_{next_num}"

def main():
    # 检查本地模型
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"本地模型不存在: {LOCAL_MODEL_PATH}")

    # 创建输出目录
    os.makedirs(PROJECT_DIR, exist_ok=True)

    # 自动生成训练子文件夹
    train_name = get_next_train_name(PROJECT_DIR)
    print(f"训练结果将保存在: {os.path.join(PROJECT_DIR, train_name)}")

    # 加载模型
    print(f"加载本地模型: {LOCAL_MODEL_PATH}")
    model = YOLO(LOCAL_MODEL_PATH)

    # 开始训练
    print("开始训练...")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        project=PROJECT_DIR,
        name=train_name,
        device=DEVICE
    )

    print(f"训练完成，结果保存在: {os.path.join(PROJECT_DIR, train_name)}")

if __name__ == "__main__":
    main()
