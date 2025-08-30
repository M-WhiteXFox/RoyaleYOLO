# -*- coding: utf-8 -*-
import cv2
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import logging
import sys
import gc
import threading
import queue
from datetime import datetime
import time


class TextHandler(logging.Handler):
    """自定义日志处理器，将日志消息重定向到Tkinter Text控件。"""

    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.text_widget.configure(state='disabled')
        # 创建一个队列来安全地传递日志消息
        self.text_queue = queue.Queue()
        self.start_monitoring()

    def emit(self, record):
        msg = self.format(record)
        # 将消息放入队列，而不是直接修改UI
        self.text_queue.put(msg)

    def start_monitoring(self):
        self.root = self.text_widget.winfo_toplevel()
        # 使用 after() 方法定期检查队列并更新UI
        self.root.after(100, self.check_queue)

    def check_queue(self):
        # 从队列中取出所有消息并更新UI
        while not self.text_queue.empty():
            msg = self.text_queue.get()
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
            self.text_widget.configure(state='disabled')
        self.root.after(100, self.check_queue)


class YoloAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO 视频标注工具")
        self.root.geometry("1600x1000")

        self.setup_logging()

        # Configuration
        self.config_file = "config.json"
        self.video_path = tk.StringVar()
        self.model_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.conf_threshold = tk.DoubleVar(value=0.25)
        self.iou_threshold = tk.DoubleVar(value=0.7)
        self.load_config()

        # Model and video
        self.model = None
        self.video_capture = None
        self.current_frame = None
        self.photo_image = None
        self.current_frame_number = 0
        self.total_frames = 0
        self.fps = 0
        self.is_playing = False

        # 性能优化：创建线程和队列
        self.inference_thread = None
        self.frame_queue = queue.Queue(maxsize=1)  # 用于主线程向推理线程发送帧
        self.result_queue = queue.Queue(maxsize=1)  # 用于推理线程向主线程发送结果
        self.stop_event = threading.Event()  # 用于优雅地停止线程

        # Annotations
        self.detections = []
        self.selected_detection_id = None
        self.next_detection_id = 1
        self.label_mapping = {}

        self.setup_ui()
        self.update_status()

    def setup_logging(self):
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # File handler
        log_file = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log"))
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.video_path.set(config.get("video_path", ""))
                    self.model_path.set(config.get("model_path", ""))
                    self.output_dir.set(config.get("output_dir", ""))
                    self.conf_threshold.set(config.get("conf_threshold", 0.25))
                    self.iou_threshold.set(config.get("iou_threshold", 0.7))
                    self.logger.info("已成功加载配置文件。")
            except (IOError, json.JSONDecodeError) as e:
                self.logger.error(f"加载配置文件时出错: {e}")
        else:
            self.logger.warning("未找到配置文件，使用默认配置。")

    def save_config(self):
        config = {
            "video_path": self.video_path.get(),
            "model_path": self.model_path.get(),
            "output_dir": self.output_dir.get(),
            "conf_threshold": self.conf_threshold.get(),
            "iou_threshold": self.iou_threshold.get()
        }
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            self.logger.info("已保存配置到文件。")
        except IOError as e:
            self.logger.error(f"保存配置文件时出错: {e}")

    def setup_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0, minsize=400)
        self.root.rowconfigure(1, weight=1)

        # Control Panel
        control_frame = ttk.LabelFrame(self.root, text="控制面板")
        control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)

        ttk.Label(control_frame, text="视频文件:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(control_frame, textvariable=self.video_path).grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(control_frame, text="浏览...", command=self.select_video).grid(row=0, column=2, padx=5, pady=2)

        ttk.Label(control_frame, text="模型文件:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(control_frame, textvariable=self.model_path).grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(control_frame, text="浏览...", command=self.select_model).grid(row=1, column=2, padx=5, pady=2)

        ttk.Label(control_frame, text="输出目录:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(control_frame, textvariable=self.output_dir).grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(control_frame, text="浏览...", command=self.select_output_dir).grid(row=2, column=2, padx=5, pady=2)

        ttk.Label(control_frame, text="置信度阈值:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        ttk.Scale(control_frame, variable=self.conf_threshold, from_=0.0, to=1.0, orient="horizontal",
                  command=self.update_status).grid(row=3, column=1, sticky="ew", padx=5, pady=2)
        ttk.Label(control_frame, textvariable=self.conf_threshold, text="0.00", relief="sunken", width=5).grid(row=3,
                                                                                                               column=2,
                                                                                                               padx=5,
                                                                                                               pady=2)

        ttk.Label(control_frame, text="IOU 阈值:").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        ttk.Scale(control_frame, variable=self.iou_threshold, from_=0.0, to=1.0, orient="horizontal",
                  command=self.update_status).grid(row=4, column=1, sticky="ew", padx=5, pady=2)
        ttk.Label(control_frame, textvariable=self.iou_threshold, text="0.00", relief="sunken", width=5).grid(row=4,
                                                                                                              column=2,
                                                                                                              padx=5,
                                                                                                              pady=2)

        ttk.Button(control_frame, text="加载视频并模型", command=self.load_video_and_model).grid(row=5, column=0,
                                                                                                 columnspan=3,
                                                                                                 sticky="ew", padx=5,
                                                                                                 pady=5)
        ttk.Button(control_frame, text="保存配置", command=self.save_config).grid(row=6, column=0, columnspan=3,
                                                                                  sticky="ew", padx=5, pady=5)

        # Video Player and Canvas
        video_frame = ttk.Frame(self.root)
        video_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        self.video_canvas = tk.Canvas(video_frame, bg="black")
        self.video_canvas.grid(row=0, column=0, sticky="nsew")
        self.video_canvas.bind("<Button-1>", self.on_canvas_click)

        # Playback controls
        playback_frame = ttk.Frame(video_frame)
        playback_frame.grid(row=1, column=0, sticky="ew")
        playback_frame.columnconfigure(1, weight=1)

        self.play_pause_btn = ttk.Button(playback_frame, text="播放", command=self.play_pause_video)
        self.play_pause_btn.grid(row=0, column=0, padx=5)

        self.slider = ttk.Scale(playback_frame, from_=0, to=100, orient="horizontal", command=self.on_slider_move)
        self.slider.grid(row=0, column=1, sticky="ew", padx=5)

        self.current_time_label = ttk.Label(playback_frame, text="00:00:00 / 00:00:00")
        self.current_time_label.grid(row=0, column=2, padx=5)

        # Right sidebar
        sidebar_frame = ttk.Frame(self.root)
        sidebar_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=10, pady=10)
        sidebar_frame.columnconfigure(0, weight=1)
        sidebar_frame.rowconfigure(1, weight=1)

        # Status and Logger
        status_frame = ttk.LabelFrame(sidebar_frame, text="状态与日志")
        status_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.status_label = ttk.Label(status_frame, text="准备就绪。")
        self.status_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.log_text = tk.Text(status_frame, wrap="word", height=10)
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        # 优化日志处理，使用线程安全的方式
        self.log_handler = TextHandler(self.log_text)
        self.logger.addHandler(self.log_handler)

        # Detection List
        detection_frame = ttk.LabelFrame(sidebar_frame, text="标注列表")
        detection_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        detection_frame.rowconfigure(0, weight=1)
        detection_frame.columnconfigure(0, weight=1)

        self.detection_list = tk.Listbox(detection_frame, height=15)
        self.detection_list.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.detection_list.bind("<<ListboxSelect>>", self.on_list_select)

        # Buttons
        button_frame = ttk.Frame(detection_frame)
        button_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)

        ttk.Button(button_frame, text="导出标注", command=self.export_detections).grid(row=0, column=0, sticky="ew",
                                                                                       padx=2, pady=2)
        self.delete_btn = ttk.Button(button_frame, text="删除选中", command=self.delete_selected_detection,
                                     state="disabled")
        self.delete_btn.grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="清除所有", command=self.clear_all_detections).grid(row=0, column=2, sticky="ew",
                                                                                          padx=2, pady=2)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.bind("<Configure>", self.on_resize)
        self.update_status()

    def on_resize(self, event):
        # 仅在调整窗口大小时重绘画布
        self.redraw_canvas()

    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("视频文件", "*.mp4;*.avi")])
        if path:
            self.video_path.set(path)

    def select_model(self):
        path = filedialog.askopenfilename(filetypes=[("模型文件", "*.pt")])
        if path:
            self.model_path.set(path)

    def select_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def load_video_and_model(self):
        self.save_config()
        video_path = self.video_path.get()
        model_path = self.model_path.get()
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("错误", "请选择一个有效的视频文件。")
            return
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("错误", "请选择一个有效的模型文件。")
            return

        self.logger.info("正在加载视频和模型...")
        try:
            self.model = YOLO(model_path)
            self.label_mapping = self.model.names
            self.video_capture = cv2.VideoCapture(video_path)
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)

            self.slider.config(to=self.total_frames - 1)
            self.current_frame_number = 0
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
            self.play_next_frame()
            self.update_status()
            self.logger.info("视频和模型加载完成。")
        except Exception as e:
            self.logger.error(f"加载视频或模型时出错: {e}")
            messagebox.showerror("错误", f"加载视频或模型时出错: {e}")

    def play_pause_video(self):
        if not self.video_capture:
            messagebox.showinfo("信息", "请先加载视频。")
            return

        self.is_playing = not self.is_playing
        self.play_pause_btn.config(text="暂停" if self.is_playing else "播放")

        if self.is_playing:
            # 仅在未启动推理线程时启动
            if self.inference_thread is None or not self.inference_thread.is_alive():
                self.stop_event.clear()
                self.inference_thread = threading.Thread(target=self.run_inference)
                self.inference_thread.daemon = True
                self.inference_thread.start()
            self.play_video_loop()
        else:
            self.stop_event.set()  # 发出停止线程信号

    def play_video_loop(self):
        if self.is_playing:
            # 播放循环，每隔一定时间读取并显示下一帧
            delay = int(1000 / self.fps)
            self.play_next_frame()
            # 从结果队列中获取最新结果并处理，不阻塞UI
            try:
                results = self.result_queue.get_nowait()
                self.process_results(results, self.current_frame_number)
            except queue.Empty:
                pass
            self.root.after(delay, self.play_video_loop)

    def play_next_frame(self):
        if self.video_capture and self.current_frame_number < self.total_frames:
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = frame
                self.process_frame_for_display(frame)
                self.current_frame_number = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                self.slider.set(self.current_frame_number)
                self.update_status()

                # 将帧放入队列供推理线程使用
                try:
                    self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    # 如果队列已满，则跳过此帧，防止阻塞主线程
                    pass
            else:
                # 播放结束
                self.is_playing = False
                self.play_pause_btn.config(text="播放")
                self.stop_event.set()

    def run_inference(self):
        """在单独的线程中执行 YOLO 推理。"""
        self.logger.info("推理线程已启动。")
        while not self.stop_event.is_set():
            try:
                # 从队列中获取帧，设置超时以避免永久阻塞
                frame = self.frame_queue.get(timeout=1)
                # 执行 YOLO 模型推理
                results = self.model.track(
                    frame,
                    persist=True,
                    conf=self.conf_threshold.get(),
                    iou=self.iou_threshold.get(),
                    verbose=False
                )
                # 将结果放入队列供主线程使用
                try:
                    self.result_queue.put_nowait(results)
                except queue.Full:
                    # 如果结果队列已满，则丢弃结果，等待主线程处理
                    pass
            except queue.Empty:
                # 队列为空时短暂休眠，避免过度占用CPU
                time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"推理线程出错: {e}")
                self.is_playing = False
                self.stop_event.set()  # 遇到错误停止线程
                break
        self.logger.info("推理线程已停止。")

    def on_slider_move(self, value):
        if self.video_capture:
            frame_num = int(float(value))
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            self.is_playing = False
            self.play_pause_btn.config(text="播放")
            self.play_next_frame()
            self.update_status()

    def process_frame_for_display(self, frame):
        if frame is None:
            return

        # 绘制检测框（在主线程中进行，以保证UI安全）
        frame_copy = frame.copy()
        frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

        for det in self.detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']

            color = (255, 0, 0) if det['id'] == self.selected_detection_id else (0, 255, 0)

            cv2.rectangle(frame_rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame_rgb, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        image = Image.fromarray(frame_rgb)

        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            aspect_ratio = image.width / image.height
            if canvas_width / canvas_height > aspect_ratio:
                new_width = int(canvas_height * aspect_ratio)
                new_height = canvas_height
            else:
                new_width = canvas_width
                new_height = int(canvas_width / aspect_ratio)

            resized_image = image.resize((new_width, new_height), Image.LANCZOS)

            self.photo_image = ImageTk.PhotoImage(image=resized_image)

            self.video_canvas.delete("all")
            self.video_canvas.create_image(
                canvas_width / 2,
                canvas_height / 2,
                image=self.photo_image,
                anchor=tk.CENTER
            )

            gc.collect()

    def process_results(self, results, frame_number):
        """处理 YOLO 推理结果并更新标注列表。"""
        if results is None:
            return

        for result in results:
            if result.boxes and result.boxes.id is not None:
                for box, track_id in zip(result.boxes.xyxy, result.boxes.id):
                    x1, y1, x2, y2 = box
                    conf = result.boxes.conf.tolist()[0]
                    cls_id = result.boxes.cls.tolist()[0]
                    label = self.label_mapping.get(cls_id, f"Class {cls_id}")

                    existing_detection = next(
                        (d for d in self.detections if d['id'] == track_id and d['frame'] == frame_number), None)
                    if existing_detection:
                        existing_detection['box'] = [x1, y1, x2, y2]
                        existing_detection['label'] = label
                        existing_detection['conf'] = conf
                    else:
                        self.detections.append({
                            "id": int(track_id),
                            "frame": frame_number,
                            "box": [float(x1), float(y1), float(x2), float(y2)],
                            "label": label,
                            "conf": float(conf)
                        })
                        self.next_detection_id = max(self.next_detection_id, int(track_id) + 1)

        self.redraw_canvas()
        self.update_detection_list()

    def on_canvas_click(self, event):
        x, y = event.x, event.y
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()

        if self.current_frame is None:
            self.logger.warning("画布上没有可供标注的图像。")
            return

        frame_width = self.current_frame.shape[1]
        frame_height = self.current_frame.shape[0]

        canvas_aspect = canvas_width / canvas_height
        frame_aspect = frame_width / frame_height

        if canvas_aspect > frame_aspect:
            scale_factor = canvas_height / frame_height
            offset_x = (canvas_width - frame_width * scale_factor) / 2
            offset_y = 0
        else:
            scale_factor = canvas_width / frame_width
            offset_x = 0
            offset_y = (canvas_height - frame_height * scale_factor) / 2

        img_x = (x - offset_x) / scale_factor
        img_y = (y - offset_y) / scale_factor

        found = False
        for det in self.detections:
            x1, y1, x2, y2 = det['box']
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                self.selected_detection_id = det['id']
                self.logger.info(f"在画布上选中了标注框 (ID: {det['id']})。")
                found = True
                break

        if not found:
            self.selected_detection_id = None
            self.logger.info("已取消选中任何标注框。")

        self.redraw_canvas()
        self.update_detection_list()

    def redraw_canvas(self):
        # 仅当有帧可用时才重绘
        if self.current_frame is not None:
            self.process_frame_for_display(self.current_frame)

    def export_detections(self):
        if not self.detections:
            messagebox.showinfo("信息", "没有可导出的标注。")
            return

        if not self.output_dir.get():
            messagebox.showerror("错误", "请先指定输出目录。")
            return

        filename = os.path.join(self.output_dir.get(), "detections.json")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.detections, f, ensure_ascii=False, indent=4)
            self.logger.info(f"标注已成功导出到: {filename}")
            messagebox.showinfo("成功", f"标注已成功导出到: {filename}")
        except IOError as e:
            self.logger.error(f"导出标注时出错: {e}")
            messagebox.showerror("错误", f"导出标注时出错: {e}")

    def delete_selected_detection(self):
        if self.selected_detection_id is None:
            messagebox.showinfo("信息", "请先在列表中选中一个标注。")
            return

        self.logger.warning(f"用户请求删除标注 (ID: {self.selected_detection_id})。")
        self.detections = [det for det in self.detections if det['id'] != self.selected_detection_id]
        self.selected_detection_id = None
        self.redraw_canvas()
        self.update_detection_list()

    def clear_all_detections(self):
        self.logger.warning("用户请求清除所有标注。")
        if messagebox.askokcancel("确认", "清除所有标注?"):
            self.detections.clear()
            self.selected_detection_id = None
            self.redraw_canvas()
            self.update_detection_list()
            self.logger.info("所有标注已清除。")
        else:
            self.logger.info("已取消清除操作。")

    def on_list_select(self, _):
        idx = self.detection_list.curselection()
        if idx:
            did = self.detections[idx[0]]["id"]
            self.selected_detection_id = did
            self.redraw_canvas()
            self.logger.info(f"在列表中选中标注框 (ID: {did})。")

    def update_detection_list(self):
        self.detection_list.delete(0, tk.END)
        for det in self.detections:
            self.detection_list.insert(
                tk.END,
                f"ID {det['id']}: {det['label']} ({det['conf']:.2f})"
            )
        self.delete_btn.config(
            state="normal" if self.selected_detection_id is not None else "disabled"
        )

    def update_status(self):
        if self.video_capture:
            total_time = self.total_frames / self.fps
            current_time = self.current_frame_number / self.fps
            total_time_str = time.strftime('%H:%M:%S', time.gmtime(total_time))
            current_time_str = time.strftime('%H:%M:%S', time.gmtime(current_time))
            self.current_time_label.config(text=f"{current_time_str} / {total_time_str}")

            status_text = f"总帧数: {self.total_frames}, FPS: {self.fps:.2f}, 当前帧: {self.current_frame_number}"
            self.status_label.config(text=status_text)
        else:
            self.status_label.config(text="准备就绪。")
            self.current_time_label.config(text="00:00:00 / 00:00:00")

    def on_closing(self):
        self.logger.info("正在关闭应用...")
        self.is_playing = False
        self.stop_event.set()
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2)
        if self.video_capture:
            self.video_capture.release()
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = YoloAnnotator(root)
    root.mainloop()
