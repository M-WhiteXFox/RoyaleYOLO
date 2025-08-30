import cv2
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO


class YoloAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO 视频标注工具")
        self.root.geometry("1600x1000")

        # Configuration
        self.config_file = "config.json"
        self.video_path = tk.StringVar()
        self.model_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.conf_threshold = tk.DoubleVar(value=0.5)
        self.img_size = tk.IntVar(value=640)
        self.frame_interval = tk.IntVar(value=5)
        self.load_config()

        # State
        self.cap = None
        self.model = None
        self.class_names = []
        self.history = []
        self.current_frame_num = 0
        self.current_frame_img = None
        self.detections = []
        self.selected_detection_id = None

        # Scaling/Zooming
        self.zoom_level = 1.0  # Current zoom level for detection canvas
        self.canvas_x_offset = 0  # X offset for canvas pan
        self.canvas_y_offset = 0  # Y offset for canvas pan

        # Interaction
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0

        self.drawing = False
        self.add_box_mode = False
        self.new_box_start_img = None
        self.temp_box_id = None
        self.start_pos_img = None
        self.start_coords = []

        # GUI
        self.setup_gui()
        self.bind_events()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                self.video_path.set(cfg.get("video_path", ""))
                self.model_path.set(cfg.get("model_path", ""))
                self.output_dir.set(cfg.get("output_dir", ""))
                self.conf_threshold.set(cfg.get("conf_threshold", 0.5))
                self.img_size.set(cfg.get("img_size", 640))
                self.frame_interval.set(cfg.get("frame_interval", 5))
            except Exception as e:
                print(f"Failed to load config file, creating default config: {e}")
                self.create_default_config()
        else:
            self.create_default_config()

    def create_default_config(self):
        default = {
            "video_path": "",
            "model_path": "",
            "output_dir": "",
            "conf_threshold": 0.5,
            "img_size": 640,
            "frame_interval": 5
        }
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(default, f, indent=4)
        self.video_path.set(default["video_path"])
        self.model_path.set(default["model_path"])
        self.output_dir.set(default["output_dir"])
        self.conf_threshold.set(default["conf_threshold"])
        self.img_size.set(default["img_size"])
        self.frame_interval.set(default["frame_interval"])

    def save_config(self):
        try:
            self.frame_interval.set(int(self.interval_entry.get()))
            self.conf_threshold.set(float(self.conf_entry.get()))
            self.img_size.set(int(self.img_size_entry.get()))
        except Exception:
            pass
        cfg = {
            "video_path": self.video_path.get(),
            "model_path": self.model_path.get(),
            "output_dir": self.output_dir.get(),
            "conf_threshold": self.conf_threshold.get(),
            "img_size": self.img_size.get(),
            "frame_interval": self.frame_interval.get()
        }
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=4)

    def on_closing(self):
        self.save_config()
        self.root.destroy()

    def setup_gui(self):
        # Main container
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # --------- Configuration Area ---------
        cfg = ttk.LabelFrame(main, text="配置", padding=10)
        cfg.pack(fill=tk.X, pady=5)
        cfg.grid_columnconfigure(1, weight=1)

        ttk.Label(cfg, text="视频文件:").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Entry(cfg, textvariable=self.video_path, state="readonly").grid(row=0, column=1, sticky="we")
        ttk.Button(cfg, text="选择...", command=self.select_video).grid(row=0, column=2, padx=5)

        ttk.Label(cfg, text="模型文件:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(cfg, textvariable=self.model_path, state="readonly").grid(row=1, column=1, sticky="we")
        ttk.Button(cfg, text="选择...", command=self.select_model).grid(row=1, column=2, padx=5)

        ttk.Label(cfg, text="输出目录:").grid(row=2, column=0, sticky="w", padx=5)
        ttk.Entry(cfg, textvariable=self.output_dir, state="readonly").grid(row=2, column=1, sticky="we")
        ttk.Button(cfg, text="选择...", command=self.select_output_dir).grid(row=2, column=2, padx=5)

        # --------- Operation Buttons Area ---------
        button_frame = ttk.LabelFrame(cfg, text="操作", padding=5)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10, sticky="we")
        for col in range(4):
            button_frame.grid_columnconfigure(col, weight=1)

        self.load_button = ttk.Button(button_frame, text="加载模型和视频", command=self.load_assets)
        self.load_button.grid(row=0, column=0, padx=5, pady=3, sticky="we")

        self.repredict_btn = ttk.Button(button_frame, text="重新预测", command=self.re_predict_frame, state="disabled")
        self.repredict_btn.grid(row=0, column=1, padx=5, pady=3, sticky="we")

        self.add_box_btn = ttk.Button(button_frame, text="添加新框", command=self.toggle_add_box_mode, state="disabled")
        self.add_box_btn.grid(row=0, column=2, padx=5, pady=3, sticky="we")

        self.delete_btn = ttk.Button(button_frame, text="删除选中", command=self.delete_selected, state="disabled")
        self.delete_btn.grid(row=0, column=3, padx=5, pady=3, sticky="we")

        self.clear_btn = ttk.Button(button_frame, text="全部清除", command=self.clear_all_detections, state="disabled")
        self.clear_btn.grid(row=1, column=0, padx=5, pady=3, sticky="we")

        self.prev_btn = ttk.Button(button_frame, text="上一张", command=lambda: self.navigate_frames(-1),
                                   state="disabled")
        self.prev_btn.grid(row=1, column=1, padx=5, pady=3, sticky="we")

        self.save_next_btn = ttk.Button(button_frame, text="保存并下一张",
                                        command=lambda: self.navigate_frames(1, save=True), state="disabled")
        self.save_next_btn.grid(row=1, column=2, padx=5, pady=3, sticky="we")

        self.skip_next_btn = ttk.Button(button_frame, text="跳过并下一张",
                                        command=lambda: self.navigate_frames(1, save=False), state="disabled")
        self.skip_next_btn.grid(row=1, column=3, padx=5, pady=3, sticky="we")

        self.check_paths()

        # --------- Parameter Area ---------
        param_frame = ttk.Frame(cfg)
        param_frame.grid(row=4, column=0, columnspan=3, sticky="we", pady=5)
        param_frame.grid_columnconfigure(3, weight=1)

        ttk.Label(param_frame, text="抽帧间隔:").grid(row=0, column=0, padx=5)
        self.interval_entry = ttk.Entry(param_frame, width=5, textvariable=self.frame_interval)
        self.interval_entry.grid(row=0, column=1)

        ttk.Label(param_frame, text="置信度:").grid(row=0, column=2, padx=5)
        self.conf_scale = ttk.Scale(param_frame, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.conf_threshold)
        self.conf_scale.grid(row=0, column=3, sticky="we")
        self.conf_entry = ttk.Entry(param_frame, width=5, textvariable=self.conf_threshold)
        self.conf_entry.grid(row=0, column=4, padx=5)

        ttk.Label(param_frame, text="图像尺寸:").grid(row=0, column=5, padx=5)
        self.img_size_entry = ttk.Entry(param_frame, width=5, textvariable=self.img_size)
        self.img_size_entry.grid(row=0, column=6)

        # --------- Workspace ---------
        workspace = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        workspace.pack(fill=tk.BOTH, expand=True)

        # Original Image (scaled down)
        left = ttk.Frame(workspace)
        ttk.Label(left, text="原图 (缩小)", font=("Arial", 12)).pack(pady=5)
        self.orig_canvas = tk.Canvas(left, bg="lightgray")
        self.orig_canvas.pack(fill=tk.BOTH, expand=True)
        workspace.add(left, weight=1)

        # Detection Results (zoomed/panned)
        right = ttk.PanedWindow(workspace, orient=tk.VERTICAL)
        cf = ttk.Frame(right)
        ttk.Label(cf, text="检测结果 (放大/拖动)", font=("Arial", 12)).pack(pady=5)
        self.detect_canvas = tk.Canvas(cf, bg="lightgray", cursor="crosshair")
        self.detect_canvas.pack(fill=tk.BOTH, expand=True)
        right.add(cf, weight=3)

        # Annotation List
        lf = ttk.LabelFrame(right, text="标注列表与操作", padding=5)
        self.detection_list = tk.Listbox(lf)
        self.detection_list.pack(fill=tk.BOTH, expand=True, padx=5)
        right.add(lf, weight=1)

        workspace.add(right, weight=3)

        # --------- Status Bar ---------
        bottom = ttk.Frame(main)
        bottom.pack(fill=tk.X, pady=5)
        self.status_label = ttk.Label(bottom, text="请先配置并加载...", anchor="w")
        self.status_label.pack(fill=tk.X, padx=10)

    def bind_events(self):
        # Left-click for panning or drawing/moving
        self.detect_canvas.bind("<ButtonPress-1>", self.on_press)
        self.detect_canvas.bind("<B1-Motion>", self.on_drag)
        self.detect_canvas.bind("<ButtonRelease-1>", self.on_release)
        # Right-click for changing class
        self.detect_canvas.bind("<Button-3>", self.on_right_press)
        # Listbox events
        self.detection_list.bind("<<ListboxSelect>>", self.on_list_select)
        # Keyboard events
        self.root.bind("<Delete>", self.delete_selected)
        self.root.bind("<Escape>", self.cancel_add_mode)

        # Mouse wheel for zooming
        self.detect_canvas.bind("<MouseWheel>", self.on_zoom)  # For Windows/Linux
        self.detect_canvas.bind("<Button-4>", self.on_zoom)  # For Linux
        self.detect_canvas.bind("<Button-5>", self.on_zoom)  # For Linux

    def on_zoom(self, event):
        if self.current_frame_img is None:
            return

        zoom_factor = 1.1 if event.delta > 0 or event.num == 4 else 1 / 1.1
        new_zoom_level = max(1.0, self.zoom_level * zoom_factor)

        # Determine the canvas center
        canvas_w, canvas_h = self.detect_canvas.winfo_width(), self.detect_canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 800, 600

        # Calculate the image coordinates of the center point
        center_x_img = (canvas_w / 2 - self.canvas_x_offset) / self.zoom_level
        center_y_img = (canvas_h / 2 - self.canvas_y_offset) / self.zoom_level

        # Update offsets to maintain the center point's position after zooming
        self.canvas_x_offset += center_x_img * (self.zoom_level - new_zoom_level)
        self.canvas_y_offset += center_y_img * (self.zoom_level - new_zoom_level)

        self.zoom_level = new_zoom_level
        self.redraw_canvas()

    def select_video(self):
        p = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if p:
            self.video_path.set(p)
            self.check_paths()
            self.save_config()

    def select_model(self):
        p = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")])
        if p:
            self.model_path.set(p)
            self.check_paths()
            self.save_config()

    def select_output_dir(self):
        p = filedialog.askdirectory()
        if p:
            self.output_dir.set(p)
            self.check_paths()
            self.save_config()

    def check_paths(self):
        if self.video_path.get() and self.model_path.get() and self.output_dir.get():
            if hasattr(self, 'load_button'):
                self.load_button.config(state="normal")
        else:
            if hasattr(self, 'load_button'):
                self.load_button.config(state="disabled")

    def load_assets(self):
        try:
            self.status_label.config(text="正在加载模型...")
            self.root.update_idletasks()
            self.model = YOLO(self.model_path.get())
            self.class_names = list(self.model.names.values())

            self.cap = cv2.VideoCapture(self.video_path.get())
            if not self.cap.isOpened():
                raise Exception("无法打开视频文件")

            os.makedirs(self.output_dir.get(), exist_ok=True)
            self.repredict_btn.config(state="normal")
            self.add_box_btn.config(state="normal")
            self.clear_btn.config(state="normal")
            self.save_next_btn.config(state="normal")
            self.skip_next_btn.config(state="normal")

            self.status_label.config(text="加载成功，请点击'保存并下一张'开始处理。")
            self.navigate_frames(1, save=False)
        except Exception as e:
            messagebox.showerror("加载失败", f"{e}")

    def re_predict_frame(self):
        if self.current_frame_img is None:
            messagebox.showwarning("无图像", "没有加载当前帧")
            return
        self.save_config()
        self.process_and_display_frame(self.current_frame_img, self.current_frame_num, predict=True)

    def process_and_display_frame(self, frame, frame_num, predict=True):
        self.current_frame_img = frame.copy()
        self.display_image(frame, self.orig_canvas, is_detect=False)

        # Reset zoom and pan for new frame
        self.zoom_level = 1.0
        self.canvas_x_offset = 0
        self.canvas_y_offset = 0
        self.detect_canvas.config(cursor="crosshair")

        if predict:
            res = self.model.predict(frame,
                                     imgsz=self.img_size.get(),
                                     conf=self.conf_threshold.get(),
                                     verbose=False)
            boxes = res[0].boxes.data.cpu().numpy() if res[0].boxes is not None else []
            self.detections = []
            for i, b in enumerate(boxes):
                x1, y1, x2, y2, conf, cls = b
                self.detections.append({
                    "id": i, "label": self.model.names[int(cls)],
                    "coords": [x1, y1, x2, y2], "conf": conf
                })

        self.redraw_canvas()
        self.update_detection_list()
        self.update_status()

    def redraw_canvas(self):
        self.detect_canvas.delete("all")
        if self.current_frame_img is None:
            return

        h, w, _ = self.current_frame_img.shape
        canvas_w, canvas_h = self.detect_canvas.winfo_width(), self.detect_canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 800, 600

        # Calculate base scale and offsets
        base_scale = min(canvas_w / w, canvas_h / h)
        img_w, img_h = int(w * base_scale), int(h * base_scale)

        # Apply zoom
        zoom_w, zoom_h = int(img_w * self.zoom_level), int(img_h * self.zoom_level)
        resized = cv2.resize(self.current_frame_img, (zoom_w, zoom_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.detect_imgtk = imgtk  # Prevent garbage collection

        # Calculate initial centering offsets
        initial_x_offset = (canvas_w - img_w) / 2
        initial_y_offset = (canvas_h - img_h) / 2

        # Apply pan
        final_x = self.canvas_x_offset + initial_x_offset
        final_y = self.canvas_y_offset + initial_y_offset

        self.detect_canvas.create_image(final_x, final_y, anchor="nw", image=imgtk)

        # Redraw detections
        for det in self.detections:
            x1, y1, x2, y2 = det["coords"]
            cx1, cy1 = self.to_canvas(x1, y1)
            cx2, cy2 = self.to_canvas(x2, y2)
            color = "cyan" if det["id"] == self.selected_detection_id else "red"
            self.detect_canvas.create_rectangle(
                cx1, cy1, cx2, cy2,
                outline=color, width=2, tags=("detection", f"det_{det['id']}"))
            self.detect_canvas.create_text(
                cx1, cy1 - 10,
                text=f"{det['label']} ({det['conf']:.2f})",
                fill=color, anchor="nw",
                tags=("detection",))

    def display_image(self, img, canvas, is_detect=False):
        canvas.delete("all")
        h, w, _ = img.shape
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 800, 600

        scale_x = cw / w
        scale_y = ch / h
        scale = min(scale_x, scale_y)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))

        if is_detect:
            self.detect_imgtk = imgtk
        else:
            self.orig_imgtk = imgtk

        x_offset = (cw - new_w) / 2
        y_offset = (ch - new_h) / 2

        canvas.create_image(x_offset, y_offset, anchor="nw", image=imgtk)
        canvas.config(scrollregion=canvas.bbox(tk.ALL))

    def to_canvas(self, x, y):
        h, w, _ = self.current_frame_img.shape
        canvas_w, canvas_h = self.detect_canvas.winfo_width(), self.detect_canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 800, 600

        base_scale = min(canvas_w / w, canvas_h / h)
        initial_x_offset = (canvas_w - w * base_scale) / 2
        initial_y_offset = (canvas_h - h * base_scale) / 2

        return x * base_scale * self.zoom_level + self.canvas_x_offset + initial_x_offset, \
               y * base_scale * self.zoom_level + self.canvas_y_offset + initial_y_offset

    def to_image(self, cx, cy):
        h, w, _ = self.current_frame_img.shape
        canvas_w, canvas_h = self.detect_canvas.winfo_width(), self.detect_canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 800, 600

        base_scale = min(canvas_w / w, canvas_h / h)
        initial_x_offset = (canvas_w - w * base_scale) / 2
        initial_y_offset = (canvas_h - h * base_scale) / 2

        return (cx - self.canvas_x_offset - initial_x_offset) / (base_scale * self.zoom_level), \
               (cy - self.canvas_y_offset - initial_y_offset) / (base_scale * self.zoom_level)

    def navigate_frames(self, direction, save=True):
        if self.cap is None:
            messagebox.showwarning("未加载", "请先加载视频和模型")
            return

        if save:
            self.save_annotations()

        if direction == 1:
            last_frame = self.history[-1] if self.history else -self.frame_interval.get()
            target_frame_num = last_frame + self.frame_interval.get()
            predict = True
        else:
            if len(self.history) <= 1:
                return
            self.history.pop()
            target_frame_num = self.history[-1]
            predict = False

        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if target_frame_num >= total:
            self.status_label.config(text="视频处理结束！")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="读取失败或结束")
            return

        self.current_frame_num = target_frame_num
        if direction == 1:
            self.history.append(target_frame_num)

        json_path = os.path.join(self.output_dir.get(), f"frame_{target_frame_num}.json")
        if os.path.exists(json_path):
            self.load_annotations(json_path)
            predict = False

        self.process_and_display_frame(frame, target_frame_num, predict=predict)
        self.prev_btn.config(state="normal" if len(self.history) > 1 else "disabled")

    def save_annotations(self):
        self.save_config()
        if not self.detections:
            p = os.path.join(self.output_dir.get(), f"frame_{self.current_frame_num}.json")
            if os.path.exists(p):
                os.remove(p)
            return

        img_name = f"frame_{self.current_frame_num}.jpg"
        img_path = os.path.join(self.output_dir.get(), img_name)
        cv2.imwrite(img_path, self.current_frame_img)
        h, w, _ = self.current_frame_img.shape

        shapes = []
        for det in self.detections:
            x1, y1, x2, y2 = det["coords"]
            shapes.append({
                "label": det["label"],
                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "group_id": None, "shape_type": "rectangle", "flags": {}
            })

        with open(os.path.join(self.output_dir.get(),
                               f"frame_{self.current_frame_num}.json"),
                  'w', encoding='utf-8') as f:
            json.dump({
                "version": "5.3.1", "flags": {}, "shapes": shapes,
                "imagePath": img_name, "imageData": None,
                "imageHeight": h, "imageWidth": w
            }, f, indent=2)

    def load_annotations(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.detections = []
        for i, s in enumerate(data['shapes']):
            p1 = s['points'][0]
            p2 = s['points'][1]
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            self.detections.append({
                "id": i, "label": s['label'],
                "coords": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
                "conf": 1.0
            })

    def on_press(self, event):
        # Image coordinates of cursor
        ix, iy = self.to_image(event.x, event.y)
        self.selected_detection_id = None
        self.panning = False

        if self.add_box_mode:
            # Start drawing new box
            self.drawing = True
            self.new_box_start_img = (ix, iy)
            x0, y0 = self.to_canvas(ix, iy)
            self.temp_box_id = self.detect_canvas.create_rectangle(
                x0, y0, x0, y0,
                outline="green", width=2, dash=(4, 4),
                tags="detection")
            self.detect_canvas.config(cursor="plus")
        else:
            item = self.get_item_at_cursor(event)
            if item:
                # Start dragging a box
                self.selected_detection_id = item["id"]
                self.start_pos_img = (ix, iy)
                self.start_coords = item["coords"].copy()
                self.detect_canvas.config(cursor="fleur")
            else:
                # Start panning
                self.panning = True
                self.pan_start_x = event.x
                self.pan_start_y = event.y
                self.detect_canvas.config(cursor="fleur")

        self.redraw_canvas()
        self.update_detection_list()

    def on_drag(self, event):
        # Image coordinates of cursor
        ix, iy = self.to_image(event.x, event.y)

        if self.panning:
            # Pan the canvas
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            self.canvas_x_offset += dx
            self.canvas_y_offset += dy
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.redraw_canvas()

        elif self.drawing and self.add_box_mode:
            # Draw temporary box
            x0, y0 = self.to_canvas(*self.new_box_start_img)
            self.detect_canvas.coords(self.temp_box_id, x0, y0, event.x, event.y)

        elif self.selected_detection_id is not None:
            # Move selected box
            dx = ix - self.start_pos_img[0]
            dy = iy - self.start_pos_img[1]
            det = next(d for d in self.detections if d["id"] == self.selected_detection_id)
            x1, y1, x2, y2 = self.start_coords
            det["coords"] = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
            self.redraw_canvas()

    def on_release(self, event):
        self.panning = False
        self.detect_canvas.config(cursor="crosshair")

        if self.drawing and self.add_box_mode:
            self.drawing = False
            self.detect_canvas.delete(self.temp_box_id)
            x0, y0 = self.new_box_start_img
            x1, y1 = self.to_image(event.x, event.y)
            cls = self.ask_class_choice()
            if cls:
                nid = max([d['id'] for d in self.detections] + [-1]) + 1
                self.detections.append({
                    "id": nid, "label": cls,
                    "coords": [min(x0, x1), min(y0, y1),
                               max(x0, x1), max(y0, y1)],
                    "conf": 1.0
                })
            self.toggle_add_box_mode()
            self.redraw_canvas()
            self.update_detection_list()

    def on_right_press(self, event):
        item = self.get_item_at_cursor(event)
        if item:
            cls = self.ask_class_choice()
            if cls:
                item['label'] = cls
                self.redraw_canvas()
                self.update_detection_list()

    def get_item_at_cursor(self, event):
        items = self.detect_canvas.find_overlapping(event.x - 2,
                                                    event.y - 2,
                                                    event.x + 2,
                                                    event.y + 2)
        for iid in reversed(items):
            tags = self.detect_canvas.gettags(iid)
            for t in tags:
                if t.startswith("det_"):
                    det_id = int(t.split("_")[1])
                    return next((d for d in self.detections if d["id"] == det_id), None)
        return None

    def ask_class_choice(self):
        if not self.class_names:
            return simpledialog.askstring("输入类别", "请输入类别:")
        dlg = tk.Toplevel(self.root)
        dlg.title("选择类别")
        tk.Label(dlg, text="请选择类别:").pack(padx=20, pady=10)
        lb = tk.Listbox(dlg)
        for n in self.class_names:
            lb.insert(tk.END, n)
        lb.pack(padx=20, pady=5)
        res = tk.StringVar()

        def on_ok():
            sel = lb.curselection()
            if sel:
                res.set(self.class_names[sel[0]])
            dlg.destroy()

        ttk.Button(dlg, text="确定", command=on_ok).pack(pady=10)
        dlg.transient(self.root)
        dlg.grab_set()
        self.root.wait_window(dlg)
        return res.get()

    def toggle_add_box_mode(self):
        self.add_box_mode = not self.add_box_mode
        if self.add_box_mode:
            self.detect_canvas.config(cursor="plus")
            self.add_box_btn.config(text="取消添加")
        else:
            self.detect_canvas.config(cursor="crosshair")
            self.add_box_btn.config(text="添加新框")

    def cancel_add_mode(self, event=None):
        if self.add_box_mode:
            self.toggle_add_box_mode()

    def delete_selected(self, event=None):
        if self.selected_detection_id is not None:
            self.detections = [d for d in self.detections
                               if d["id"] != self.selected_detection_id]
            self.selected_detection_id = None
            self.redraw_canvas()
            self.update_detection_list()

    def clear_all_detections(self):
        if messagebox.askokcancel("确认", "清除所有标注?"):
            self.detections.clear()
            self.selected_detection_id = None
            self.redraw_canvas()
            self.update_detection_list()

    def on_list_select(self, _):
        idx = self.detection_list.curselection()
        if idx:
            did = self.detections[idx[0]]["id"]
            self.selected_detection_id = did
            self.redraw_canvas()

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
        self.status_label.config(
            text=f"当前第 {self.current_frame_num} 帧 - 共检测到 {len(self.detections)} 个目标"
        )


if __name__ == '__main__':
    root = tk.Tk()
    app = YoloAnnotator(root)
    root.mainloop()
