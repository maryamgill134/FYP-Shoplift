# 🛒 Shoplifting Detection Using YOLOv11n

This project detects **shoplifting behavior** in retail store footage using **YOLOv12** — a state-of-the-art deep learning object detection model.  
It distinguishes between **Normal** and **Shoplifting** behavior from live video or CCTV footage in real time.

---

## 🚀 Features

- ⚡ Real-time detection using YOLOv11n
- 📸 Works with webcam or video input
- 🧾 Automatically logs detections to CSV
- 💾 Saves annotated frames and videos
- 🚨 Alerts visually when shoplifting is detected
- ✅ Compatible with Visual Studio Code (Python environment)

---

## 📁 Project Structure

```
Shoplifting-Detection/
│
├── _object_detection.ipynb   # YOLOv11n training notebook
├── detect_shoplift.py                     # Real-time detection script
├── best.pt                                # Trained model weights
├── demo1.mp4                              # Test video file
│
├── detections/                            # Saved detection frames
├── violations/                            # Saved shoplifting frames
├── detections_log.csv                     # CSV log file of detections
│
└── README.md                              # Project documentation
```

---

## 🧩 Requirements

Install dependencies before running:

```bash
pip install ultralytics supervision opencv-python torch torchvision torchaudio
```

If you’re using Visual Studio Code, make sure your Python environment is selected correctly in the bottom-right corner.

---

## ⚙️ How to Train the Model (YOLOv12)

If you already have `best.pt`, skip this section.

1. Open the file **`train_yolov11n_object_detection.ipynb`**.
2. Run all cells step-by-step to:
   - Load the dataset
   - Train YOLOv11n on two classes: **Normal** and **Shoplifting**
   - Save the best model as `best.pt`

The trained weights will appear at:
```
runs/detect/train/weights/best.pt
```

---

## 🧠 Dataset Format

The dataset must contain two labeled classes:
- `shoplifting`
- `normal`

Organize your dataset as:

```
dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
└── valid/
    ├── images/
    └── labels/
```

Each image should have a corresponding `.txt` label file in YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```

---

## 🎥 How to Run Detection

1. Place your trained `best.pt` file in the project folder.  
2. Update your configuration in **`detect_shoplift.py`**:
   ```python
   MODEL_PATH = "best.pt"
   VIDEO_PATH = "demo1.mp4"   # or 0 for webcam
   CONF_THRESH_SHOPLIFT = 0.65
   CONF_THRESH_NORMAL = 0.45
   ```
3. Run in Visual Studio terminal:

```bash
python detect_shoplift.py
```

The script will:
- Detect people and classify behavior
- Highlight detections with bounding boxes
- Save annotated frames and video outputs
- Display alerts when shoplifting is detected

---

## 🧾 Output Files

| File | Description |
|------|--------------|
| `output_detected.mp4` | Video with detection boxes and alerts |
| `detections/` | Frames saved for each detection |
| `violations/` | Frames saved only when shoplifting detected |
| `detections_log.csv` | Logs of detections with timestamp and coordinates |

---

## 🧠 Detection Logic Overview

1. **YOLOv11n** predicts bounding boxes and confidence scores.
2. Script checks:
   - If class = `shoplifting` → high confidence → alert!
   - If class = `normal` → green bounding box (safe)
3. **Stabilization logic**:
   - Requires multiple consecutive detections before triggering an alert
   - Avoids false positives

---

## 🖥️ Visual Studio Code Setup

1. Open the folder in **VS Code**  
2. Create & activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install the requirements:
   ```bash
   pip install ultralytics supervision opencv-python torch torchvision torchaudio
   ```
4. Run your detection script:
   ```bash
   python detect_shoplift.py
   ```

You can view live output in the VS Code terminal and the video window.

---

## 📊 Example Output

| Detection | Description |
|------------|-------------|
| 🟩 `normal (0.78)` | Normal behavior detected |
| 🟥 `shoplifting (0.92)` | 🚨 Shoplifting detected |

---

## 🧱 Troubleshooting

- If **shoplifting is detected as normal**, swap the class IDs in the code:
  ```python
  shoplift_id = 1
  normal_id = 0
  ```
  Then rerun your detection.

- If you get `CUDA` errors, run on CPU:
  ```python
  device = "cpu"
  ```

---

## 🔮 Future Improvements

- Multi-camera detection  
- Cloud storage for detection logs  
- SMS or Email alert system  
- Lightweight deployment on Jetson Nano or Raspberry Pi

---

## 📜 License

This project is for **educational and research purposes only**.  
All datasets and videos should comply with their respective licenses.

---

**👩‍💻 Developed by:** Maryam Fazal Gill  
**📅 Year:** 2025

