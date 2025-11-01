# import cv2
# import os
# import datetime
# from ultralytics import YOLO

# # Load your trained YOLOv11 model
# model = YOLO("shoplifting_wights.pt")

# # Create folder for saving detected frames
# os.makedirs("detections", exist_ok=True)

# # Open webcam (use 0 for laptop cam, or a video path)
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run YOLO detection
#     results = model(frame)

#     # Draw detection boxes and labels
#     annotated_frame = results[0].plot()

#     # Optional: Save detections when something is detected
#     if len(results[0].boxes) > 0:
#         timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         filename = f"detections/detect_{timestamp}.jpg"
#         cv2.imwrite(filename, annotated_frame)
#         print(f"🧠 Detected something at {timestamp}")

#     # Display the result in a window
#     cv2.imshow("Shoplift Detection - Press 'q' to quit", annotated_frame)

#     # Quit when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# import cv2
# import os
# import datetime
# import csv
# from ultralytics import YOLO
# import torch

# # ==============================
# # 🧩 CONFIGURATION
# # ==============================
# MODEL_PATH = "shoplifting_wights.pt"               # Path to your YOLOv11 model
# VIDEO_PATH = "demo1.mp4"             # 0 for webcam
# SAVE_FRAMES = True
# SAVE_VIDEO = True
# CSV_LOG = True

# # ✅ Confidence thresholds
# CONF_THRESH_NORMAL = 0.45
# CONF_THRESH_SHOPLIFT = 0.65

# # ✅ Stabilization
# STABILIZATION_FRAMES = 5

# # ==============================
# # ⚙️ SETUP
# # ==============================
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"⚙️ Using device: {device}")

# # Load YOLO model
# try:
#     model = YOLO(MODEL_PATH)
#     model.to(device)
# except Exception as e:
#     print(f"❌ Error loading model: {e}")
#     exit()

# # ✅ Automatically detect class names from model
# class_names = model.names
# print("🧾 Model Classes:", class_names)

# # Detect index for shoplifting vs normal automatically
# shoplift_id = None
# normal_id = None
# for cid, name in class_names.items():
#     lname = name.lower()
#     if "shop" in lname or "theft" in lname:
#         shoplift_id = cid
#     elif "normal" in lname or "regular" in lname or "customer" in lname:
#         normal_id = cid

# if shoplift_id is None or normal_id is None:
#     print("⚠️ Could not auto-detect class names properly. Please check model.names.")
#     print("Example:", class_names)
#     exit()

# print(f"✅ Mapping confirmed: Shoplifting={shoplift_id}, Normal={normal_id}")

# # Prepare folders
# os.makedirs("detections", exist_ok=True)

# # Load video or webcam
# cap = cv2.VideoCapture(VIDEO_PATH)
# if not cap.isOpened():
#     print(f"❌ Cannot open video: {VIDEO_PATH}")
#     exit()

# fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
# width, height = int(cap.get(3)), int(cap.get(4))

# if SAVE_VIDEO:
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter("output_detected.mp4", fourcc, fps, (width, height))

# if CSV_LOG:
#     csv_file = open("detections_log.csv", mode="w", newline="")
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(["Timestamp", "Class", "Confidence", "X1", "Y1", "X2", "Y2"])

# print("🚀 Starting detection... Press 'q' to quit.")

# # ==============================
# # 🎥 DETECTION LOOP
# # ==============================
# shoplift_counter = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("✅ Video processing complete.")
#         break

#     results = model.predict(frame, verbose=False, device=device)
#     boxes = results[0].boxes
#     annotated_frame = frame.copy()
#     frame_has_shoplift = False

#     if len(boxes) > 0:
#         timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#         for box in boxes:
#             cls_id = int(box.cls[0])
#             conf = float(box.conf[0])
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             class_name = class_names.get(cls_id, f"Unknown({cls_id})")

#             # ✅ Apply class-based logic
#             if cls_id == shoplift_id and conf >= CONF_THRESH_SHOPLIFT:
#                 color = (0, 0, 255)  # Red
#                 frame_has_shoplift = True
#             elif cls_id == normal_id and conf >= CONF_THRESH_NORMAL:
#                 color = (0, 255, 0)  # Green
#             else:
#                 continue

#             # ✅ Draw bounding boxes
#             cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(
#                 annotated_frame,
#                 f"{class_name} ({conf:.2f})",
#                 (x1, max(30, y1 - 10)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 color,
#                 2,
#             )

#             print(f"[{timestamp}] {class_name} ({conf:.2f})")
#             if CSV_LOG:
#                 csv_writer.writerow([timestamp, class_name, conf, x1, y1, x2, y2])

#         if SAVE_FRAMES:
#             cv2.imwrite(f"detections/detect_{timestamp}.jpg", annotated_frame)

#     # ==============================
#     # 🚨 SHOPLIFT ALERT CONTROL
#     # ==============================
#     if frame_has_shoplift:
#         shoplift_counter += 1
#     else:
#         shoplift_counter = max(shoplift_counter - 1, 0)

#     if shoplift_counter >= STABILIZATION_FRAMES:
#         cv2.putText(
#             annotated_frame,
#             "⚠️ SHOPLIFT DETECTED!",
#             (50, 80),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1.2,
#             (0, 0, 255),
#             3,
#         )

#     if SAVE_VIDEO:
#         out.write(annotated_frame)

#     cv2.imshow("🛒 Shoplifting Detection - Press 'q' to quit", annotated_frame)
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break

# # ==============================
# # 🧹 CLEANUP
# # ==============================
# cap.release()
# if SAVE_VIDEO:
#     out.release()
# if CSV_LOG:
#     csv_file.close()
# cv2.destroyAllWindows()

# print("\n✅ Detection finished successfully.")
# print("📁 Outputs:")
# print("  - output_detected.mp4 → Annotated video")
# print("  - detections/ → Saved frames")
# print("  - detections_log.csv → CSV detection log")

# from ultralytics import YOLO
# import cv2
# import os

# # ✅ Force correct class mapping
# # If your dataset.yaml has: names: ['Normal', 'Shoplifting']
# # but detections come reversed, flip here:
# FORCE_MAPPING = {0: 'Shoplifting', 1: 'Normal'}

# # ✅ Load YOLO model
# model = YOLO("best.pt")  # replace with your trained model path

# # ✅ Test video path
# video_path = "demo.mp4"  # replace with your video name
# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print("❌ Error: Could not open video.")
#     exit()

# # ✅ Create output folder
# os.makedirs("detections", exist_ok=True)
# save_path = os.path.join("detections", "output.mp4")

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = None

# # ✅ Process each video frame
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run YOLO detection
#     results = model(frame, verbose=False)[0]

#     for box in results.boxes:
#         cls = int(box.cls)
#         conf = float(box.conf)
#         x1, y1, x2, y2 = map(int, box.xyxy[0])

#         # ✅ Apply corrected labels
#         label = FORCE_MAPPING.get(cls, "Unknown")

#         # ✅ Filter low-confidence false positives
#         if label == "Shoplifting" and conf < 0.65:
#             label = "Normal"  # treat uncertain cases as Normal

#         # ✅ Draw boxes and labels
#         color = (0, 255, 0) if label == "Normal" else (0, 0, 255)
#         text = f"{label} ({conf:.2f})"

#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(frame, text, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     # ✅ Initialize video writer if not already
#     if out is None:
#         h, w = frame.shape[:2]
#         out = cv2.VideoWriter(save_path, fourcc, 20.0, (w, h))

#     # Write frame to output video
#     out.write(frame)

#     # Show in real-time
#     cv2.imshow("Shoplifting Detection", frame)

#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # ✅ Cleanup
# cap.release()
# if out:
#     out.release()
# cv2.destroyAllWindows()

# print(f"✅ Processing complete! Saved results to {save_path}")











from ultralytics import YOLO
import cv2
import os
import time

MODEL_PATH = "best.pt"       
VIDEO_PATH = "demo2.mp4"      
CONF_SHOPLIFT = 0.65         
CONF_NORMAL = 0.40           
OUTPUT_DIR = "detections"     

FORCE_MAPPING = {
    0: "Shoplifting",
    1: "Normal"
}


os.makedirs(OUTPUT_DIR, exist_ok=True)
save_path = os.path.join(OUTPUT_DIR, "output.mp4")

# Load model
model = YOLO(MODEL_PATH)

# Load video or webcam
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Error: Could not open video source.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
w, h = int(cap.get(3)), int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))



print("🚀 Detection started — press 'q' to stop.")

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing complete.")
        break

    frame_id += 1
    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

     
        label = FORCE_MAPPING.get(cls, "Unknown")

        # ✅ Confidence filtering & correction
        if label == "Shoplifting" and conf < CONF_SHOPLIFT:
            label = "Normal"  
        elif label == "Normal" and conf < CONF_NORMAL:
            continue  

        # ✅ Draw boxes and labels
        color = (0, 0, 255) if label == "Shoplifting" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ✅ Optional: Save frame when shoplifting detected
        if label == "Shoplifting":
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"alert_{timestamp}.jpg"), frame)

    # ✅ Save annotated frame to output video
    out.write(frame)
    cv2.imshow("🛒 Shoplifting Detection", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("\n✅ Detection finished successfully!")
print(f"📁 Saved video: {save_path}")
print(f"🖼️ Frames saved in: {OUTPUT_DIR}/")
