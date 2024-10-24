import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8l.pt') # Con yolov8l no detecta las ovejas si están muy lejos, pero yolov8x no detecta al perro robot como motorcycle

video_path = "sheeppen_1_30fps.mp4"
cap = cv2.VideoCapture(video_path)

target_classes = ['dog', 'sheep', 'motorcycle']
class_names = model.names
target_class_indices = [idx for idx, name in class_names.items() if name in target_classes]

def plot_results_without_id(frame, results, target_class_indices):
    for result in results:
        for box in result.boxes:
            if int(box.cls) in target_class_indices:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_name = class_names[int(box.cls)]
                bbox_id = int(box.id)
                label = f"{class_name} {bbox_id}"
                
                if class_name == 'sheep':
                    color = (232, 210, 44)  # Blue 
                elif class_name == 'dog':
                    color = (52, 52, 235)  # Red
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                font_scale = 0.5 
                font_thickness = 1
                ((text_width, text_height), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness)
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), font_thickness)
    return frame


while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)

        # Filter results to only include 'dog' and 'sheep'
        filtered_boxes = []
        for result in results:
            filtered_boxes.extend([box for box in result.boxes if int(box.cls) in target_class_indices])

        results[0].boxes = filtered_boxes
        annotated_frame = plot_results_without_id(frame, results, target_class_indices)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()