import cv2
from ultralytics import YOLO
import numpy as np

# font
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
org = (25, 25)
fontScale = 1
color = (255,255,255)
thickness = 2

model = YOLO('yolov8l.pt')
video_path = 'sheeppen.mp4'
cap = cv2.VideoCapture(video_path)

pixel_ratio_array = []
averagePR = 0

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()
        min_distance = float('inf')

        for i in range (len(results[0].boxes.cls)):
            label = results[0].boxes.cls[i] 
            x, y, w, h =  map(int, results[0].boxes.xywh[i])
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = x + h / 2
            bbox = (x1, y1, x2, y2)


            if(label == 3): # Motorcycle (perro robot)
                dog_long_meters = 0.92 #Lo que mide de largo el perro
                dog_long = max(h, w)
                dog_bbox = (x, y, w, h)


                pixel_to_meters = dog_long / dog_long_meters
                pixel_ratio_array.append(pixel_to_meters)

                if(averagePR != 0):
                    object_meters = dog_long / averagePR

                    meters_ORG = (int(x1+5), int(y1+30))
                    annotated_frame = cv2.putText(annotated_frame, 'Meters: ' + str(object_meters)[:4], meters_ORG, font, fontScale, color, thickness, cv2.LINE_AA)

            if(label == 18): # Sheep
                sheep_bbox = (x, y, w, h)

                if(averagePR != 0):
                    object_meters = h / averagePR

                    inches_ORG = (int(x1+5), int(y1+30))
                    annotated_frame = cv2.putText(annotated_frame, 'Meters: ' + str(object_meters)[:4], inches_ORG, font, fontScale, color, thickness, cv2.LINE_AA) 
            

            try:
                # Distancia entre los puntos medios de las bboxes
                distance = np.sqrt((dog_bbox[0] - sheep_bbox[0]) ** 2 + (dog_bbox[1] - sheep_bbox[1]) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    min_distance_point1 = (dog_bbox[0], dog_bbox[1])
                    min_distance_point2 = (sheep_bbox[0], sheep_bbox[1])

                # Dibujar la línea entre los puntos medios del perro robot y la oveja más cercana
                distance_meters = distance / averagePR
                annotated_frame = cv2.line(annotated_frame, min_distance_point1, min_distance_point2, color, thickness)
                mid_point = ((min_distance_point1[0] + min_distance_point2[0]) // 2, (min_distance_point1[1] + min_distance_point2[1]) // 2)
                annotated_frame = cv2.putText(annotated_frame, '~' + str(distance_meters)[:4] + ' m', mid_point, font, fontScale, color, thickness-1, cv2.LINE_AA)
                
            except:
                pass

        averagePR = np.mean(pixel_ratio_array)

        # Display the annotated frame
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
