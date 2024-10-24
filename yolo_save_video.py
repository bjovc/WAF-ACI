import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8x.pt')

# Open the video file
video_path = "ambasaguas/video6.mp4"
cap = cv2.VideoCapture(video_path)

# Get the video frame width, height, and frames per second (fps)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Write the frame to the output video
        out.write(annotated_frame)

        # Display the annotated frame (optional, can be commented out)
        #cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and writer objects and close the display window
cap.release()
out.release()  # Don't forget to release the VideoWriter object
cv2.destroyAllWindows()
