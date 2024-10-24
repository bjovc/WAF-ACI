Videos are uploaded [here](https://drive.google.com/drive/folders/1XeGM1-IeaCfuifsB0Q6NtWaVfl5mfO4p?usp=sharing).

**1_extract_data.py**: tracks the objects of interest, their positions and distances.
**2_data_processing**: calculates the speed for each object.

**data**: folder with the CSVs for all the videos. Inside **preprocessed_data** are the dataframes after adding the estimated speeds of all objects.

Each row of the dataframe represents a frame of the video.
The columns of the dataframes are:

- **frame**: id of the frame starting from 0.
- **frame_bboxes**: list of all the detections in the frame. Each detection stores detection_id, class_name and bounding box(x1,y1,x2,y2).
- **distances**: list of distances of every individual sheep to the detected dog. It is calculated using the center of the bbox as reference.
- **min_distance**: minimum sheep-dog distance.
- **max_distance**: maximum sheep-dog distance.
- **avg_distance**: mean sheep-dog distance.
- **furthest_sheep_distance**: maximum distance between 2 sheep.
- **speeds**: list of the estimated speed of every sheep.
- **dog_speed**: the estimated speed of the dog.
- **min_speed**: the speed of the slowest detected sheep in the herd.
- **max_speed**: the speed of the fastest detected sheep in the herd.
- **avg_speed**: the mean speed of the herd.


For statistic analisis with JASP, the CSVs have to be combined into one. 
