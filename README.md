# Football Match Object Detection
This project uses object detection to detect key elements in a football match:

1 - Players

2 - Referee

3 - Ball

Despite a crowd and many distractions, the model successfully detects only the relevant objects.

Additionally, the project tracks a specific player's coordinates and saves them into a CSV file for further analysis.

# Features
 1 - Detects players, referees, and the ball with high accuracy.

2 - Ignores unrelated people and background noise.

3 - Extracts and saves the position (bounding box coordinates) of one target player to a CSV file.

4 - It could be extended to full tracking and analytics for matches.

# Tech Stack
1 - Python

2 - OpenCV

3 - YOLO (or specify the model you used, e.g., YOLOv8)

4 - Pandas (for handling CSV files)
