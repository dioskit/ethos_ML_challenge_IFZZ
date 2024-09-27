import cv2
import os
import sys

# # Adjust the path to point to the folder containing 'hall-of-faces'
project_dir = 'C:/Users/stanz/OneDrive/Desktop/project/hall-of-faces'
sys.path.append(project_dir)
# Now import from 'hof' inside 'hall-of-faces'
from hof.face_detectors import YOLOv2FaceDetector  # Single face detector

# Set paths
input_video_dir = 'raw_videos'
output_faces_dir = 'faces'

# Ensure the output directory exists
os.makedirs(output_faces_dir, exist_ok=True)

# Initialize face detector (YOLOv2 in this case)
detector = YOLOv2FaceDetector(min_confidence=0.5)

def extract_faces_from_video(video_path, output_dir):
    # Open video file
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break  # End of video

        # Detect faces in the frame
        detections = detector.detect(frame)
        
        # Extract each detected face
        for i, detection in enumerate(detections):
            x, y, w, h = detection['box']
            face = frame[y:y+h, x:x+w]
            
            # Save the face as an image
            face_filename = os.path.join(output_dir, f"frame_{frame_count}_face_{i}.png")
            cv2.imwrite(face_filename, face)

        frame_count += 1

    # Release the video capture object
    video_capture.release()

# Iterate over all videos in the input directory
for video_file in os.listdir(input_video_dir):
    if video_file.endswith(('.mp4', '.avi', '.mov')):  # Add other video formats if needed
        video_path = os.path.join(input_video_dir, video_file)
        print(f"Processing {video_path}...")
        extract_faces_from_video(video_path, output_faces_dir)

# gan 
