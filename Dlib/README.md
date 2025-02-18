# Face Recognition Attendance System

This project implements a real-time face recognition system for tracking attendance using Python, OpenCV, and Dlib. The system captures video from a webcam, recognizes faces, and maintains attendance records in a SQLite database.

### Features

Real-time face detection and recognition
Automated attendance tracking
Database storage of attendance records
Support for multiple faces in a single frame
FPS counter and performance metrics
User-friendly GUI interface

### Prerequisites

Before running this project, ensure you have the following dependencies installed:

    pip install dlib
    pip install opencv-python
    pip install numpy
    pip install pandas

You'll also need the following model files in the data/data_dlib/ directory:
1. shape_predictor_68_face_landmarks.dat
2. dlib_face_recognition_resnet_model_v1.dat

## Project Structure

![Screenshot 2025-02-18 at 12 13 01 PM](https://github.com/user-attachments/assets/e4c5bd27-1022-4340-b253-d9186b56276b)


## Setup Instructions

1. Clone the repository:
   
        git clone Face_Detection
        cd Dlib

2. Create a faces directory and add face images:


Add clear frontal face images of people to be recognized
Name the images as person_name.jpg (e.g., john.jpg)
Images should be well-lit and face should be clearly visible


3. Run the feature extraction script:
   
       python feature_extraction_to_csv.py

4. Start the attendance system:

        python attendance_tracker.py

## Usage

1. Adding New People:
  Add their face image to the faces folder
  Run feature_extraction_to_csv.py to update the features database
  Restart attendance_tracker.py if it's running

2. Taking Attendance:
   Run attendance_tracker.py
   The system will automatically detect and recognize faces
   Attendance is recorded in the SQLite database
   Press 'q' to quit the application

3. Viewing Attendance Records:

   The attendance records are stored in attendance.db
   Use any SQLite browser to view the records
   Records include name, date, and time of attendance


