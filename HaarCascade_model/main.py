import os
import sys
import numpy as np
import cv2
from datetime import datetime
import json
import logging
from collections import Counter

class EnhancedFaceRecognition:
    def __init__(self):
        # Initialize cascade classifiers
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
        # Check if cascade files loaded properly
        if self.face_cascade.empty():
            print(f"Error: Couldn't load face cascade from {cascade_path}")
            sys.exit(1)
        if self.eye_cascade.empty():
            print(f"Error: Couldn't load eye cascade from {eye_cascade_path}")
            sys.exit(1)
            
        print("Cascade classifiers loaded successfully")

        # Recognition parameters
        self.known_face_encodings = []
        self.known_face_names = []
        self.recognition_threshold = 0.5  # Threshold

        # Setup directories and load faces
        self.faces_dir = 'faces'
        self.ensure_directories()
        self.encode_faces()

    def ensure_directories(self):
        """Ensure necessary directories exist."""
        os.makedirs(self.faces_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        print("Directories checked/created")

    def extract_face_features(self, face_roi):
        """Extract histogram features from face ROI."""
        try:
            # Resize for consistency
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Calculate histogram features
            hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            return hist.flatten()
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def encode_faces(self):
        """Encode known faces using histogram features."""
        if not os.path.exists(self.faces_dir):
            print("Error: Faces directory not found")
            sys.exit(1)

        face_files = os.listdir(self.faces_dir)
        if not face_files:
            print("Warning: No images found in faces directory")
            return

        print(f"Found {len(face_files)} files in faces directory")

        for image_file in face_files:
            try:
                # Load and preprocess image
                image_path = os.path.join(self.faces_dir, image_file)
                print(f"Processing {image_path}")
                
                face_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if face_image is None:
                    print(f"Could not load image {image_file}")
                    continue
                
                print(f"Image shape: {face_image.shape}")

                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    face_image,
                    scaleFactor=1.3,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                print(f"Found {len(faces)} faces in {image_file}")

                if len(faces) == 0:
                    print(f"No face found in {image_file}")
                    continue

                # Get largest face
                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                print(f"Selected face region: x={x}, y={y}, w={w}, h={h}")
                
                face_roi = face_image[y:y+h, x:x+w]
                
                # Extract features
                features = self.extract_face_features(face_roi)
                if features is not None:
                    self.known_face_encodings.append(features)
                    self.known_face_names.append(os.path.splitext(image_file)[0])
                    print(f"Successfully encoded: {image_file}")
                else:
                    print(f"Failed to extract features from {image_file}")

            except Exception as e:
                print(f"Error processing {image_file}: {e}")

        print(f"Successfully loaded {len(self.known_face_names)} faces: {self.known_face_names}")

    def compare_faces(self, face_encoding):
        """Compare face encoding with known faces using histogram correlation."""
        if len(self.known_face_encodings) == 0:
            return "Unknown", 0

        similarities = []
        for i, known_encoding in enumerate(self.known_face_encodings):
            score = cv2.compareHist(
                face_encoding.reshape(-1, 1),
                known_encoding.reshape(-1, 1),
                cv2.HISTCMP_CORREL
            )
            print(f"Comparing with {self.known_face_names[i]}: score = {score:.3f}")
            similarities.append(score)

        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]
        
        print(f"Best match: {self.known_face_names[best_match_idx]} with score {best_score:.3f}")

        if best_score > self.recognition_threshold:
            return self.known_face_names[best_match_idx], best_score
        return "Unknown", best_score

    def run_recognition(self):
        """Run face recognition with enhanced features."""
        print("Initializing camera...")
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("Error: Could not access camera")
            sys.exit(1)

        print("Starting face recognition. Press 'q' to quit.")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Frame capture failed")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            print(f"Detected faces: {len(faces)}")  # Debug print

            for (x, y, w, h) in faces:
                # Draw rectangle immediately
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                face_roi = gray[y:y+h, x:x+w]
                
                # Extract and compare features
                features = self.extract_face_features(face_roi)
                if features is None:
                    continue

                name, confidence = self.compare_faces(features)
                
                # Draw name and confidence
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = f"{name} ({confidence:.2f})"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        fr = EnhancedFaceRecognition()
        fr.run_recognition()
    except Exception as e:
        print(f"An error occurred: {e}")