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
        # Initialize cascade classifiers for face and eyes
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Recognition parameters
        self.known_face_encodings = []
        self.known_face_names = []
        self.recognition_threshold = 0.8
        self.confidence_frames = 5

        # Attendance tracking
        self.attendance_log = {}
        self.attendance_file = "attendance_log.json"
        self.load_attendance_log()

        # Setup logging
        logging.basicConfig(filename='face_recognition.log', level=logging.INFO,
                            format='%(asctime)s:%(levelname)s:%(message)s')

        # Load or create faces directory
        self.faces_dir = 'faces'
        self.ensure_directories()
        self.encode_faces()

    def ensure_directories(self):
        """Ensure necessary directories exist."""
        os.makedirs(self.faces_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('attendance', exist_ok=True)

    def load_attendance_log(self):
        """Load existing attendance log or create new one."""
        try:
            if os.path.exists(self.attendance_file):
                with open(self.attendance_file, 'r') as f:
                    self.attendance_log = json.load(f)
        except Exception as e:
            logging.error(f"Error loading attendance log: {e}")
            self.attendance_log = {}

    def save_attendance_log(self):
        """Save attendance log to file."""
        try:
            with open(self.attendance_file, 'w') as f:
                json.dump(self.attendance_log, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving attendance log: {e}")

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
            logging.error(f"Error extracting face features: {e}")
            return None

    def encode_faces(self):
        """Encode known faces using histogram features."""
        if not os.path.exists(self.faces_dir):
            logging.error("Faces directory not found")
            sys.exit(1)

        for image_file in os.listdir(self.faces_dir):
            try:
                # Load and preprocess image
                image_path = os.path.join(self.faces_dir, image_file)
                face_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if face_image is None:
                    logging.warning(f"Could not load image {image_file}")
                    continue

                # Detect and extract face
                faces = self.face_cascade.detectMultiScale(face_image, 1.1, 5, minSize=(30, 30))
                if len(faces) == 0:
                    logging.warning(f"No face found in {image_file}")
                    continue

                # Get largest face
                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                face_roi = face_image[y:y+h, x:x+w]

                # Extract features
                features = self.extract_face_features(face_roi)
                if features is not None:
                    self.known_face_encodings.append(features)
                    self.known_face_names.append(os.path.splitext(image_file)[0])

            except Exception as e:
                logging.error(f"Error processing {image_file}: {e}")

        logging.info(f"Loaded {len(self.known_face_names)} faces: {self.known_face_names}")

    def verify_face(self, frame, face_roi):
        """Verify if detected ROI is a real face using eye detection."""
        eyes = self.eye_cascade.detectMultiScale(face_roi)
        return len(eyes) >= 2

    def compare_faces(self, face_encoding):
        """Compare face encoding with known faces using histogram correlation."""
        if len(self.known_face_encodings) == 0:
            return "Unknown", 0

        similarities = [cv2.compareHist(face_encoding, known_encoding, cv2.HISTCMP_CORREL)
                        for known_encoding in self.known_face_encodings]

        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]

        if best_score > self.recognition_threshold:
            return self.known_face_names[best_match_idx], best_score
        return "Unknown", best_score

    def update_attendance(self, name):
        """Update attendance log with timestamp."""
        if name != "Unknown":
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.now().strftime("%H:%M:%S")

            if current_date not in self.attendance_log:
                self.attendance_log[current_date] = {}

            if name not in self.attendance_log[current_date]:
                self.attendance_log[current_date][name] = {
                    "first_seen": current_time,
                    "last_seen": current_time
                }
                logging.info(f"Marked attendance for {name}")
            else:
                self.attendance_log[current_date][name]["last_seen"] = current_time

            self.save_attendance_log()

    def run_recognition(self):
        """Run face recognition with enhanced features."""
        video_capture = cv2.VideoCapture(0)
        frame_count = 0
        face_buffer = []

        if not video_capture.isOpened():
            logging.error("Could not access camera")
            sys.exit("Error: Could not access the camera.")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                logging.warning("Frame capture failed")
                continue

            # Process every other frame for performance
            frame_count += 1
            if frame_count % 2 != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]

                # Verify face
                if not self.verify_face(frame, face_roi):
                    continue

                # Extract and compare features
                features = self.extract_face_features(face_roi)
                if features is None:
                    continue

                name, confidence = self.compare_faces(features)

                # Use temporal consistency
                face_buffer.append(name)
                if len(face_buffer) > self.confidence_frames:
                    face_buffer.pop(0)

                if len(face_buffer) == self.confidence_frames:
                    final_name = Counter(face_buffer).most_common(1)[0][0]

                    # Update attendance
                    self.update_attendance(final_name)

                    # Draw results
                    color = (0, 255, 0) if final_name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{final_name} ({confidence:.2f})",
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow('Enhanced Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        fr = EnhancedFaceRecognition()
        fr.run_recognition()
    except Exception as e:
        logging.error(f"Application error: {e}")
        print(f"An error occurred. Check logs for details.")
