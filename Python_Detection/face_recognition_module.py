import face_recognition
import os
import sys
import numpy as np
import cv2
import math
from PIL import Image

def face_confidence(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        return str(round((1.0 - face_distance) / ((1.0 - face_match_threshold) * 2) * 100, 2)) + "%"
    else:
        return str(round(((1.0 - face_distance) / ((1.0 - face_match_threshold) * 2) + (1.0 - (1.0 - face_distance) / ((1.0 - face_match_threshold) * 2)) * ((face_distance - 0.5) * 2) ** 0.2) * 100, 2)) + "%"

class FaceRecognition:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_current_frame = True
        self.encode_faces()

    def encode_faces(self):
        """Loads and encodes known faces from the 'faces' directory."""
        if not os.path.exists('faces'):
            print("Error: 'faces' directory not found. Create one and add images.")
            sys.exit(1)

        for image in os.listdir('faces'):
            try:
                # Load image and convert to RGB format
                face_image = Image.open(f'faces/{image}').convert('RGB')
                face_image = np.array(face_image)
                
                # Find face locations in the image
                face_locations = face_recognition.face_locations(face_image)
                if len(face_locations) == 0:
                    print(f"Warning: No face found in {image}. Skipping...")
                    continue
                
                # Get face encodings
                face_encoding = face_recognition.face_encodings(face_image, face_locations)[0]
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(os.path.splitext(image)[0])
                
            except Exception as e:
                print(f"Error processing {image}: {e}")

        if len(self.known_face_encodings) == 0:
            print("Error: No valid faces were found in the 'faces' directory.")
            sys.exit(1)

        print("Loaded known faces:", self.known_face_names)

    def run_recognition(self):
        """Starts video capture and performs face recognition in real-time."""
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit("Error: Could not access the camera.")

        while True:
            ret, frame = video_capture.read()
            if not ret or frame is None:
                print("Warning: Frame capture failed. Retrying...")
                continue

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.process_current_frame:
                small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)

                # Find all face locations in the current frame
                self.face_locations = face_recognition.face_locations(small_frame)
                
                # Get face encodings for the current frame
                self.face_encodings = face_recognition.face_encodings(small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                    name = "Unknown"
                    confidence = "Unknown"

                    if True in matches:
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f"{name} ({confidence})")

            self.process_current_frame = not self.process_current_frame

            # Draw the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()