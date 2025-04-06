import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime

"""DLIB MODELS"""
# Dlib's frontal face detector 
detector = dlib.get_frontal_face_detector()

# Dlib's shape predictor
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_5_face_landmarks.dat')

# Using Resnet to get the 128D vector
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

"""DATABASE SETUP"""
# Create a connection to the sql database
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

# Create a table for attendance with name and registration number columns
table_name = "attendance"
create_table_sql = f"""CREATE TABLE IF NOT EXISTS {table_name} (
    name TEXT, 
    registration_number TEXT,
    time TEXT, 
    date DATE, 
    UNIQUE(registration_number, date)
)"""
cursor.execute(create_table_sql)

# Commit changes and close the connection
conn.commit()
conn.close()

class FaceRecognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC

        # FPS tracking
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # Frame counter
        self.frame_cnt = 0

        # Database faces
        self.face_features_known_list = []
        self.face_name_known_list = []
        self.face_reg_known_list = []  # New list to store registration numbers

        # Centroid tracking
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # Name tracking
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []
        self.last_frame_face_reg_list = []         # New list for registration tracking
        self.current_frame_face_reg_list = []      # New list for registration tracking

        # Face counts
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # Recognition data
        self.eDistanceX = []
        self.current_frame_face_position_list = []
        self.current_frame_face_feature_list = []
        self.last_current_frame_centroid_e_distance = 0

        # Reclassify interval
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10
        
        # List to store recognized people information for display
        self.recognized_people = []  # Will store tuples of (name, reg_number)

    # Get known faces from features_all.csv
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                
                # Parse name and registration from format: Name_RegNumber
                full_id = csv_rd.iloc[i][0]
                if '_' in full_id:
                    name, reg_number = full_id.split('_', 1)
                else:
                    name = full_id
                    reg_number = "Unknown"
                
                self.face_name_known_list.append(name)
                self.face_reg_known_list.append(reg_number)
                
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'feature_extraction_to_csv.py' before 'attendance_tracker.py'")
            return 0

    def update_fps(self):
        now = time.time()
        # Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    # Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            # Compute e-distance with objects in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])
                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]
            self.current_frame_face_reg_list[i] = self.last_frame_face_reg_list[last_frame_num]

    # Display information on the window without bounding boxes
    def draw_note(self, img_rd):
        # Add info on window
        cv2.putText(img_rd, "Face Recognition Attendance System", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display recognized people (name and registration number) at the bottom of the screen
        for i, person in enumerate(self.recognized_people):
            name, reg_number = person
            if name != "unknown":
                # Display name and registration number
                cv2.putText(img_rd, f"Name: {name}", (20, 190 + i*60), self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img_rd, f"ID: {reg_number}", (20, 220 + i*60), self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

    # Insert data in database
    def attendance(self, name, reg_number):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        # Check if the registration number already has an entry for the current date
        cursor.execute("SELECT * FROM attendance WHERE registration_number = ? AND date = ?", (reg_number, current_date))
        existing_entry = cursor.fetchone()

        if existing_entry:
            print(f"{name} ({reg_number}) is already marked as present for {current_date}")
        else:
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            cursor.execute("INSERT INTO attendance (name, registration_number, time, date) VALUES (?, ?, ?, ?)", 
                          (name, reg_number, current_time, current_date))
            conn.commit()
            print(f"{name} ({reg_number}) marked as present for {current_date} at {current_time}")

        conn.close()

    # Face detection and recognition without displaying bounding boxes
    def process(self, stream):
        # 1. Get faces known from "features_all.csv"
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                ret, img_rd = stream.read()
                if not ret:
                    break
                kk = cv2.waitKey(1)

                # 2. Detect faces for frame X
                faces = detector(img_rd, 0)

                # 3. Update cnt for faces in frames
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)

                # 4. Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]
                self.last_frame_face_reg_list = self.current_frame_face_reg_list[:]

                # 5. update frame centroid list
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                # Clear the recognized people list
                self.recognized_people = []

                # 6.1 if cnt not changes
                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                        self.reclassify_interval_cnt != self.reclassify_interval):
                    logging.debug("scene 1: No face cnt changes in this frame!!!")

                    self.current_frame_face_position_list = []
                    if "unknown" in self.current_frame_face_name_list:
                        self.reclassify_interval_cnt += 1

                    if self.current_frame_face_cnt != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            # No longer drawing bounding box

                    # Multi-faces in current frame, use centroid-tracker to track
                    if self.current_frame_face_cnt != 1:
                        self.centroid_tracker()

                    # Add recognized names to our list
                    for i, name in enumerate(self.current_frame_face_name_list):
                        if name != "unknown":
                            reg_number = self.current_frame_face_reg_list[i]
                            person_info = (name, reg_number)
                            if person_info not in self.recognized_people:
                                self.recognized_people.append(person_info)
                            
                    self.draw_note(img_rd)

                # 6.2 If cnt of faces changes, 0->1 or 1->0 or ...
                else:
                    logging.debug("scene 2: Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.eDistanceX = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt = 0

                    # 6.2.1 Face cnt decreases: 1->0, 2->1, ...
                    if self.current_frame_face_cnt == 0:
                        logging.debug(" No faces in this frame!!!")
                        # clear list of names and features
                        self.current_frame_face_name_list = []
                        self.current_frame_face_reg_list = []
                    # 6.2.2 Face cnt increase: 0->1, 0->2, ..., 1->2, ...
                    else:
                        logging.debug(" scene 2.2 Get faces in this frame and do face recognition")
                        self.current_frame_face_name_list = []
                        self.current_frame_face_reg_list = []
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))
                            self.current_frame_face_name_list.append("unknown")
                            self.current_frame_face_reg_list.append("unknown")

                        # 6.2.2.1 Traverse all the faces in the database
                        for k in range(len(faces)):
                            logging.debug(" For face %d in current frame:", k + 1)
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            self.eDistanceX = []

                            # 6.2.2.2 Positions of faces captured
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # 6.2.2.3 For every faces detected, compare the faces in the database
                            for i in range(len(self.face_features_known_list)):
                                if str(self.face_features_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_feature_list[k],
                                        self.face_features_known_list[i])
                                    logging.debug(" with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                    self.eDistanceX.append(e_distance_tmp)
                                else:
                                    self.eDistanceX.append(999999999)

                            # 6.2.2.4 Find the one with minimum e distance
                            similar_person_num = self.eDistanceX.index(
                                min(self.eDistanceX))

                            if min(self.eDistanceX) < 0.55:
                                self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                                self.current_frame_face_reg_list[k] = self.face_reg_known_list[similar_person_num]
                                
                                logging.debug(" Face recognition result: %s (%s)",
                                              self.face_name_known_list[similar_person_num],
                                              self.face_reg_known_list[similar_person_num])
                                
                                # Add to recognized people list if not already there
                                name = self.face_name_known_list[similar_person_num]
                                reg_number = self.face_reg_known_list[similar_person_num]
                                person_info = (name, reg_number)
                                if person_info not in self.recognized_people:
                                    self.recognized_people.append(person_info)

                                # Insert attendance record
                                self.attendance(name, reg_number)
                            else:
                                logging.debug(" Face recognition result: Unknown person")

                        # Draw note with recognized people
                        self.draw_note(img_rd)

                self.update_fps()
                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", img_rd)

                # 8. 'q' / Press 'q' to exit
                if kk == ord('q'):
                    break

                logging.debug("Frame ends\n\n")

    def run(self):
        cap = cv2.VideoCapture(0)  # Get video stream from camera
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()

def main():
    logging.basicConfig(level=logging.INFO)
    FaceRecognizerCon = FaceRecognizer()
    FaceRecognizerCon.run()

if __name__ == '__main__':
    main()