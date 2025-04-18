import os
import dlib
import csv
import numpy as np
import logging
import cv2

# Path of cropped faces
path_images_from_camera = "faces"

# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_5_face_landmarks.dat')

# Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Return 128D features for single image
def return_128d_features(path_img):
    img_rd = cv2.imread(path_img)
    faces = detector(img_rd, 1)

    logging.info("%-40s %-20s", " Image with faces detected:", path_img)

    # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        logging.warning("no face")
    return face_descriptor

# Return the mean value of 128D face descriptor for person X
def returnFeatureMeanPersonX(path_face_personX):
    featuresListPersonX = []
    # Directly use the image path
    logging.info("%-40s %-20s", " / Reading image:", path_face_personX)
    features_128d = return_128d_features(path_face_personX)
    # Jump if no face detected from image
    if features_128d != 0:
        featuresListPersonX.append(features_128d)

    if featuresListPersonX:
        featuresMeanPersonX = np.array(featuresListPersonX, dtype=object).mean(axis=0)
    else:
        featuresMeanPersonX = np.zeros(128, dtype=object, order='C')
    return featuresMeanPersonX

def main():
    logging.basicConfig(level=logging.INFO)
    # Get the list of image files in the 'faces' folder
    image_files = [f for f in os.listdir(path_images_from_camera) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Create the directories if they don't exist
    os.makedirs('data', exist_ok=True)
    
    with open("data/features_all.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for image_file in image_files:
            # Keep the full identifier (Name_RegNumber) as stored in the filename
            person_id = os.path.splitext(image_file)[0]
            image_path = os.path.join(path_images_from_camera, image_file)

            logging.info("Processing image: %s", image_path)
            logging.info("Person ID: %s", person_id)
            
            featuresMeanPersonX = returnFeatureMeanPersonX(image_path)

            # Insert the full identifier as the first element
            featuresMeanPersonX = np.insert(featuresMeanPersonX, 0, person_id, axis=0)
            # featuresMeanPersonX will be 129D, person identifier + 128 features
            writer.writerow(featuresMeanPersonX)
            logging.info('\n')
        logging.info("Save all the features of faces registered into: data/features_all.csv")

if __name__ == '__main__':
    main()