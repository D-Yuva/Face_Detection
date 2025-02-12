import face_recognition

# Load a sample picture and learn how to recognize it.
image = face_recognition.load_image_file("path_to_a_sample_image.jpg")
face_encoding = face_recognition.face_encodings(image)[0]

print("Face encoding completed successfully.")