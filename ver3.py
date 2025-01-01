import serial
import cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity
from utils.send import send_message
import os
import shutil
import matplotlib.pyplot as plt

# Set ONNX Runtime logging level to ERROR
ort.set_default_logger_severity(3)  # 3 = ERROR, 4 = FATAL

# Initialize Serial Communication
arduino = serial.Serial(port='COM5', baudrate=9600, timeout=1)

MyPhoneNumber = os.getenv('MY_PHONE_NUMBER')
MaxNTries = 4

# Camera Setup
camera = cv2.VideoCapture(1)

# Registered Face Path
registered_faces = "D:\\ElectronicsProject\\RegisterdFaces"
if not os.path.exists(registered_faces):
    print(f"Error: Registered face directory not found at {registered_faces}")
    exit(1)

# Load SFace ONNX Model
onnx_model_path = "D:\\ElectronicsProject\\models\\face_recognition_sface_2021dec_int8_fixed.onnx"
session = ort.InferenceSession(onnx_model_path)

# Model input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_dtype = session.get_inputs()[0].type

# Load Haar Cascade for Face Detection
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

if face_cascade.empty():
    print("Error loading Haar Cascade for face detection.")
    exit(1)

def preprocess_image(image):
    """ Preprocess the image for model input. """
    # Resize to model input size
    resized_image = cv2.resize(image, (input_shape[2], input_shape[3]))

    # Normalize to [-1, 1]
    normalized_image = (resized_image / 127.5) - 1

    # Transpose to channel-first format
    transposed_image = np.transpose(normalized_image, (2, 0, 1))

    # Add batch dimension
    input_tensor = np.expand_dims(transposed_image, axis=0).astype(np.float32)

    return input_tensor

def detect_and_crop_face(image_path):
    """ Detect and crop the largest face in the image. """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    # Select the largest face
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    cropped_face = image[y:y+h, x:x+w]

    return cropped_face

def get_embedding(image_path):
    """ Get the embedding of the cropped face. """
    cropped_face = detect_and_crop_face(image_path)
    input_tensor = preprocess_image(cropped_face)
    embedding = session.run(None, {input_name: input_tensor})[0]

    # Normalize the embedding to unit vector
    normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    return normalized_embedding

def compare_faces(embedding1, embedding2, threshold=0.8):
    """ Compare two face embeddings. """
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity >= threshold, similarity

def capture_image():
    """ Capture an image from the camera. """
    ret, frame = camera.read()
    if ret:
        os.makedirs("./CachedFaces", exist_ok=True)
        img_path = "./CachedFaces/current_face.jpg"
        cv2.imwrite(img_path, frame)
        if os.path.exists(img_path):
            return img_path
        else:
            print("Failed to save the captured image")
            return None
    print("Error: Failed to capture image")
    return None

# Main Loop
while True:
    try:
        # Read serial data if available
        if arduino.in_waiting > 0:
            data = arduino.readline().decode('utf-8').strip()
            print(f"Received: {data}")

            if data == "DETECTED":
                current_image_path = capture_image()
                if current_image_path:
                    unknown = 0
                    match = False
                    for registered_identity in os.listdir(registered_faces):
                        registered_identity_full = os.path.join(registered_faces, registered_identity)
                        # print(f"Verifying: {registered_identity_full}")
                        for registered_face_path in os.listdir(registered_identity_full):
                            registered_image_path = os.path.join(registered_identity_full, registered_face_path)

                            # print(f"Verifying: {registered_face_path}")
                            try:
                                registered_embedding = get_embedding(registered_image_path)
                                current_embedding = get_embedding(current_image_path)

                                # Compare faces
                                match, similarity = compare_faces(registered_embedding, current_embedding)

                                if match:
                                    print(f"Face recognized: {registered_identity}, Similarity: {similarity}")
                                    send_message(f"{registered_identity} has entered!", MyPhoneNumber)
                                    arduino.write(b"1\n")
                                    break
                            except Exception as df_error:
                                print(f"Verification error: {df_error}")
                                continue
                        

                        if not match:
                            unknown += 1
                    if unknown >= len(os.listdir(registered_faces)):
                        send_message("WARNING!! Intruder.", MyPhoneNumber)
                        arduino.write(b"2\n")  # Signal unrecognized face

                        register = input("Do you want to register the face? (y/n): ")
                        if register == "y":
                            name = input("Enter name: ")
                            if not os.path.exists(os.path.join(registered_faces, name)):
                                os.mkdir(os.path.join(registered_faces, name))
                            shutil.copy(current_image_path, os.path.join(registered_faces, name, name + ".jpg"))
                            print(f"Face registered: {name}")
                        else:
                            print("Face not registered.")
                else:
                    print("Failed to capture image. Skipping detection.")
                    arduino.write(b"0\n")  # Signal image failure

    except Exception as main_error:
        print(f"Error: {main_error}")
    finally:
        continue
