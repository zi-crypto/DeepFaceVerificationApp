import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import serial
import cv2
from utils.send import send_message
import shutil
from layers import L1Dist
from facenet_pytorch import MTCNN

# Load model
model = tf.keras.models.load_model('./siamesemodel_finetuned_20.h5',
    custom_objects={'L1Dist': L1Dist, 'BinaryCrossEntropy':tf.losses.BinaryCrossentropy})

# Initialize Serial Communication
# arduino = serial.Serial(port='COM5', baudrate=9600, timeout=1)

MyPhoneNumber = os.getenv('MY_PHONE_NUMBER')
MaxNTries = 4

mtcnn = MTCNN(margin=20, keep_all=False, device='cpu')
# Initialize MTCNN for face detection and alignment
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)
if face_cascade.empty():
    print("Error loading Haar Cascade for face detection.")
    exit(1) 

validation_data = tf.data.Dataset.load('validation')

# Preprocessing
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path) # Loads the file
    img = tf.io.decode_jpeg(byte_img)     # Decodes it in order to deal only with image pixel values
    img = tf.image.resize(img, (105,105)) # Resizing it: to the size mentioned in the Siamese paper
    img = img / 255.0                     # Preforming the Scaling (0 --> 1)
    return img

# Verification Funciton
def verify_vectorized(model, detection_threshold, verification_threshold):
    """IMP: Its Hardcoded to just use 50 images of validation images"""
    input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
    input_img_repeated = tf.data.Dataset.from_tensors(input_img).repeat(50)
    dataset = tf.data.Dataset.zip((validation_data, input_img_repeated))
    dataset = dataset.batch(50)
    dataset = dataset.prefetch(50)

    test_input, test_val = dataset.as_numpy_iterator().next()
    results = model.predict([test_input, test_val])
    results = results.to_tensor()
    results = results.numpy()

    detection = np.sum(results > detection_threshold)
    verification = detection / len(test_input) # 50
    verified = verification > verification_threshold

    return results, verified

# Open CV realtime Verification
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120+250, 200:200+250]
    if True: # arduino.in_waiting > 0:
        # data = arduino.readline().decode('utf-8').strip()
        # print(f"Received: {data}")
        # if data == "DETECTED":
        if cv2.waitKey(10) & 0xFF == ord('v'):
            # Detect and align face using MTCNN
            aligned_face = mtcnn(frame)
            if aligned_face is None:
                frame_old = frame
                for i in range(5):
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    aligned_face = mtcnn(frame)
                    if aligned_face is not None:
                        break
                if aligned_face is None:
                    print("No face detected or alignment failed.")
                    continue

            cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
            # Verification
            results, verified = verify_vectorized(model, 0.8, 0.75) # 0.82 0.75

            # # Debugging
            # for i in range(len(results)):
                # print(f" {i}: {results[i][0][0][0]}")
            # print(f"Average: {np.mean(np.stack(results), axis=0)[0][0][0]}")

            if verified:
                print("Face recognized")
                # send_message(f"Ziad has entered!", MyPhoneNumber)
                # arduino.write(b"1\n")
                print("------------------------------------")
            else:
                print("Face not recognized")
                # send_message("WARNING!! Intruder.", MyPhoneNumber)
                # arduino.write(b"2\n")
                print("------------------------------------")
    cv2.imshow('Verification', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
