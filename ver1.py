import serial
from deepface import DeepFace
import cv2
from utils.send import send_message
import os 
import shutil


# Initialize Serial Communication
arduino = serial.Serial(port='COM5', baudrate=9600, timeout=1)

MyPhoneNumber = os.getenv('MY_PHONE_NUMBER')
MaxNTries = 4

# Camera Setup
camera = cv2.VideoCapture(0)

# Registered Face Path
# registered_face_path = "D:\\ElectronicsProject\\RegisterdFaces\\Ziad.jpg"
registered_faces = "D:\\ElectronicsProject\\RegisterdFaces"

if not os.path.exists(registered_faces):
    print(f"Error: Registered face file not found at {registered_faces}")
    exit(1)

def capture_image():
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

i = 0
while True:
    try:
        # Read serial data if available
        if arduino.in_waiting > 0:
            data = arduino.readline().decode('utf-8').strip()
            print(f"Received: {data}")
            
            if data == "DETECTED":
                face_path = capture_image()
                if face_path:
                    # Send Face to Twilio
                    unknown = 0
                    recognized = False
                    for registered_identity in os.listdir(registered_faces):
                        registered_identity_full = registered_faces + "\\" + registered_identity
                        print(f"Verifying: {registered_identity_full}")
                        for registered_face_path in os.listdir(registered_identity_full):
                            registered_face_path = os.path.join(registered_identity_full, registered_face_path)
                            print(f"Verifying: {registered_face_path}")
                            try:
                                result = DeepFace.verify(
                                    img1_path=registered_face_path,
                                    img2_path=face_path,
                                    model_name="Facenet512"
                                )
                                if result.get('verified', False):
                                    print(f"Face recognized: {registered_identity}")
                                    send_message(f"{registered_identity} has entered!", MyPhoneNumber)
                                    arduino.write(b"1\n")
                                    recognized = True
                                    break
                            except Exception as df_error:
                                print(f"Verification error: {df_error}")
                                continue

                    if not recognized:
                        unknown += 1
                        arduino.write(b"2\n")  # Signal unrecognized face
                        if unknown >= MaxNTries:
                            send_message("WARNING!! Intruder.", MyPhoneNumber)

                        rigester = input("Do you want to register the face? (y/n): ")
                        if rigester == "y":
                            name = input("Enter name: ")
                            os.mkdir(os.path.join(registered_faces, name))
                            shutil.copy("D:\\ElectronicsProject\\CachedFaces\\current_face.jpg", os.path.join(f"D:\\ElectronicsProject\\RegisterdFaces\\{name}", name + ".jpg"))
                            print(f"Face registered: {name}")
                        else:
                            print("Face not registered.")
                else:
                    print("Failed to capture image. Skipping detection.")
                    arduino.write(b"0\n")  # Signal image failure

    except Exception as main_error:
        print(f"Error: {main_error}")
    finally:
        # Add a short delay to avoid high CPU usage
        continue
