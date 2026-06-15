import face_recognition
import os
import numpy as np

FACE_DIR = "faces"
ENC_DIR = "encodings"
os.makedirs(ENC_DIR, exist_ok=True)

for file in os.listdir(FACE_DIR):
    if file.endswith(".jpg"):
        img_path = f"{FACE_DIR}/{file}"
        image = face_recognition.load_image_file(img_path)

        encoding = face_recognition.face_encodings(image)[0]

        sid = file.split(".")[0]

        np.save(f"{ENC_DIR}/{sid}.npy", encoding)

        print(f"Encoded: {sid}")