import cv2
import csv
import os
import qrcode
import numpy as np
import face_recognition

# ---------------- FOLDERS ----------------
STUDENTS_FILE = "students.csv"
FACE_DIR = "faces"
QR_DIR = "qrcodes"
ENC_DIR = "encodings"

os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(QR_DIR, exist_ok=True)
os.makedirs(ENC_DIR, exist_ok=True)

# ---------------- INIT CSV ----------------
if not os.path.exists(STUDENTS_FILE):
    with open(STUDENTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Name", "Department"])

# ---------------- CHECK DUPLICATE ----------------
def is_duplicate(sid):
    with open(STUDENTS_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[0] == sid:
                return True
    return False

# ---------------- FACE CAPTURE ----------------
def capture_face(student_id):

    cam = cv2.VideoCapture(0)

    print("📸 Press SPACE to capture face, Q to quit")

    while True:
        ret, frame = cam.read()
        cv2.imshow("Face Capture", frame)

        key = cv2.waitKey(1)

        if key == ord(' '):

            img_path = f"{FACE_DIR}/{student_id}.jpg"
            cv2.imwrite(img_path, frame)

            # 🔥 CREATE FACE ENCODING (IMPORTANT FIX)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            enc = face_recognition.face_encodings(rgb)

            if len(enc) > 0:
                np.save(f"{ENC_DIR}/{student_id}.npy", enc[0])
                print("✅ Face & Encoding Saved")
            else:
                print("❌ No face detected, try again")

            break

        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# ---------------- QR GENERATE ----------------
def generate_qr(student_id):
    img = qrcode.make(student_id)
    img.save(f"{QR_DIR}/{student_id}.png")

# ---------------- REGISTER STUDENT ----------------
def register():

    sid = input("ID: ").strip()
    name = input("Name: ").strip()
    dept = input("Department: ").strip()

    # 🔥 DUPLICATE CHECK FIX
    if is_duplicate(sid):
        print("❌ This ID already exists!")
        return

    # save student
    with open(STUDENTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([sid, name, dept])

    generate_qr(sid)
    capture_face(sid)

    print("\n✅ Student Registered Successfully!")

# ---------------- RUN ----------------
register()