import cv2
import csv
import os
import qrcode

STUDENTS_FILE = "students.csv"
FACE_DIR = "faces"
QR_DIR = "qrcodes"

os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(QR_DIR, exist_ok=True)

if not os.path.exists(STUDENTS_FILE):
    with open(STUDENTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Name", "Department"])

def capture_face(student_id):
    cam = cv2.VideoCapture(0)
    count = 0

    print("📸 Press SPACE to capture face, Q to quit")

    while True:
        ret, frame = cam.read()
        cv2.imshow("Face Capture", frame)

        key = cv2.waitKey(1)

        if key == ord(' '):
            path = f"{FACE_DIR}/{student_id}.jpg"
            cv2.imwrite(path, frame)
            print("Face Saved")
            break

        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def generate_qr(student_id):
    img = qrcode.make(student_id)
    img.save(f"{QR_DIR}/{student_id}.png")

def register():
    sid = input("ID: ")
    name = input("Name: ")
    dept = input("Department: ")

    with open(STUDENTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([sid, name, dept])

    generate_qr(sid)
    capture_face(sid)

    print("✅ Student Registered Successfully!")

register()