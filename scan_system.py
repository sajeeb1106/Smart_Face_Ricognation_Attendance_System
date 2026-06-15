import cv2
import face_recognition
import numpy as np
import csv
import os
import threading
import time
from datetime import datetime, timedelta

# ---------------- FOLDERS ----------------
ENC_DIR = "encodings"
ATT_DIR = "attendance"
STUDENTS_FILE = "students.csv"

os.makedirs(ATT_DIR, exist_ok=True)

# ---------------- GLOBAL ----------------
encodings = {}
pending_students = set()

session_active = False
session_subject = ""
session_start_time = None
session_duration = timedelta(hours=2)

# ---------------- LOAD STUDENTS ----------------
def load_students():
    students = []

    with open(STUDENTS_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            students.append({"id": row[0], "name": row[1], "dept": row[2]})

    return students


STUDENTS = load_students()


# ---------------- LOAD ENCODINGS ----------------
def load_encodings():
    enc = {}

    for file in os.listdir(ENC_DIR):
        if file.endswith(".npy"):
            sid = file.split(".")[0]
            enc[sid] = np.load(f"{ENC_DIR}/{file}")

    return enc


encodings = load_encodings()


# ---------------- SUBJECT FILE ----------------
def get_subject_file(subject):
    return os.path.join(ATT_DIR, f"{subject}.csv")


def init_subject_file(subject):
    file = get_subject_file(subject)

    if not os.path.exists(file):
        with open(file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Name", "Dept"])

            for s in STUDENTS:
                writer.writerow([s["id"], s["name"], s["dept"]])

    return file


# ---------------- ADD SESSION COLUMN ----------------
def add_session_column(subject):
    file = get_subject_file(subject)

    session_name = datetime.now().strftime("%Y-%m-%d %H:%M")

    rows = []

    with open(file, "r") as f:
        reader = list(csv.reader(f))
        header = reader[0]
        data = reader[1:]

    header.append(session_name)

    for row in data:
        row.append("Absent")   # default

    rows.append(header)
    rows.extend(data)

    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return session_name


# ---------------- MARK PRESENT ----------------
def mark_present(subject, sid):
    file = get_subject_file(subject)

    with open(file, "r") as f:
        rows = list(csv.reader(f))

    header = rows[0]
    data = rows[1:]

    if sid not in pending_students:
        return

    col_index = len(header) - 1  # latest session column

    for row in data:
        if row[0] == sid:
            row[col_index] = "Present"

    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


# ---------------- AUTO ABSENT AFTER 2 HOURS ----------------
def session_closer(subject, session_start):
    global session_active, pending_students

    while session_active:
        if datetime.now() - session_start >= session_duration:

            file = get_subject_file(subject)

            with open(file, "r") as f:
                rows = list(csv.reader(f))

            header = rows[0]
            data = rows[1:]

            col_index = len(header) - 1

            for row in data:
                if row[0] in pending_students:
                    row[col_index] = "Absent"

            with open(file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data)

            print("⛔ SESSION CLOSED (AUTO ABSENT DONE)")
            session_active = False
            pending_students.clear()
            break

        time.sleep(30)


# ---------------- QR SCAN ----------------
def scan_qr():
    cam = cv2.VideoCapture(0)
    detector = cv2.QRCodeDetector()

    print("📷 Scan QR Code...")

    while True:
        ret, frame = cam.read()
        data, bbox, _ = detector.detectAndDecode(frame)

        if data:
            cam.release()
            cv2.destroyAllWindows()
            return data

        cv2.imshow("QR Scanner", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    return None


# ---------------- FACE VERIFY ----------------
def face_verify(sid):

    if sid not in encodings:
        return False

    cam = cv2.VideoCapture(0)
    known = encodings[sid]

    match = 0

    while True:
        ret, frame = cam.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, faces)

        for face in encs:
            res = face_recognition.compare_faces([known], face, tolerance=0.45)

            if res[0]:
                match += 1
            else:
                match = 0

            if match >= 3:
                cam.release()
                cv2.destroyAllWindows()
                return True

        cv2.imshow("Face Verify", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    return False


# ---------------- MAIN ----------------
print("🚀 SYSTEM STARTED")

subject = input("📘 Enter Subject Name: ")

init_subject_file(subject)

session_start_time = datetime.now()
session_active = True

session_name = add_session_column(subject)

print(f"📌 Attendance Started for: {subject}")
print(f"⏳ Session: {session_name} (2 hours)")

threading.Thread(
    target=session_closer,
    args=(subject, session_start_time),
    daemon=True
).start()


while session_active:

    sid = scan_qr()

    if not sid:
        continue

    pending_students.add(sid)

    if face_verify(sid):
        mark_present(subject, sid)
        print(f"✅ PRESENT: {sid}")
    else:
        print(f"❌ FACE FAILED: {sid}")