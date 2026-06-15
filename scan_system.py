import cv2
import face_recognition
import numpy as np
import csv
import os
from datetime import datetime

ENC_DIR = "encodings"
ATT_FILE = "attendance.csv"
STUDENTS_FILE = "students.csv"

# ---------------- INIT ----------------
if not os.path.exists(ATT_FILE):
    with open(ATT_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["ID", "Date", "Time", "Status"])


# ---------------- STUDENT INFO ----------------
def get_student_info(sid):
    with open(STUDENTS_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[0] == sid:
                return row[1], row[2]
    return None, None


# ---------------- LOAD ENCODINGS ----------------
def load_encodings():
    enc = {}

    if not os.path.exists(ENC_DIR):
        print("❌ Encodings folder missing!")
        return enc

    for file in os.listdir(ENC_DIR):
        if file.endswith(".npy"):
            sid = file.split(".")[0]
            try:
                enc[sid] = np.load(f"{ENC_DIR}/{file}")
            except:
                print(f"⚠ Error loading: {sid}")

    return enc


encodings = load_encodings()


# ---------------- ATTENDANCE ----------------
def mark_attendance(sid):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M")

    with open(ATT_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row and row[0] == sid and row[1] == date:
                print("⚠ Already marked today")
                return

    with open(ATT_FILE, "a", newline="") as f:
        csv.writer(f).writerow([sid, date, time, "Present"])


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
            print(f"✔ QR Detected: {data}")
            return data

        cv2.imshow("QR Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    return None


# ---------------- FACE VERIFY (FIXED FULL GOD MODE) ----------------
def face_verify(sid):
    if sid not in encodings:
        print("❌ No face data found for this ID")
        return False

    known_encoding = encodings[sid]
    cam = cv2.VideoCapture(0)

    print("📸 Face verification started... Look at camera")

    match_count = 0
    wrong_count = 0

    while True:
        ret, frame = cam.read()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_recognition.face_locations(rgb)

        # ---------------- NO FACE ----------------
        if len(faces) == 0:
            cv2.putText(frame, "NO FACE DETECTED", (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Face Verify", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        encs = face_recognition.face_encodings(rgb, faces)

        for face in encs:
            result = face_recognition.compare_faces(
                [known_encoding], face, tolerance=0.45
            )

            distance = face_recognition.face_distance(
                [known_encoding], face
            )[0]

            # ---------------- MATCH ----------------
            if result[0] and distance < 0.45:
                match_count += 1
                wrong_count = 0

                cv2.putText(frame, "FACE MATCHED", (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                print(f"✔ Match confidence: {1-distance:.2f}")

                if match_count >= 3:
                    cam.release()
                    cv2.destroyAllWindows()
                    print("✔ FACE VERIFIED SUCCESSFULLY")
                    return True

            # ---------------- WRONG FACE ----------------
            else:
                wrong_count += 1
                match_count = 0

                cv2.putText(frame, "FACE NOT MATCHED", (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                print("❌ Wrong Face Detected")

                # AUTO REJECT
                if wrong_count >= 10:
                    cam.release()
                    cv2.destroyAllWindows()
                    print("❌ FACE REJECTED (Too many wrong attempts)")
                    return False

        cv2.imshow("Face Verify", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    return False


# ---------------- STATS ----------------
def get_stats(sid):
    present = 0
    days = set()

    with open(ATT_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if row[0] == sid:
                days.add(row[1])
                if row[3] == "Present":
                    present += 1

    total = len(days)
    absent = total - present if total > 0 else 0
    percent = (present / total * 100) if total > 0 else 0

    return present, absent, percent


# ---------------- MAIN SYSTEM ----------------
print("🚀 GOD MODE SYSTEM STARTED")

sid = scan_qr()

if sid:
    name, dept = get_student_info(sid)

    print(f"✔ Student: {name} ({dept})")

    if face_verify(sid):
        mark_attendance(sid)

        present, absent, percent = get_stats(sid)

        print("\n==============================")
        print("🎓 ATTENDANCE SUCCESS")
        print(f"ID: {sid}")
        print(f"Name: {name}")
        print(f"Dept: {dept}")
        print(f"Present: {present}")
        print(f"Absent: {absent}")
        print(f"Attendance: {percent:.1f}%")
        print("==============================\n")

    else:
        print("❌ Face verification FAILED")

else:
    print("❌ QR scan FAILED")