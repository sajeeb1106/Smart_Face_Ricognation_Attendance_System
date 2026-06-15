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
pending_attendance = {}   # ⬅️ QR scan pending list


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


# ---------------- ATTENDANCE MARK ----------------
def mark_attendance(sid, status="Present", custom_time=None):

    now = custom_time if custom_time else datetime.now()

    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    daily_file = os.path.join(ATT_DIR, f"attendance_{date}.csv")

    # create file
    if not os.path.exists(daily_file):
        with open(daily_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Date", "Time", "Status"])

    # duplicate check
    with open(daily_file, "r") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if row and row[0] == sid:
                print("⚠ Already marked today")
                return False

    # write
    with open(daily_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([sid, date, time_str, status])

    print(f"✅ Attendance Saved: {status}")
    return True


# ---------------- AUTO ABSENT SYSTEM ----------------
def auto_absent_checker():

    while True:
        now = datetime.now()

        to_remove = []

        for sid, scan_time in pending_attendance.items():

            if now - scan_time >= timedelta(hours=3):

                daily_file = os.path.join(
                    ATT_DIR,
                    f"attendance_{scan_time.strftime('%Y-%m-%d')}.csv"
                )

                already = False

                if os.path.exists(daily_file):
                    with open(daily_file, "r") as f:
                        reader = csv.reader(f)
                        next(reader)

                        for row in reader:
                            if row and row[0] == sid:
                                already = True
                                break

                if not already:
                    mark_attendance(
                        sid,
                        status="Absent",
                        custom_time=scan_time
                    )
                    print(f"⛔ AUTO ABSENT: {sid}")

                to_remove.append(sid)

        for sid in to_remove:
            pending_attendance.pop(sid, None)

        time.sleep(60)  # check every 1 min


# ---------------- QR SCANNER ----------------
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


# ---------------- FACE VERIFY ----------------
def face_verify(sid):

    if sid not in encodings:
        print("❌ No face data found")
        return False

    known_encoding = encodings[sid]
    cam = cv2.VideoCapture(0)

    print("📸 Face verification started...")

    match_count = 0
    wrong_count = 0

    while True:
        ret, frame = cam.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_recognition.face_locations(rgb)

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

            if result[0] and distance < 0.45:
                match_count += 1
                wrong_count = 0

                cv2.putText(frame, "MATCHED", (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if match_count >= 3:
                    cam.release()
                    cv2.destroyAllWindows()
                    print("✔ FACE VERIFIED")
                    return True

            else:
                wrong_count += 1
                match_count = 0

                cv2.putText(frame, "NOT MATCHED", (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if wrong_count >= 10:
                    cam.release()
                    cv2.destroyAllWindows()
                    print("❌ FACE REJECTED")
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
    total_days = 0

    if not os.path.exists(ATT_DIR):
        return 0, 0, 0

    for file in os.listdir(ATT_DIR):

        if file.endswith(".csv"):
            total_days += 1

            with open(os.path.join(ATT_DIR, file), "r") as f:
                reader = csv.reader(f)
                next(reader)

                for row in reader:
                    if row[0] == sid:
                        if row[3] == "Present":
                            present += 1
                        break

    absent = total_days - present
    percent = (present / total_days * 100) if total_days > 0 else 0

    return present, absent, percent


# ---------------- MAIN ----------------
print("🚀 SYSTEM STARTED")

# START AUTO THREAD
threading.Thread(target=auto_absent_checker, daemon=True).start()

sid = scan_qr()

if sid:

    pending_attendance[sid] = datetime.now()   # ⬅️ IMPORTANT

    name, dept = get_student_info(sid)

    print(f"✔ Student: {name} ({dept})")

    if face_verify(sid):

        mark_attendance(sid, status="Present")

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

        # remove from pending (important)
        pending_attendance.pop(sid, None)

    else:
        print("❌ Face verification FAILED")

else:
    print("❌ QR scan FAILED")