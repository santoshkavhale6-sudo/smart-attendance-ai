import os, cv2, time, datetime, threading, io, base64, json
import numpy as np
import pandas as pd
from pathlib import Path
from flask import (Flask, render_template, request, redirect, url_for,
                   session, Response, jsonify, send_file, flash)
from database.db import get_db, init_db
from recognition.face_engine import (detect_faces, train_model, load_model,
                                     recognize_face, DATASET)

app = Flask(__name__)
app.secret_key = "smart_attendance_secret_2024"
BASE = Path(__file__).resolve().parent

# ── Global state ──
camera = None
camera_lock = threading.Lock()
camera_url = "0"
model, label_map = None, None
marked_today = set()

def get_model():
    global model, label_map
    if model is None:
        model, label_map = load_model()
    return model, label_map

# ── Auth ──
@app.route("/", methods=["GET"])
def index():
    if session.get("user"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "").strip()
        db = get_db()
        row = db.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p)).fetchone()
        db.close()
        if row:
            session["user"] = u
            return redirect(url_for("dashboard"))
        flash("Invalid credentials", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ── Dashboard ──
@app.route("/dashboard")
def dashboard():
    if not session.get("user"):
        return redirect(url_for("login"))
    db = get_db()
    total_students = db.execute("SELECT COUNT(*) c FROM students").fetchone()["c"]
    today = datetime.date.today().isoformat()
    present_today = db.execute("SELECT COUNT(DISTINCT student_id) c FROM attendance WHERE date=?", (today,)).fetchone()["c"]
    total_records = db.execute("SELECT COUNT(*) c FROM attendance").fetchone()["c"]
    recent = db.execute(
        "SELECT s.name, s.roll_number, a.date, a.time, a.confidence "
        "FROM attendance a JOIN students s ON a.student_id=s.id "
        "ORDER BY a.id DESC LIMIT 20"
    ).fetchall()
    db.close()
    return render_template("dashboard.html",
                           total_students=total_students,
                           present_today=present_today,
                           absent_today=max(0, total_students - present_today),
                           total_records=total_records,
                           recent=recent)

# ── Student Registration ──
@app.route("/register", methods=["GET", "POST"])
def register():
    if not session.get("user"):
        return redirect(url_for("login"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        roll = request.form.get("roll_number", "").strip()
        dept = request.form.get("department", "").strip()
        if not name or not roll:
            flash("Name and Roll Number are required", "error")
            return redirect(url_for("register"))
        db = get_db()
        try:
            db.execute("INSERT INTO students (name, roll_number, department) VALUES (?,?,?)",
                       (name, roll, dept))
            db.commit()
        except Exception as e:
            flash(f"Error: {e}", "error")
            db.close()
            return redirect(url_for("register"))
        db.close()
        student_dir = DATASET / roll
        student_dir.mkdir(parents=True, exist_ok=True)
        flash(f"Student {name} registered! Now capture face images.", "success")
        return redirect(url_for("capture_faces", roll=roll))
    return render_template("register.html")

# ── Face Capture (persistent camera) ──
capture_cam = {"cap": None, "url": None}

@app.route("/capture/<roll>")
def capture_faces(roll):
    if not session.get("user"):
        return redirect(url_for("login"))
    return render_template("capture.html", roll=roll)

@app.route("/api/start_camera", methods=["POST"])
def api_start_camera():
    """Open camera once and keep it open for fast frame grabs."""
    cam_url = request.form.get("camera_url", "0")
    parsed = int(cam_url) if cam_url.isdigit() else cam_url
    # Release old camera if different URL
    if capture_cam["cap"] is not None:
        capture_cam["cap"].release()
        capture_cam["cap"] = None
    cap = cv2.VideoCapture(parsed)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open camera. Check URL or device."})
    # Warm up – discard first few frames
    for _ in range(5):
        cap.read()
    capture_cam["cap"] = cap
    capture_cam["url"] = cam_url
    return jsonify({"success": True, "message": "Camera opened"})

@app.route("/api/stop_camera", methods=["POST"])
def api_stop_camera():
    if capture_cam["cap"] is not None:
        capture_cam["cap"].release()
        capture_cam["cap"] = None
    return jsonify({"success": True})

capture_counter = {"count": 0}

@app.route("/api/capture_frame", methods=["POST"])
def api_capture_frame():
    roll = request.form.get("roll")
    cam_url = request.form.get("camera_url", "0")
    skip_preview = request.form.get("skip_preview", "0")
    student_dir = DATASET / roll
    student_dir.mkdir(parents=True, exist_ok=True)

    # Use counter for speed instead of scanning directory each time
    if capture_counter.get("roll") != roll:
        capture_counter["roll"] = roll
        capture_counter["count"] = len(list(student_dir.glob("*.jpg")))
    existing = capture_counter["count"]

    # If camera not started yet, start it now
    if capture_cam["cap"] is None or not capture_cam["cap"].isOpened():
        parsed = int(cam_url) if cam_url.isdigit() else cam_url
        cap = cv2.VideoCapture(parsed)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open camera", "count": existing})
        for _ in range(3):
            cap.read()
        capture_cam["cap"] = cap
        capture_cam["url"] = cam_url

    cap = capture_cam["cap"]
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Failed to read frame", "count": existing})
    faces, gray = detect_faces(frame)
    if len(faces) == 0:
        # Lightweight response — no preview image when no face
        return jsonify({"error": "No face detected", "count": existing})
    x, y, w, h = faces[0]
    face_img = gray[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (100, 100))
    new_count = existing + 1
    img_path = student_dir / f"{new_count}.jpg"
    cv2.imwrite(str(img_path), face_img)
    capture_counter["count"] = new_count

    # Only send preview image every 5th capture to save bandwidth
    b64 = ""
    if skip_preview != "1" and (new_count % 5 == 1 or new_count >= 50):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        b64 = base64.b64encode(buf).decode()

    return jsonify({"success": True, "count": new_count, "frame": b64})

@app.route("/api/preview_frame", methods=["POST"])
def api_preview_frame():
    """Read a frame from open camera and return as base64 (no saving)."""
    if capture_cam["cap"] is None or not capture_cam["cap"].isOpened():
        return jsonify({"error": "Camera not open"})
    ret, frame = capture_cam["cap"].read()
    if not ret:
        return jsonify({"error": "Failed to read frame"})
    faces, gray = detect_faces(frame)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Face detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    _, buf = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(buf).decode()
    return jsonify({"success": True, "frame": b64, "faces": len(faces)})

# ── Training ──
@app.route("/train", methods=["GET", "POST"])
def train():
    if not session.get("user"):
        return redirect(url_for("login"))
    if request.method == "POST":
        global model, label_map
        ok, msg = train_model()
        if ok:
            model, label_map = load_model()
            flash(msg, "success")
        else:
            flash(msg, "error")
        return redirect(url_for("train"))
    return render_template("train.html")

# ── Live Recognition (MJPEG stream) ──
def gen_frames(cam_url):
    global marked_today
    mdl, lmap = get_model()
    if cam_url.isdigit():
        cam_url = int(cam_url)
    cap = cv2.VideoCapture(cam_url)
    today = datetime.date.today().isoformat()
    # Cache roll_number -> student name lookups
    name_cache = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        faces, gray = detect_faces(frame)
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            roll_number, conf = "Unknown", 0
            if mdl is not None:
                roll_number, conf = recognize_face(face_roi, mdl, lmap)

            # Look up the student's actual name from DB using roll_number
            display_name = "Unknown"
            if roll_number != "Unknown" and conf > 60:
                if roll_number in name_cache:
                    display_name = name_cache[roll_number]
                else:
                    db = get_db()
                    stu = db.execute("SELECT name FROM students WHERE roll_number=?", (roll_number,)).fetchone()
                    db.close()
                    if stu:
                        # Use first name only
                        display_name = stu["name"].split()[0]
                        name_cache[roll_number] = display_name

            color = (0, 255, 0) if display_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            # Corner accents
            l = 20
            cv2.line(frame, (x, y), (x+l, y), color, 3)
            cv2.line(frame, (x, y), (x, y+l), color, 3)
            cv2.line(frame, (x+w, y), (x+w-l, y), color, 3)
            cv2.line(frame, (x+w, y), (x+w, y+l), color, 3)
            cv2.line(frame, (x, y+h), (x+l, y+h), color, 3)
            cv2.line(frame, (x, y+h), (x, y+h-l), color, 3)
            cv2.line(frame, (x+w, y+h), (x+w-l, y+h), color, 3)
            cv2.line(frame, (x+w, y+h), (x+w, y+h-l), color, 3)
            label = f"{display_name} ({conf}%)" if display_name != "Unknown" else "Unknown"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # Auto-mark attendance
            if display_name != "Unknown" and conf > 60:
                key = f"{roll_number}_{today}"
                if key not in marked_today:
                    db = get_db()
                    row = db.execute("SELECT id FROM students WHERE roll_number=?", (roll_number,)).fetchone()
                    if row:
                        already = db.execute("SELECT 1 FROM attendance WHERE student_id=? AND date=?",
                                             (row["id"], today)).fetchone()
                        if not already:
                            now = datetime.datetime.now().strftime("%H:%M:%S")
                            db.execute("INSERT INTO attendance (student_id, date, time, confidence) VALUES (?,?,?,?)",
                                       (row["id"], today, now, conf))
                            db.commit()
                            marked_today.add(key)
                    db.close()
        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    cap.release()

@app.route("/recognition")
def recognition():
    if not session.get("user"):
        return redirect(url_for("login"))
    return render_template("recognition.html")

@app.route("/video_feed")
def video_feed():
    url = request.args.get("url", "0")
    return Response(gen_frames(url), mimetype="multipart/x-mixed-replace; boundary=frame")

# ── Attendance History ──
@app.route("/attendance")
def attendance_history():
    if not session.get("user"):
        return redirect(url_for("login"))
    date_filter = request.args.get("date", "")
    db = get_db()
    if date_filter:
        rows = db.execute(
            "SELECT s.name, s.roll_number, s.department, a.date, a.time, a.confidence "
            "FROM attendance a JOIN students s ON a.student_id=s.id WHERE a.date=? ORDER BY a.time DESC",
            (date_filter,)).fetchall()
    else:
        rows = db.execute(
            "SELECT s.name, s.roll_number, s.department, a.date, a.time, a.confidence "
            "FROM attendance a JOIN students s ON a.student_id=s.id ORDER BY a.id DESC LIMIT 100"
        ).fetchall()
    db.close()
    return render_template("attendance.html", records=rows, date_filter=date_filter)

# ── Analytics ──
@app.route("/analytics")
def analytics():
    if not session.get("user"):
        return redirect(url_for("login"))
    db = get_db()
    # Daily counts for last 7 days
    daily = db.execute(
        "SELECT date, COUNT(DISTINCT student_id) c FROM attendance "
        "GROUP BY date ORDER BY date DESC LIMIT 7"
    ).fetchall()
    daily_labels = [r["date"] for r in reversed(daily)]
    daily_values = [r["c"] for r in reversed(daily)]
    # Per-student attendance count
    students = db.execute(
        "SELECT s.name, COUNT(a.id) c FROM students s LEFT JOIN attendance a ON s.id=a.student_id "
        "GROUP BY s.id ORDER BY c DESC LIMIT 10"
    ).fetchall()
    stu_labels = [r["name"] for r in students]
    stu_values = [r["c"] for r in students]
    total = db.execute("SELECT COUNT(*) c FROM students").fetchone()["c"]
    today = datetime.date.today().isoformat()
    present = db.execute("SELECT COUNT(DISTINCT student_id) c FROM attendance WHERE date=?", (today,)).fetchone()["c"]
    db.close()
    return render_template("analytics.html",
                           daily_labels=json.dumps(daily_labels),
                           daily_values=json.dumps(daily_values),
                           stu_labels=json.dumps(stu_labels),
                           stu_values=json.dumps(stu_values),
                           present=present, absent=max(0, total-present))

# ── Export ──
@app.route("/export")
def export_page():
    if not session.get("user"):
        return redirect(url_for("login"))
    return render_template("export.html")

@app.route("/export/csv")
def export_csv():
    db = get_db()
    df = pd.read_sql_query(
        "SELECT s.name, s.roll_number, s.department, a.date, a.time, a.confidence "
        "FROM attendance a JOIN students s ON a.student_id=s.id ORDER BY a.date DESC", db)
    db.close()
    out = BASE / "exports" / "attendance.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return send_file(out, as_attachment=True)

@app.route("/export/excel")
def export_excel():
    db = get_db()
    df = pd.read_sql_query(
        "SELECT s.name, s.roll_number, s.department, a.date, a.time, a.confidence "
        "FROM attendance a JOIN students s ON a.student_id=s.id ORDER BY a.date DESC", db)
    db.close()
    out = BASE / "exports" / "attendance.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out, index=False)
    return send_file(out, as_attachment=True)

# ── Students List ──
@app.route("/students")
def students_list():
    if not session.get("user"):
        return redirect(url_for("login"))
    db = get_db()
    rows = db.execute("SELECT * FROM students ORDER BY id DESC").fetchall()
    db.close()
    # Count face images per student
    students_data = []
    for s in rows:
        face_dir = DATASET / s["roll_number"]
        face_count = len(list(face_dir.glob("*.jpg"))) if face_dir.exists() else 0
        students_data.append({"student": s, "face_count": face_count})
    return render_template("students.html", students=students_data)

@app.route("/students/delete/<int:sid>")
def delete_student(sid):
    if not session.get("user"):
        return redirect(url_for("login"))
    db = get_db()
    db.execute("DELETE FROM attendance WHERE student_id=?", (sid,))
    db.execute("DELETE FROM students WHERE id=?", (sid,))
    db.commit()
    db.close()
    flash("Student deleted", "success")
    return redirect(url_for("students_list"))

# ── Settings ──
@app.route("/settings")
def settings():
    if not session.get("user"):
        return redirect(url_for("login"))
    return render_template("settings.html")

# ── About ──
@app.route("/about")
def about():
    if not session.get("user"):
        return redirect(url_for("login"))
    return render_template("about.html")

if __name__ == "__main__":
    os.makedirs(BASE / "dataset", exist_ok=True)
    os.makedirs(BASE / "models", exist_ok=True)
    os.makedirs(BASE / "exports", exist_ok=True)
    init_db()
    print("\n  Smart Attendance Application")
    print("  Open in browser: http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
