
import os
import sqlite3
import pickle
import csv
import io
import math
from datetime import datetime, timedelta
from functools import wraps

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, make_response
)

from flask_wtf.csrf import CSRFProtect

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


# ---------------- APP CONFIGURATION ----------------

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "local-development-secret-key-change-before-deployment"
)

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=(
        os.environ.get(
            "SESSION_COOKIE_SECURE", "false"
        ).lower() == "true"
    ),
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),
    WTF_CSRF_TIME_LIMIT=3600
)

csrf = CSRFProtect(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "students.db")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "student_performance_model.pkl"
)

PERFORMANCE_CATEGORIES = [
    "High",
    "Average",
    "Low",
    "Excellent",
    "Good",
    "Poor"
]


# ---------------- DATABASE CONNECTION ----------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'student'
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attendance REAL,
            study_hours REAL,
            previous_marks REAL,
            assignments REAL,
            performance TEXT,
            created_at TEXT,
            student_id INTEGER
        )
    """)

    cursor.execute("PRAGMA table_info(predictions)")

    columns = [
        row["name"] for row in cursor.fetchall()
    ]

    migrations = {
        "student_id": "INTEGER",
        "created_at": "TEXT",
        "performance": "TEXT",
        "attendance": "REAL",
        "study_hours": "REAL",
        "previous_marks": "REAL",
        "assignments": "REAL"
    }

    for column, data_type in migrations.items():

        if column not in columns:

            cursor.execute(
                f"ALTER TABLE predictions ADD COLUMN {column} {data_type}"
            )

    conn.commit()
    conn.close()


# ---------------- PERFORMANCE HELPERS ----------------

def normalize_performance(value):

    label = str(value or "").strip().lower()

    labels = {
        "high": "High",
        "average": "Average",
        "low": "Low",
        "excellent": "Excellent",
        "good": "Good",
        "poor": "Poor"
    }

    return labels.get(label, label.title())


def get_performance_counts(records):

    counts = {
        label: 0 for label in PERFORMANCE_CATEGORIES
    }

    for record in records:

        label = normalize_performance(
            record["performance"]
        )

        if label in counts:
            counts[label] += 1

    return counts


def get_percentages(counts, total):

    return {
        label: round((count / total) * 100) if total else 0
        for label, count in counts.items()
    }


# ---------------- PERSONALIZED IMPROVEMENT SUGGESTIONS ----------------

def generate_suggestions(attendance, study_hours, previous_marks, assignments):

    suggestions = []

    if attendance < 75:
        suggestions.append(
            "Try to attend classes more regularly. Consistent attendance can "
            "help you understand concepts better. Also check your institute's "
            "attendance requirements."
        )

    if study_hours < 2:
        suggestions.append(
            "Build a realistic daily study routine. Start with short, focused "
            "study sessions and gradually improve consistency."
        )

    if previous_marks < 60:
        suggestions.append(
            "Revise your basic concepts and practice questions regularly. "
            "Ask your teacher for help with topics you find difficult."
        )

    if assignments < 70:
        suggestions.append(
            "Try to complete assignments on time. Break larger tasks into "
            "smaller parts to make them easier to manage."
        )

    if not suggestions:
        suggestions.append(
            "Keep maintaining your current study habits. Continue regular "
            "revision, attend classes consistently, and practice questions "
            "to strengthen your understanding."
        )

    return suggestions


# ---------------- CREATE ADMIN ----------------

def create_admin():

    admin_email = os.environ.get("ADMIN_EMAIL")
    admin_password = os.environ.get("ADMIN_PASSWORD")

    admin_name = os.environ.get(
        "ADMIN_NAME", "Administrator"
    )

    if not admin_email or not admin_password:
        return

    conn = get_db()
    cursor = conn.cursor()

    try:

        cursor.execute(
            "SELECT id FROM students WHERE email = ?",
            (admin_email.strip().lower(),)
        )

        existing = cursor.fetchone()

        if existing:

            cursor.execute("""
                UPDATE students
                SET role = 'admin'
                WHERE email = ?
            """, (admin_email.strip().lower(),))

        else:

            cursor.execute("""
                INSERT INTO students
                (name, email, password, role)
                VALUES (?, ?, ?, 'admin')
            """, (
                admin_name,
                admin_email.strip().lower(),
                generate_password_hash(admin_password)
            ))

        conn.commit()

    finally:
        conn.close()


# ---------------- LOGIN DECORATORS ----------------

def login_required(f):

    @wraps(f)
    def decorated_function(*args, **kwargs):

        if "student_id" not in session:

            flash(
                "Please login first.",
                "warning"
            )

            return redirect(url_for("login"))

        return f(*args, **kwargs)

    return decorated_function


def admin_required(f):

    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):

        if session.get("role") != "admin":

            flash(
                "Admin access required.",
                "danger"
            )

            return redirect(url_for("dashboard"))

        return f(*args, **kwargs)

    return decorated_function


# ---------------- KNN MODEL ----------------

def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    try:

        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)

    except Exception as error:

        print("Model loading error:", error)
        return None


def knn_predict(input_data):

    model = load_model()

    if not isinstance(model, dict):

        raise ValueError(
            "KNN model file not found or invalid."
        )

    X = model.get("X")
    y = model.get("y")
    k = model.get("k", 3)

    if X is None or y is None or len(X) == 0:

        raise ValueError(
            "Invalid KNN model data."
        )

    if len(X) != len(y):

        raise ValueError(
            "KNN model data is inconsistent."
        )

    if len(input_data) != len(X[0]):

        raise ValueError(
            "Invalid prediction input features."
        )

    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 3

    if k < 1:
        k = 3

    distances = []

    for i, row in enumerate(X):

        if len(row) != len(input_data):

            raise ValueError(
                "Invalid feature count in KNN model."
            )

        distance = math.sqrt(
            sum(
                (
                    float(input_data[j]) -
                    float(row[j])
                ) ** 2

                for j in range(len(input_data))
            )
        )

        distances.append((distance, y[i]))

    distances.sort(key=lambda item: item[0])

    nearest = distances[:min(k, len(distances))]

    votes = {}

    for distance, label in nearest:

        votes[label] = votes.get(label, 0) + 1

    return max(
        votes,
        key=votes.get
    )


# ---------------- HOME ----------------

@app.route("/")
def index():

    if "student_id" in session:

        if session.get("role") == "admin":
            return redirect(url_for("admin"))

        return redirect(url_for("dashboard"))

    return redirect(url_for("login"))


# ---------------- REGISTER ----------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get(
            "name", ""
        ).strip()

        email = request.form.get(
            "email", ""
        ).strip().lower()

        password = request.form.get(
            "password", ""
        )

        if not name or not email or not password:

            flash(
                "Please fill all fields.",
                "danger"
            )

            return redirect(url_for("register"))

        if len(password) < 8:

            flash(
                "Password must be at least 8 characters.",
                "danger"
            )

            return redirect(url_for("register"))

        conn = get_db()
        cursor = conn.cursor()

        try:

            cursor.execute("""
                INSERT INTO students
                (name, email, password, role)
                VALUES (?, ?, ?, 'student')
            """, (
                name,
                email,
                generate_password_hash(password)
            ))

            conn.commit()

            flash(
                "Registration successful. Please login.",
                "success"
            )

            return redirect(url_for("login"))

        except sqlite3.IntegrityError:

            flash(
                "Email already registered.",
                "danger"
            )

            return redirect(url_for("register"))

        finally:
            conn.close()

    return render_template("register.html")


# ---------------- LOGIN ----------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get(
            "email", ""
        ).strip().lower()

        password = request.form.get(
            "password", ""
        )

        conn = get_db()
        cursor = conn.cursor()

        try:

            cursor.execute("""
                SELECT *
                FROM students
                WHERE email = ?
            """, (email,))

            student = cursor.fetchone()

        finally:
            conn.close()

        if student and check_password_hash(
            student["password"], password
        ):

            session.clear()
            session.permanent = True

            session["student_id"] = student["id"]
            session["name"] = student["name"]
            session["email"] = student["email"]
            session["role"] = student["role"]

            flash(
                "Login successful.",
                "success"
            )

            if student["role"] == "admin":
                return redirect(url_for("admin"))

            return redirect(url_for("dashboard"))

        flash(
            "Invalid email or password.",
            "danger"
        )

    return render_template("login.html")

# ---------------- DASHBOARD ----------------

@app.route("/dashboard")
@login_required
def dashboard():

    if session.get("role") == "admin":
        return redirect(url_for("admin"))

    conn = get_db()
    cursor = conn.cursor()

    try:

        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM predictions
            WHERE student_id = ?
        """, (session["student_id"],))

        total_predictions = cursor.fetchone()["total"]

        # Latest prediction
        cursor.execute("""
            SELECT *
            FROM predictions
            WHERE student_id = ?
            ORDER BY id DESC
            LIMIT 1
        """, (session["student_id"],))

        latest = cursor.fetchone()

        # Fetch two latest predictions for tracker
        cursor.execute("""
            SELECT *
            FROM predictions
            WHERE student_id = ?
            ORDER BY id DESC
            LIMIT 2
        """, (session["student_id"],))

        comparison_records = cursor.fetchall()

        # Performance category counts
        cursor.execute("""
            SELECT performance
            FROM predictions
            WHERE student_id = ?
        """, (session["student_id"],))

        records = cursor.fetchall()

    finally:
        conn.close()

    performance_counts = get_performance_counts(records)

    tracker = None

    if len(comparison_records) >= 2:

        current = comparison_records[0]
        previous = comparison_records[1]

        tracker = {
            "marks_change": round(
                float(current["previous_marks"] or 0) -
                float(previous["previous_marks"] or 0), 2
            ),

            "attendance_change": round(
                float(current["attendance"] or 0) -
                float(previous["attendance"] or 0), 2
            ),

            "study_change": round(
                float(current["study_hours"] or 0) -
                float(previous["study_hours"] or 0), 2
            ),

            "assignments_change": round(
                float(current["assignments"] or 0) -
                float(previous["assignments"] or 0), 2
            ),

            "previous_date": previous["created_at"],
            "current_date": current["created_at"]
        }

    return render_template(
        "dashboard.html",
        name=session.get("name", "Student"),
        total_predictions=total_predictions,
        latest=latest,
        latest_prediction=latest,
        performance_counts=performance_counts,
        tracker=tracker
    )


# ---------------- PREDICTION ----------------

@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():

    if session.get("role") == "admin":
        return redirect(url_for("admin"))

    if request.method == "POST":

        try:

            attendance = float(
                request.form.get("attendance", "")
            )

            study_hours = float(
                request.form.get("study_hours", "")
            )

            previous_marks = float(
                request.form.get("previous_marks", "")
            )

            assignments = float(
                request.form.get("assignments", "")
            )

            values = [
                attendance,
                study_hours,
                previous_marks,
                assignments
            ]

            if not all(math.isfinite(v) for v in values):
                raise ValueError(
                    "Please enter valid numeric values."
                )

            if not 0 <= attendance <= 100:
                raise ValueError(
                    "Attendance must be between 0 and 100."
                )

            if not 0 <= study_hours <= 24:
                raise ValueError(
                    "Study hours must be between 0 and 24."
                )

            if not 0 <= previous_marks <= 100:
                raise ValueError(
                    "Previous marks must be between 0 and 100."
                )

            if not 0 <= assignments <= 100:
                raise ValueError(
                    "Assignments must be between 0 and 100."
                )

            # Normalize KNN input
            input_data = [
                attendance / 100,
                study_hours / 24,
                previous_marks / 100,
                assignments / 100
            ]

            # Generate KNN prediction
            result = knn_predict(input_data)

            # Generate personalized suggestions
            suggestions = generate_suggestions(
                attendance,
                study_hours,
                previous_marks,
                assignments
            )

            # Save prediction in database
            conn = get_db()
            cursor = conn.cursor()

            try:

                cursor.execute("""
                    INSERT INTO predictions (
                        attendance,
                        study_hours,
                        previous_marks,
                        assignments,
                        performance,
                        created_at,
                        student_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    attendance,
                    study_hours,
                    previous_marks,
                    assignments,
                    str(result),
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    session["student_id"]
                ))

                conn.commit()

            finally:
                conn.close()

            return render_template(
                "prediction.html",
                prediction=result,
                result=result,
                suggestions=suggestions
            )

        except ValueError as error:

            flash(
                str(error),
                "danger"
            )

        except Exception as error:

            print("Prediction error:", error)

            flash(
                "Prediction could not be generated.",
                "danger"
            )

    return render_template("prediction.html")


# ---------------- PREDICTION HISTORY ----------------

@app.route("/history")
@login_required
def history():

    if session.get("role") == "admin":
        return redirect(url_for("admin"))

    conn = get_db()
    cursor = conn.cursor()

    try:

        cursor.execute("""
            SELECT *
            FROM predictions
            WHERE student_id = ?
            ORDER BY id DESC
        """, (session["student_id"],))

        records = cursor.fetchall()

    finally:
        conn.close()

    return render_template(
        "history.html",
        records=records,
        predictions=records
    )


# ---------------- ANALYTICS ----------------

@app.route("/analytics")
@login_required
def analytics():

    if session.get("role") == "admin":
        return redirect(url_for("admin"))

    conn = get_db()
    cursor = conn.cursor()

    try:

        cursor.execute("""
            SELECT performance
            FROM predictions
            WHERE student_id = ?
            ORDER BY id ASC
        """, (session["student_id"],))

        records = cursor.fetchall()

    finally:
        conn.close()

    total_predictions = len(records)

    performance_counts = get_performance_counts(records)

    percentages = get_percentages(
        performance_counts,
        total_predictions
    )

    latest = (
        normalize_performance(
            records[-1]["performance"]
        )
        if records else "No Data"
    )

    return render_template(
        "analytics.html",
        total_predictions=total_predictions,
        performance_counts=performance_counts,
        percentages=percentages,
        latest=latest,
        status=(
            "Active"
            if total_predictions > 0
            else "No Data"
        )
    )


# ---------------- DOWNLOAD STUDENT REPORT ----------------

@app.route("/download_report")
@login_required
def download_report():

    conn = get_db()
    cursor = conn.cursor()

    try:

        cursor.execute("""
            SELECT
                attendance,
                study_hours,
                previous_marks,
                assignments,
                performance,
                created_at
            FROM predictions
            WHERE student_id = ?
            ORDER BY id DESC
        """, (session["student_id"],))

        records = cursor.fetchall()

    finally:
        conn.close()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Attendance",
        "Study Hours",
        "Previous Marks",
        "Assignments",
        "Performance",
        "Date"
    ])

    for record in records:
        writer.writerow(list(record))

    response = make_response(
        output.getvalue()
    )

    response.headers["Content-Disposition"] = (
        "attachment; filename=student_report.csv"
    )

    response.headers["Content-Type"] = (
        "text/csv; charset=utf-8"
    )

    return response

# ---------------- ADMIN DASHBOARD ----------------

@app.route("/admin")
@admin_required
def admin():

    conn = get_db()
    cursor = conn.cursor()

    try:

        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM students
            WHERE role = 'student'
        """)

        total_students = cursor.fetchone()["total"]

        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM predictions
        """)

        total_predictions = cursor.fetchone()["total"]

        cursor.execute("""
            SELECT performance
            FROM predictions
        """)

        all_records = cursor.fetchall()

        performance_counts = get_performance_counts(
            all_records
        )

        cursor.execute("""
            SELECT p.*, s.name, s.email
            FROM predictions p
            LEFT JOIN students s
            ON p.student_id = s.id
            ORDER BY p.id DESC
            LIMIT 10
        """)

        recent_predictions = cursor.fetchall()

    finally:
        conn.close()

    return render_template(
        "admin.html",
        total_students=total_students,
        total_predictions=total_predictions,
        performance_counts=performance_counts,
        recent_predictions=recent_predictions
    )


# ---------------- ADMIN STUDENTS ----------------

@app.route("/admin/students")
@admin_required
def admin_students():

    search = request.args.get(
        "search", ""
    ).strip()

    filter_type = request.args.get(
        "filter_type",
        request.args.get("filter", "all")
    ).strip().lower()

    allowed_filters = {
        "all",
        "high",
        "average",
        "low",
        "excellent",
        "good",
        "poor",
        "not available"
    }

    if filter_type not in allowed_filters:
        filter_type = "all"

    conn = get_db()
    cursor = conn.cursor()

    try:

        query = """
            SELECT
                s.id,
                s.name,
                s.email,
                s.role,

                COALESCE(
                    (
                        SELECT p.performance
                        FROM predictions p
                        WHERE p.student_id = s.id
                        ORDER BY p.id DESC
                        LIMIT 1
                    ),
                    'Not Available'
                ) AS performance

            FROM students s
            WHERE s.role = 'student'
        """

        params = []

        if search:

            query += """
                AND (
                    s.name LIKE ?
                    OR s.email LIKE ?
                )
            """

            params.extend([
                f"%{search}%",
                f"%{search}%"
            ])

        if filter_type != "all":

            query += """
                AND LOWER(COALESCE(
                    (
                        SELECT p.performance
                        FROM predictions p
                        WHERE p.student_id = s.id
                        ORDER BY p.id DESC
                        LIMIT 1
                    ),
                    'not available'
                )) = ?
            """

            params.append(filter_type)

        query += " ORDER BY s.id DESC"

        cursor.execute(query, params)

        students = cursor.fetchall()

        total_students = len(students)

    finally:
        conn.close()

    return render_template(
        "admin_students.html",
        students=students,
        search=search,
        filter_type=filter_type,
        total_students=total_students
    )


# ---------------- ADMIN STUDENT DETAIL ----------------

@app.route("/admin/student/<int:student_id>")
@admin_required
def admin_student_detail(student_id):

    conn = get_db()
    cursor = conn.cursor()

    try:

        cursor.execute("""
            SELECT id, name, email, role
            FROM students
            WHERE id = ? AND role = 'student'
        """, (student_id,))

        student = cursor.fetchone()

        if not student:

            flash(
                "Student not found.",
                "danger"
            )

            return redirect(
                url_for("admin_students")
            )

        cursor.execute("""
            SELECT *
            FROM predictions
            WHERE student_id = ?
            ORDER BY id DESC
        """, (student_id,))

        records = cursor.fetchall()

    finally:
        conn.close()

    return render_template(
        "admin_student_detail.html",
        student=student,
        records=records,
        predictions=records
    )


# ---------------- DELETE STUDENT ----------------

@app.route(
    "/admin/student/delete/<int:student_id>",
    methods=["POST"]
)
@admin_required
def delete_student(student_id):

    conn = get_db()
    cursor = conn.cursor()

    try:

        cursor.execute("""
            SELECT id
            FROM students
            WHERE id = ? AND role = 'student'
        """, (student_id,))

        student = cursor.fetchone()

        if not student:

            flash(
                "Student not found.",
                "danger"
            )

            return redirect(
                url_for("admin_students")
            )

        cursor.execute("""
            UPDATE predictions
            SET student_id = NULL
            WHERE student_id = ?
        """, (student_id,))

        cursor.execute("""
            DELETE FROM students
            WHERE id = ? AND role = 'student'
        """, (student_id,))

        conn.commit()

    except Exception:

        conn.rollback()
        raise

    finally:
        conn.close()

    flash(
        "Student deleted successfully.",
        "success"
    )

    return redirect(
        url_for("admin_students")
    )


# ---------------- ADMIN ALL PREDICTIONS ----------------

@app.route("/admin/predictions")
@admin_required
def admin_predictions():

    conn = get_db()
    cursor = conn.cursor()

    try:

        cursor.execute("""
            SELECT p.*, s.name, s.email
            FROM predictions p
            LEFT JOIN students s
            ON p.student_id = s.id
            ORDER BY p.id DESC
        """)

        predictions = cursor.fetchall()

    finally:
        conn.close()

    return render_template(
        "admin_predictions.html",
        predictions=predictions
    )


# ---------------- ADMIN EXPORT ----------------

@app.route("/admin/students/export")
@admin_required
def export_students():

    conn = get_db()
    cursor = conn.cursor()

    try:

        cursor.execute("""
            SELECT id, name, email
            FROM students
            WHERE role = 'student'
            ORDER BY id DESC
        """)

        students = cursor.fetchall()

    finally:
        conn.close()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "ID",
        "Name",
        "Email"
    ])

    for student in students:
        writer.writerow(list(student))

    response = make_response(
        output.getvalue()
    )

    response.headers["Content-Disposition"] = (
        "attachment; filename=students.csv"
    )

    response.headers["Content-Type"] = (
        "text/csv; charset=utf-8"
    )

    return response


# ---------------- LOGOUT ----------------

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(
        url_for("login")
    )


# ---------------- START APPLICATION ----------------

init_db()
create_admin()


if __name__ == "__main__":

    app.run(
        debug=os.environ.get(
            "FLASK_DEBUG", "true"
        ).lower() == "true"
    )
