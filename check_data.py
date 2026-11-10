
import sqlite3

conn = sqlite3.connect("students.db")
cursor = conn.cursor()

cursor.execute("""
    SELECT id, student_id, performance
    FROM predictions
    ORDER BY id DESC
""")

records = cursor.fetchall()

print("\n----- PREDICTION DATABASE -----")

for record in records:
    print(record)

print("\nTotal records:", len(records))

conn.close()
