import sqlite3
from datetime import datetime

DB_NAME = "detections.db"

# Initialize DB and ensure correct schema
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Create table with the correct schema
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            cow INTEGER,
            stranger_cow INTEGER,
            dog INTEGER,
            timestamp TEXT
        )
    ''')

    # Optional: Ensure all required columns exist
    # You can also do ALTER TABLE here if upgrading from old schema

    conn.commit()
    conn.close()

# Insert new detection into table
def insert_detection(image_path, cow, stranger_cow, dog):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO detections (image_path, cow, stranger_cow, dog, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (image_path, cow, stranger_cow, dog, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# Fetch all detection records
def fetch_all_detections():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM detections")
    rows = c.fetchall()
    conn.close()
    return rows

# Query records by detected label type
def query_by_label(label):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    if label == "cow":
        c.execute("SELECT * FROM detections WHERE cow > 0")
    elif label == "stranger_cow":
        c.execute("SELECT * FROM detections WHERE stranger_cow > 0")
    elif label == "dog":
        c.execute("SELECT * FROM detections WHERE dog > 0")
    else:
        conn.close()
        return []
    rows = c.fetchall()
    conn.close()
    return rows
