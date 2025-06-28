import sqlite3
import os

DB_PATH = "cow_detection.db"

def init_db():
    """Initialize the database and create table if not exists."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS cow_detections (
            frame_no INTEGER PRIMARY KEY AUTOINCREMENT,
            cow_count INTEGER NOT NULL,
            image_path TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def insert_detection(cow_count, image_path):
    """Insert one detection row into the table."""
    if cow_count < 1 or not image_path:
        print("⚠️ Invalid insert attempt skipped (cow_count or image_path missing)")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO cow_detections (cow_count, image_path) VALUES (?, ?)", (cow_count, image_path))
    conn.commit()
    conn.close()
