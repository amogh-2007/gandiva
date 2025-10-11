# database_manager.py
import sqlite3
from datetime import datetime

DB_FILE = "coastal_defender.db"


# ---------------------------
# Connection Helper
# ---------------------------
def connect_db():
    """Create and return a database connection."""
    return sqlite3.connect(DB_FILE)


# ---------------------------
# Database Initialization
# ---------------------------
def initialize_db():
    """Create tables if they don’t exist."""
    conn = connect_db()
    c = conn.cursor()

    # Main boats table
    c.execute("""
    CREATE TABLE IF NOT EXISTS boats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        type TEXT,
        x REAL,
        y REAL,
        speed REAL,
        direction REAL,
        status TEXT,
        threat_level REAL,
        crew INTEGER,
        cargo TEXT,
        weapons TEXT
    )
    """)

    # Logs table for actions
    c.execute("""
    CREATE TABLE IF NOT EXISTS simulation_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        boat_id INTEGER,
        action TEXT,
        timestamp TEXT,
        details TEXT,
        FOREIGN KEY (boat_id) REFERENCES boats(id)
    )
    """)

    conn.commit()
    conn.close()


# ---------------------------
# Boat Operations
# ---------------------------
def insert_boat(data: dict):
    """Insert a new boat record."""
    conn = connect_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO boats (name, type, x, y, speed, direction, status, threat_level, crew, cargo, weapons)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("name"), data.get("type"), data.get("x"), data.get("y"),
        data.get("speed"), data.get("direction"), data.get("status"),
        data.get("threat_level"), data.get("crew"), data.get("cargo"),
        data.get("weapons")
    ))
    conn.commit()
    conn.close()


def get_all_boats():
    """Fetch all boats."""
    conn = connect_db()
    c = conn.cursor()
    c.execute("SELECT * FROM boats")
    boats = c.fetchall()
    conn.close()
    return boats


def get_boat_by_id(boat_id: int):
    """Fetch a specific boat by its ID."""
    conn = connect_db()
    c = conn.cursor()
    c.execute("SELECT * FROM boats WHERE id=?", (boat_id,))
    boat = c.fetchone()
    conn.close()
    return boat


def update_boat_position(boat_id: int, x: float, y: float):
    """Update a boat’s position."""
    conn = connect_db()
    c = conn.cursor()
    c.execute("UPDATE boats SET x=?, y=? WHERE id=?", (x, y, boat_id))
    conn.commit()
    conn.close()


def update_boat_status(boat_id: int, new_status: str):
    """Update a boat’s status (e.g., threat, suspect, good, intercepted)."""
    conn = connect_db()
    c = conn.cursor()
    c.execute("UPDATE boats SET status=? WHERE id=?", (new_status, boat_id))
    conn.commit()
    conn.close()


def delete_boat(boat_id: int):
    """Remove a boat (if destroyed or removed from zone)."""
    conn = connect_db()
    c = conn.cursor()
    c.execute("DELETE FROM boats WHERE id=?", (boat_id,))
    conn.commit()
    conn.close()


# ---------------------------
# Logging System
# ---------------------------
def log_action(boat_id: int, action: str, details: str = ""):
    """Log an event in the simulation (e.g., intercept, rescue)."""
    conn = connect_db()
    c = conn.cursor()
    timestamp = datetime.now().isoformat(timespec="seconds")
    c.execute("""
        INSERT INTO simulation_logs (boat_id, action, timestamp, details)
        VALUES (?, ?, ?, ?)
    """, (boat_id, action, timestamp, details))
    conn.commit()
    conn.close()
