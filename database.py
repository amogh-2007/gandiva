# database.py
import sqlite3
import json
import time
from backend import Vessel  # ✅ ADD THIS IMPORT

def save_boat(boat: Vessel):  # ✅ Add type hint
    """Insert or update a boat in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # ✅ FIXED: Use correct Vessel class attributes
    data = (
        boat.vessel_type,           # ✅ Use vessel_type instead of name
        boat.vessel_type,           # ✅ Use vessel_type for type field
        boat.x,                     # ✅ Use x instead of position[0]
        boat.y,                     # ✅ Use y instead of position[1]
        boat.vx,                    # ✅ Use vx instead of velocity[0]
        boat.vy,                    # ✅ Use vy instead of velocity[1]
        boat.threat_level,
        boat.crew_count,
        json.dumps(boat.items),     # ✅ Use items instead of cargo
        json.dumps(boat.weapons),
        time.time()
    )

    cursor.execute("""
        INSERT INTO boats (name, type, position_x, position_y, velocity_x, velocity_y, 
                           threat_level, crew_count, cargo, weapons, last_update)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)

    conn.commit()
    conn.close()

DB_NAME = "coastal_defender.db"


def get_connection():
    """Create or connect to SQLite database."""
    return sqlite3.connect(DB_NAME)


def setup_database():
    """Initialize all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # Table for storing all boats in the simulation
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS boats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            type TEXT,
            position_x REAL,
            position_y REAL,
            velocity_x REAL,
            velocity_y REAL,
            threat_level TEXT,
            crew_count INTEGER,
            cargo TEXT,
            weapons TEXT,
            last_update REAL
        );
    """)

    # Table for AI's long-term memory (Q-table, policy data)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            q_table_json TEXT NOT NULL
        );
    """)

    # Table for AI training history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            state_json TEXT NOT NULL,
            action TEXT NOT NULL,
            reward REAL NOT NULL,
            next_state_json TEXT NOT NULL
        );
    """)

    conn.commit()
    conn.close()

# ---------------------------------------------------
# 🚢 BOAT MANAGEMENT FUNCTIONS
# ---------------------------------------------------

def save_boat(boat):
    """Insert or update a boat in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    data = (
        boat.name,
        boat.boat_type,
        boat.position[0],
        boat.position[1],
        boat.velocity[0],
        boat.velocity[1],
        boat.threat_level,
        boat.crew_count,
        json.dumps(boat.cargo),
        json.dumps(boat.weapons),
        time.time()
    )

    cursor.execute("""
        INSERT INTO boats (name, type, position_x, position_y, velocity_x, velocity_y, 
                           threat_level, crew_count, cargo, weapons, last_update)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)

    conn.commit()
    conn.close()

def add_vessel(self, x, y, vessel_type, etc):
    vessel = Vessel(...)
    self.db.save_boat(vessel.to_dict())  # Save to DB
    return vessel

def load_boats():
    """Load all boats into the backend."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM boats")
    rows = cursor.fetchall()
    conn.close()
    return rows


def clear_boats():
    """Remove all boats (useful for resetting sim)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM boats")
    conn.commit()
    conn.close()

# ---------------------------------------------------
# 🧠 AI MEMORY FUNCTIONS
# ---------------------------------------------------

def save_q_table(q_table):
    """Save or update the AI's Q-table as JSON."""
    conn = get_connection()
    cursor = conn.cursor()
    q_json = json.dumps(q_table)
    cursor.execute("INSERT INTO ai_memory (q_table_json) VALUES (?)", (q_json,))
    conn.commit()
    conn.close()


def load_q_table():
    """Load the most recent Q-table from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT q_table_json FROM ai_memory ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return {}

# ---------------------------------------------------
# 🧩 TRAINING LOG FUNCTIONS
# ---------------------------------------------------

def log_decision(session_id, state, action, reward, next_state):
    """Log one AI decision into the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO training_log (session_id, timestamp, state_json, action, reward, next_state_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, time.time(), json.dumps(state), action, reward, json.dumps(next_state)))
    conn.commit()
    conn.close()
