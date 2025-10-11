# database.py
import sqlite3
from typing import List, Dict

DB_FILE = "naval_sim.db"

def init_db():
    """Create database and tables if they don't exist"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Table for boats
    c.execute("""
    CREATE TABLE IF NOT EXISTS boats (
        boat_id INTEGER PRIMARY KEY,
        name TEXT,
        vessel_type TEXT,
        threat_level TEXT,
        x REAL,
        y REAL,
        vx REAL,
        vy REAL,
        speed REAL,
        heading REAL,
        status TEXT
    )
    """)

    # Table for crew
    c.execute("""
    CREATE TABLE IF NOT EXISTS crew (
        crew_id INTEGER PRIMARY KEY,
        boat_id INTEGER,
        name TEXT,
        rank TEXT,
        FOREIGN KEY (boat_id) REFERENCES boats(boat_id)
    )
    """)

    # Table for items/weapons
    c.execute("""
    CREATE TABLE IF NOT EXISTS inventory (
        item_id INTEGER PRIMARY KEY,
        boat_id INTEGER,
        item_name TEXT,
        quantity INTEGER,
        item_type TEXT,
        FOREIGN KEY (boat_id) REFERENCES boats(boat_id)
    )
    """)

    # Table to track simulation state (optional)
    c.execute("""
    CREATE TABLE IF NOT EXISTS sim_state (
        sim_id INTEGER PRIMARY KEY,
        time_step INTEGER,
        player_x REAL,
        player_y REAL
    )
    """)

    conn.commit()
    conn.close()
    print("Database initialized.")

# Helper functions
def insert_boat(boat: Dict):
    """Insert a new boat into the database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    INSERT INTO boats (name, vessel_type, threat_level, x, y, vx, vy, speed, heading, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        boat.get("name"),
        boat.get("vessel_type"),
        boat.get("threat_level"),
        boat.get("x"),
        boat.get("y"),
        boat.get("vx"),
        boat.get("vy"),
        boat.get("speed"),
        boat.get("heading"),
        boat.get("status", "active")
    ))
    conn.commit()
    boat_id = c.lastrowid
    conn.close()
    return boat_id

def update_boat_status(boat_id: int, status: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE boats SET status=? WHERE boat_id=?", (status, boat_id))
    conn.commit()
    conn.close()

def get_boat(boat_id: int) -> Dict:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM boats WHERE boat_id=?", (boat_id,))
    row = c.fetchone()
    conn.close()
    if row:
        keys = ["boat_id","name","vessel_type","threat_level","x","y","vx","vy","speed","heading","status"]
        return dict(zip(keys, row))
    return {}

def get_all_boats() -> List[Dict]:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM boats")
    rows = c.fetchall()
    conn.close()
    keys = ["boat_id","name","vessel_type","threat_level","x","y","vx","vy","speed","heading","status"]
    return [dict(zip(keys, r)) for r in rows]

def insert_crew(boat_id: int, crew_members: List[Dict]):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    for member in crew_members:
        c.execute("""
        INSERT INTO crew (boat_id, name, rank)
        VALUES (?, ?, ?)
        """, (boat_id, member["name"], member["rank"]))
    conn.commit()
    conn.close()

def insert_inventory(boat_id: int, items: List[Dict]):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    for item in items:
        c.execute("""
        INSERT INTO inventory (boat_id, item_name, quantity, item_type)
        VALUES (?, ?, ?, ?)
        """, (boat_id, item["item_name"], item["quantity"], item["item_type"]))
    conn.commit()
    conn.close()

# Example usage
if __name__ == "__main__":
    init_db()

    # Example boat
    boat_data = {
        "name": "HMS Test",
        "vessel_type": "Patrol Boat",
        "threat_level": "possible",
        "x": 100,
        "y": 150,
        "vx": 1.2,
        "vy": 0.5,
        "speed": 2.0,
        "heading": 45,
    }

    boat_id = insert_boat(boat_data)
    print("Inserted boat with ID:", boat_id)

    insert_crew(boat_id, [{"name":"John Doe","rank":"Captain"},{"name":"Jane Roe","rank":"Sailor"}])
    insert_inventory(boat_id, [{"item_name":"Rifle","quantity":5,"item_type":"Weapon"}])

    print(get_all_boats())
