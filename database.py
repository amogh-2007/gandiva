# database.py - UPDATED TO MATCH OTHER FILES
import sqlite3
import json
import time
from typing import Dict, List, Optional, Any

DB_NAME = "coastal_defender.db"

def get_connection():
    """Create or connect to SQLite database."""
    return sqlite3.connect(DB_NAME)

def setup_database():
    """Initialize all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # Updated table for storing all boats in the simulation - MATCHES Vessel class
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vessels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vessel_type TEXT NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            vx REAL DEFAULT 0.0,
            vy REAL DEFAULT 0.0,
            speed REAL DEFAULT 0.0,
            heading REAL DEFAULT 0.0,
            threat_level TEXT DEFAULT 'unknown',
            true_threat_level TEXT DEFAULT 'neutral',
            scanned BOOLEAN DEFAULT 0,
            active BOOLEAN DEFAULT 1,
            distance_from_patrol REAL DEFAULT 9999.0,
            crew_count INTEGER DEFAULT 0,
            items TEXT DEFAULT '[]',
            weapons TEXT DEFAULT '[]',
            detection_range INTEGER DEFAULT 200,
            aggressiveness REAL DEFAULT 0.1,
            evasion_chance REAL DEFAULT 0.1,
            behavior TEXT DEFAULT 'idle',
            last_update REAL NOT NULL
        );
    """)

    # Table for AI's long-term memory (model weights, Q-values)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            model_type TEXT DEFAULT 'dqn',
            weights_json TEXT NOT NULL,
            epsilon REAL DEFAULT 1.0,
            training_step INTEGER DEFAULT 0,
            cumulative_reward REAL DEFAULT 0.0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
    """)

    # Table for AI training history with enhanced metrics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            episode INTEGER DEFAULT 0,
            step INTEGER DEFAULT 0,
            timestamp REAL NOT NULL,
            state_json TEXT NOT NULL,
            action TEXT NOT NULL,
            reward REAL NOT NULL,
            next_state_json TEXT NOT NULL,
            done BOOLEAN DEFAULT 0,
            confidence REAL DEFAULT 0.0,
            vessel_id INTEGER,
            true_threat_level TEXT,
            predicted_threat_level TEXT,
            correct BOOLEAN DEFAULT 0
        );
    """)

    # Table for performance metrics tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            episode INTEGER DEFAULT 0,
            total_decisions INTEGER DEFAULT 0,
            hitl_requests INTEGER DEFAULT 0,
            successful_intercepts INTEGER DEFAULT 0,
            false_positives INTEGER DEFAULT 0,
            missed_threats INTEGER DEFAULT 0,
            correct_monitors INTEGER DEFAULT 0,
            correct_ignores INTEGER DEFAULT 0,
            cumulative_reward REAL DEFAULT 0.0,
            threat_detection_accuracy REAL DEFAULT 0.0,
            average_confidence REAL DEFAULT 0.0,
            epsilon REAL DEFAULT 1.0
        );
    """)

    # Table for communication logs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS communication_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            vessel_id INTEGER,
            vessel_type TEXT,
            player_message TEXT,
            vessel_reply TEXT,
            threat_level TEXT,
            is_suspicious BOOLEAN DEFAULT 0,
            session_id TEXT
        );
    """)

    conn.commit()
    conn.close()
    print("✅ Database setup completed with updated schema")

# ---------------------------------------------------
# 🚢 VESSEL MANAGEMENT FUNCTIONS - UPDATED
# ---------------------------------------------------

def save_vessel(vessel_data: Dict[str, Any]) -> int:
    """Insert or update a vessel in the database. Returns vessel ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Convert lists to JSON strings
    items_json = json.dumps(vessel_data.get('items', []))
    weapons_json = json.dumps(vessel_data.get('weapons', []))
    
    data = (
        vessel_data.get('vessel_type', 'Unknown'),
        vessel_data.get('x', 400.0),
        vessel_data.get('y', 300.0),
        vessel_data.get('vx', 0.0),
        vessel_data.get('vy', 0.0),
        vessel_data.get('speed', 0.0),
        vessel_data.get('heading', 0.0),
        vessel_data.get('threat_level', 'unknown'),
        vessel_data.get('true_threat_level', 'neutral'),
        1 if vessel_data.get('scanned', False) else 0,
        1 if vessel_data.get('active', True) else 0,
        vessel_data.get('distance_from_patrol', 9999.0),
        vessel_data.get('crew_count', 0),
        items_json,
        weapons_json,
        vessel_data.get('detection_range', 200),
        vessel_data.get('aggressiveness', 0.1),
        vessel_data.get('evasion_chance', 0.1),
        vessel_data.get('behavior', 'idle'),
        time.time()
    )

    if 'id' in vessel_data and vessel_data['id']:
        # Update existing vessel
        cursor.execute("""
            UPDATE vessels SET 
            vessel_type=?, x=?, y=?, vx=?, vy=?, speed=?, heading=?,
            threat_level=?, true_threat_level=?, scanned=?, active=?,
            distance_from_patrol=?, crew_count=?, items=?, weapons=?,
            detection_range=?, aggressiveness=?, evasion_chance=?, behavior=?, last_update=?
            WHERE id=?
        """, data + (vessel_data['id'],))
        vessel_id = vessel_data['id']
    else:
        # Insert new vessel
        cursor.execute("""
            INSERT INTO vessels (
                vessel_type, x, y, vx, vy, speed, heading, threat_level,
                true_threat_level, scanned, active, distance_from_patrol,
                crew_count, items, weapons, detection_range, aggressiveness,
                evasion_chance, behavior, last_update
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        vessel_id = cursor.lastrowid

    conn.commit()
    conn.close()
    return vessel_id

def load_vessels() -> List[Dict[str, Any]]:
    """Load all vessels from database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vessels WHERE active = 1")
    rows = cursor.fetchall()
    conn.close()
    
    vessels = []
    for row in rows:
        vessel = {
            'id': row[0],
            'vessel_type': row[1],
            'x': row[2],
            'y': row[3],
            'vx': row[4],
            'vy': row[5],
            'speed': row[6],
            'heading': row[7],
            'threat_level': row[8],
            'true_threat_level': row[9],
            'scanned': bool(row[10]),
            'active': bool(row[11]),
            'distance_from_patrol': row[12],
            'crew_count': row[13],
            'items': json.loads(row[14]),
            'weapons': json.loads(row[15]),
            'detection_range': row[16],
            'aggressiveness': row[17],
            'evasion_chance': row[18],
            'behavior': row[19],
            'last_update': row[20]
        }
        vessels.append(vessel)
    
    return vessels

def get_vessel(vessel_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific vessel by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vessels WHERE id = ?", (vessel_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'id': row[0],
            'vessel_type': row[1],
            'x': row[2],
            'y': row[3],
            'vx': row[4],
            'vy': row[5],
            'speed': row[6],
            'heading': row[7],
            'threat_level': row[8],
            'true_threat_level': row[9],
            'scanned': bool(row[10]),
            'active': bool(row[11]),
            'distance_from_patrol': row[12],
            'crew_count': row[13],
            'items': json.loads(row[14]),
            'weapons': json.loads(row[15]),
            'detection_range': row[16],
            'aggressiveness': row[17],
            'evasion_chance': row[18],
            'behavior': row[19],
            'last_update': row[20]
        }
    return None

def deactivate_vessel(vessel_id: int):
    """Soft delete a vessel by setting active to 0."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE vessels SET active = 0 WHERE id = ?", (vessel_id,))
    conn.commit()
    conn.close()

def clear_vessels():
    """Remove all vessels (useful for resetting sim)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM vessels")
    conn.commit()
    conn.close()

# ---------------------------------------------------
# 🧠 AI MEMORY FUNCTIONS - UPDATED
# ---------------------------------------------------

def save_ai_model(session_id: str, model_weights: Dict, epsilon: float = 1.0, 
                  training_step: int = 0, cumulative_reward: float = 0.0):
    """Save AI model weights and state."""
    conn = get_connection()
    cursor = conn.cursor()
    
    weights_json = json.dumps(model_weights)
    current_time = time.time()
    
    # Check if session exists
    cursor.execute("SELECT id FROM ai_memory WHERE session_id = ?", (session_id,))
    existing = cursor.fetchone()
    
    if existing:
        # Update existing
        cursor.execute("""
            UPDATE ai_memory SET 
            weights_json = ?, epsilon = ?, training_step = ?, 
            cumulative_reward = ?, updated_at = ?
            WHERE session_id = ?
        """, (weights_json, epsilon, training_step, cumulative_reward, current_time, session_id))
    else:
        # Insert new
        cursor.execute("""
            INSERT INTO ai_memory (session_id, weights_json, epsilon, training_step, 
                                 cumulative_reward, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, weights_json, epsilon, training_step, cumulative_reward, 
              current_time, current_time))
    
    conn.commit()
    conn.close()

def load_ai_model(session_id: str) -> Optional[Dict[str, Any]]:
    """Load AI model weights and state."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT weights_json, epsilon, training_step, cumulative_reward 
        FROM ai_memory WHERE session_id = ? ORDER BY updated_at DESC LIMIT 1
    """, (session_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'weights': json.loads(row[0]),
            'epsilon': row[1],
            'training_step': row[2],
            'cumulative_reward': row[3]
        }
    return None

def get_recent_ai_sessions(limit: int = 5) -> List[Dict[str, Any]]:
    """Get recent AI training sessions."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT session_id, epsilon, training_step, cumulative_reward, updated_at
        FROM ai_memory ORDER BY updated_at DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    sessions = []
    for row in rows:
        sessions.append({
            'session_id': row[0],
            'epsilon': row[1],
            'training_step': row[2],
            'cumulative_reward': row[3],
            'last_updated': row[4]
        })
    
    return sessions

# ---------------------------------------------------
# 📊 TRAINING LOG FUNCTIONS - UPDATED
# ---------------------------------------------------

def log_training_decision(session_id: str, state: Dict, action: str, reward: float, 
                         next_state: Dict, done: bool = False, confidence: float = 0.0,
                         vessel_id: int = None, true_threat_level: str = None, 
                         episode: int = 0, step: int = 0):
    """Log one AI training decision with enhanced metrics."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Calculate if prediction was correct
    predicted_threat = None
    correct = None
    if true_threat_level and action:
        # Simple heuristic for threat assessment correctness
        if true_threat_level == "confirmed" and action == "intercept":
            correct = 1
        elif true_threat_level == "neutral" and action == "ignore":
            correct = 1
        else:
            correct = 0
    
    cursor.execute("""
        INSERT INTO training_log (
            session_id, episode, step, timestamp, state_json, action, reward,
            next_state_json, done, confidence, vessel_id, true_threat_level,
            predicted_threat_level, correct
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id, episode, step, time.time(), json.dumps(state), action, reward,
        json.dumps(next_state), 1 if done else 0, confidence, vessel_id,
        true_threat_level, predicted_threat, correct
    ))
    
    conn.commit()
    conn.close()

def get_training_history(session_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Get training history with optional session filter."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if session_id:
        cursor.execute("""
            SELECT * FROM training_log 
            WHERE session_id = ? 
            ORDER BY timestamp DESC LIMIT ?
        """, (session_id, limit))
    else:
        cursor.execute("""
            SELECT * FROM training_log 
            ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    logs = []
    for row in rows:
        logs.append({
            'log_id': row[0],
            'session_id': row[1],
            'episode': row[2],
            'step': row[3],
            'timestamp': row[4],
            'state': json.loads(row[5]),
            'action': row[6],
            'reward': row[7],
            'next_state': json.loads(row[8]),
            'done': bool(row[9]),
            'confidence': row[10],
            'vessel_id': row[11],
            'true_threat_level': row[12],
            'predicted_threat_level': row[13],
            'correct': bool(row[14]) if row[14] is not None else None
        })
    
    return logs

# ---------------------------------------------------
# 📈 PERFORMANCE METRICS FUNCTIONS
# ---------------------------------------------------

def log_performance_metrics(session_id: str, metrics: Dict[str, Any]):
    """Log performance metrics for analysis."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO performance_metrics (
            session_id, timestamp, episode, total_decisions, hitl_requests,
            successful_intercepts, false_positives, missed_threats,
            correct_monitors, correct_ignores, cumulative_reward,
            threat_detection_accuracy, average_confidence, epsilon
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        time.time(),
        metrics.get('episode', 0),
        metrics.get('total_decisions', 0),
        metrics.get('hitl_requests', 0),
        metrics.get('successful_intercepts', 0),
        metrics.get('false_positives', 0),
        metrics.get('missed_threats', 0),
        metrics.get('correct_monitors', 0),
        metrics.get('correct_ignores', 0),
        metrics.get('cumulative_reward', 0.0),
        metrics.get('threat_detection_accuracy', 0.0),
        metrics.get('average_confidence', 0.0),
        metrics.get('epsilon', 1.0)
    ))
    
    conn.commit()
    conn.close()

def get_performance_history(session_id: str = None) -> List[Dict[str, Any]]:
    """Get performance metrics history."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if session_id:
        cursor.execute("""
            SELECT * FROM performance_metrics 
            WHERE session_id = ? ORDER BY timestamp
        """, (session_id,))
    else:
        cursor.execute("SELECT * FROM performance_metrics ORDER BY timestamp")
    
    rows = cursor.fetchall()
    conn.close()
    
    metrics = []
    for row in rows:
        metrics.append({
            'id': row[0],
            'session_id': row[1],
            'timestamp': row[2],
            'episode': row[3],
            'total_decisions': row[4],
            'hitl_requests': row[5],
            'successful_intercepts': row[6],
            'false_positives': row[7],
            'missed_threats': row[8],
            'correct_monitors': row[9],
            'correct_ignores': row[10],
            'cumulative_reward': row[11],
            'threat_detection_accuracy': row[12],
            'average_confidence': row[13],
            'epsilon': row[14]
        })
    
    return metrics

# ---------------------------------------------------
# 📡 COMMUNICATION LOG FUNCTIONS
# ---------------------------------------------------

def log_communication(vessel_id: int, vessel_type: str, player_message: str, 
                     vessel_reply: str, threat_level: str, is_suspicious: bool = False,
                     session_id: str = "main"):
    """Log communication interactions."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO communication_log (
            timestamp, vessel_id, vessel_type, player_message, 
            vessel_reply, threat_level, is_suspicious, session_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        time.time(), vessel_id, vessel_type, player_message,
        vessel_reply, threat_level, 1 if is_suspicious else 0, session_id
    ))
    
    conn.commit()
    conn.close()

def get_communication_log(vessel_id: int = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Get communication history."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if vessel_id:
        cursor.execute("""
            SELECT * FROM communication_log 
            WHERE vessel_id = ? ORDER BY timestamp DESC LIMIT ?
        """, (vessel_id, limit))
    else:
        cursor.execute("""
            SELECT * FROM communication_log 
            ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    communications = []
    for row in rows:
        communications.append({
            'id': row[0],
            'timestamp': row[1],
            'vessel_id': row[2],
            'vessel_type': row[3],
            'player_message': row[4],
            'vessel_reply': row[5],
            'threat_level': row[6],
            'is_suspicious': bool(row[7]),
            'session_id': row[8]
        })
    
    return communications

# ---------------------------------------------------
# 🧹 DATABASE MAINTENANCE FUNCTIONS
# ---------------------------------------------------

def cleanup_old_data(days_old: int = 30):
    """Clean up old data to prevent database bloat."""
    conn = get_connection()
    cursor = conn.cursor()
    cutoff_time = time.time() - (days_old * 24 * 60 * 60)
    
    # Clean up old training logs
    cursor.execute("DELETE FROM training_log WHERE timestamp < ?", (cutoff_time,))
    
    # Clean up old performance metrics
    cursor.execute("DELETE FROM performance_metrics WHERE timestamp < ?", (cutoff_time,))
    
    # Clean up old communication logs
    cursor.execute("DELETE FROM communication_log WHERE timestamp < ?", (cutoff_time,))
    
    # Keep only latest AI models per session
    cursor.execute("""
        DELETE FROM ai_memory 
        WHERE id NOT IN (
            SELECT id FROM ai_memory 
            WHERE session_id IN (
                SELECT DISTINCT session_id FROM ai_memory
            )
            GROUP BY session_id 
            HAVING MAX(updated_at)
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"🧹 Cleaned up data older than {days_old} days")

def get_database_stats() -> Dict[str, int]:
    """Get database statistics."""
    conn = get_connection()
    cursor = conn.cursor()
    
    stats = {}
    
    # Count records in each table
    tables = ['vessels', 'ai_memory', 'training_log', 'performance_metrics', 'communication_log']
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        stats[table] = cursor.fetchone()[0]
    
    # Get latest update times
    cursor.execute("SELECT MAX(last_update) FROM vessels WHERE active = 1")
    stats['last_vessel_update'] = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT MAX(updated_at) FROM ai_memory")
    stats['last_ai_update'] = cursor.fetchone()[0] or 0
    
    conn.close()
    return stats

# Initialize database when module is imported
setup_database()
print("✅ Database module initialized and ready")