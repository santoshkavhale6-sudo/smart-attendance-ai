import os
import sqlite3
from pathlib import Path

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'database', 'attendance.db')

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    if not Path(schema_path).exists():
        raise FileNotFoundError(f'Schema file not found: {schema_path}')
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    conn = get_connection()
    conn.executescript(schema_sql)
    conn.commit()
    conn.close()

if __name__ == '__main__':
    initialize_database()
    print('Database initialized at', DB_PATH)
