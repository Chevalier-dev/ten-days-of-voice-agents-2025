import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "shared-data" / "fraud_cases.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

print("\n=== fraud_cases ===\n")
for row in cur.execute("SELECT id, user_name, merchant_name, status, outcome_note FROM fraud_cases;"):
    print(row)

conn.close()
