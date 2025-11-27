from pathlib import Path
import sqlite3

# DB will live in backend/shared-data/fraud_cases.db
DB_PATH = Path(__file__).resolve().parents[1] / "shared-data" / "fraud_cases.db"


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fraud_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT NOT NULL,
            security_identifier TEXT NOT NULL,
            masked_card TEXT NOT NULL,
            transaction_amount REAL NOT NULL,
            merchant_name TEXT NOT NULL,
            location TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            transaction_category TEXT NOT NULL,
            transaction_source TEXT NOT NULL,
            security_question TEXT NOT NULL,
            security_answer TEXT NOT NULL,
            status TEXT NOT NULL,
            outcome_note TEXT
        );
        """
    )

    # Clear old data (for dev)
    cur.execute("DELETE FROM fraud_cases;")

    sample_cases = [
        (
            "John",
            "12345",
            "**** 4242",
            249.99,
            "ABC Industries",
            "San Francisco, USA",
            "2025-11-25 14:32",
            "e-commerce",
            "alibaba.com",
            "What is your favorite color?",
            "blue",
            "pending_review",
            None,
        ),
        (
            "John",
            "12345",
            "**** 4242",
            1299.00,
            "GizmoTech Online",
            "New York, USA",
            "2025-11-26 08:15",
            "electronics",
            "gizmoshop.com",
            "What is your favorite color?",
            "blue",
            "pending_review",
            None,
        ),
        (
            "Alice",
            "67890",
            "**** 9876",
            59.50,
            "QuickMart Grocery",
            "Chicago, USA",
            "2025-11-20 19:45",
            "groceries",
            "quickmart.com",
            "What city were you born in?",
            "boston",
            "pending_review",
            None,
        ),
    ]

    cur.executemany(
        """
        INSERT INTO fraud_cases (
            user_name,
            security_identifier,
            masked_card,
            transaction_amount,
            merchant_name,
            location,
            timestamp,
            transaction_category,
            transaction_source,
            security_question,
            security_answer,
            status,
            outcome_note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        sample_cases,
    )

    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {DB_PATH}")


if __name__ == "__main__":
    init_db()
