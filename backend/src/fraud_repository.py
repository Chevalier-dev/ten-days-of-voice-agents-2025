from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import sqlite3

DB_PATH = Path(__file__).resolve().parents[1] / "shared-data" / "fraud_cases.db"


@dataclass
class FraudCase:
    id: int
    user_name: str
    security_identifier: str
    masked_card: str
    transaction_amount: float
    merchant_name: str
    location: str
    timestamp: str
    transaction_category: str
    transaction_source: str
    security_question: str
    security_answer: str
    status: str
    outcome_note: Optional[str]

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_name": self.user_name,
            "masked_card": self.masked_card,
            "transaction_amount": self.transaction_amount,
            "merchant_name": self.merchant_name,
            "location": self.location,
            "timestamp": self.timestamp,
            "transaction_category": self.transaction_category,
            "transaction_source": self.transaction_source,
            "security_question": self.security_question,
            "status": self.status,
            "outcome_note": self.outcome_note,
        }


class FraudRepository:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = str(db_path)

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def get_pending_case_for_user(self, user_name: str) -> Optional[FraudCase]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, user_name, security_identifier, masked_card,
                   transaction_amount, merchant_name, location, timestamp,
                   transaction_category, transaction_source,
                   security_question, security_answer,
                   status, outcome_note
            FROM fraud_cases
            WHERE user_name = ?
              AND status = 'pending_review'
            ORDER BY id ASC
            LIMIT 1
            """,
            (user_name,),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return FraudCase(*row)

    def verify_security_answer(self, case_id: int, user_answer: str) -> bool:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT security_answer FROM fraud_cases WHERE id = ?", (case_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return False
        correct = row[0].strip().lower()
        return correct == user_answer.strip().lower()

    def update_status(self, case_id: int, status: str, outcome_note: str):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE fraud_cases
            SET status = ?, outcome_note = ?
            WHERE id = ?
            """,
            (status, outcome_note, case_id),
        )
        conn.commit()
        conn.close()
