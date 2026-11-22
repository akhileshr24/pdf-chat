
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "app.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def init_db():
    schema_path = Path(__file__).with_name("schema.sql")
    schema = schema_path.read_text(encoding="utf-8")

    with get_connection() as connection:
        # Create tables if they don't exist.
        connection.executescript(schema)

        # Safely check existing messages table columns.
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(messages)"
            ).fetchall()
        }

        # Add only missing columns.
        if "citations" not in columns:
            connection.execute(
                "ALTER TABLE messages ADD COLUMN citations TEXT"
            )

        if "citation_verification" not in columns:
            connection.execute(
                """
                ALTER TABLE messages
                ADD COLUMN citation_verification TEXT
                """
            )
