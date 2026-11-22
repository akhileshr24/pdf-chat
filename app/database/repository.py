
from contextlib import closing
from uuid import uuid4
import json

from app.database.connection import get_connection


def save_document(document_id, filename, file_path):
    with closing(get_connection()) as db, db:
        db.execute(
            """
            INSERT OR IGNORE INTO documents (
                id, filename, file_path
            )
            VALUES (?, ?, ?)
            """,
            (document_id, filename, file_path),
        )


def get_documents():
    with closing(get_connection()) as db:
        rows = db.execute(
            """
            SELECT *
            FROM documents
            ORDER BY created_at DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]


def create_conversation(title=None):
    conversation_id = str(uuid4())

    with closing(get_connection()) as db, db:
        if title is None:
            count = db.execute(
                "SELECT COUNT(*) FROM conversations"
            ).fetchone()[0]

            title = f"Conversation {count + 1}"

        db.execute(
            """
            INSERT INTO conversations (id, title)
            VALUES (?, ?)
            """,
            (conversation_id, title),
        )

    return conversation_id


def get_conversations():
    with closing(get_connection()) as db:
        rows = db.execute(
            """
            SELECT *
            FROM conversations
            ORDER BY created_at DESC, rowid DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]


def update_conversation_title(conversation_id, title):
    with closing(get_connection()) as db, db:
        db.execute(
            """
            UPDATE conversations
            SET title = ?
            WHERE id = ?
            """,
            (title, conversation_id),
        )


def save_message(
    conversation_id,
    role,
    content,
    citations=None,
    citation_verification=None,
):
    message_id = str(uuid4())

    with closing(get_connection()) as db, db:
        db.execute(
            """
            INSERT INTO messages (
                id,
                conversation_id,
                role,
                content,
                citations,
                citation_verification
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                conversation_id,
                role,
                content,
                json.dumps(citations) if citations is not None else None,
                json.dumps(citation_verification)
                if citation_verification is not None
                else None,
            ),
        )

    return message_id


def get_messages(conversation_id):
    with closing(get_connection()) as db:
        rows = db.execute(
            """
            SELECT *
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC, rowid ASC
            """,
            (conversation_id,),
        ).fetchall()

    messages = []

    for row in rows:
        message = dict(row)

        if message["citations"]:
            message["citations"] = json.loads(
                message["citations"]
            )

        if message["citation_verification"]:
            message["citation_verification"] = json.loads(
                message["citation_verification"]
            )

        messages.append(message)

    return messages


def attach_document(conversation_id, document_id):
    with closing(get_connection()) as db, db:
        db.execute(
            """
            INSERT OR IGNORE INTO conversation_documents (
                conversation_id,
                document_id
            )
            VALUES (?, ?)
            """,
            (conversation_id, document_id),
        )


def get_conversation_documents(conversation_id):
    with closing(get_connection()) as db:
        rows = db.execute(
            """
            SELECT d.*
            FROM documents d
            JOIN conversation_documents cd
                ON d.id = cd.document_id
            WHERE cd.conversation_id = ?
            ORDER BY d.created_at DESC
            """,
            (conversation_id,),
        ).fetchall()

        return [dict(row) for row in rows]


def conversation_exists(conversation_id):
    with closing(get_connection()) as db:
        row = db.execute(
            """
            SELECT 1
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()

        return row is not None
