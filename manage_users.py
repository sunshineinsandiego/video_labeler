#!/usr/bin/env python3
import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from passlib.hash import bcrypt

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "users.db"
DEFAULT_ADMIN_EMAIL = "cd2859@cumc.columbia.edu"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def create_user(email: str, password: str, is_admin: bool = False) -> None:
    email = _normalize_email(email)
    password_hash = bcrypt.hash(password)
    now = _utc_now()
    with _connect() as conn:
        try:
            conn.execute(
                """
                INSERT INTO users (email, password_hash, is_admin, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (email, password_hash, 1 if is_admin else 0, now, now),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            raise SystemExit(f"User already exists: {email}") from exc


def reset_password(email: str, password: str) -> None:
    email = _normalize_email(email)
    password_hash = bcrypt.hash(password)
    now = _utc_now()
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE users SET password_hash = ?, updated_at = ? WHERE email = ?",
            (password_hash, now, email),
        )
        conn.commit()
        if cur.rowcount == 0:
            raise SystemExit(f"User not found: {email}")


def list_users() -> None:
    with _connect() as conn:
        cur = conn.execute(
            "SELECT email, is_admin, created_at, updated_at FROM users ORDER BY email"
        )
        rows = cur.fetchall()
        if not rows:
            print("No users found.")
            return
        for email, is_admin, created_at, updated_at in rows:
            role = "admin" if is_admin else "user"
            print(f"{email} ({role}) created={created_at} updated={updated_at}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="User management for XR Annotate")
    sub = parser.add_subparsers(dest="command", required=True)

    create_admin = sub.add_parser("create-admin", help="Create the admin user")
    create_admin.add_argument("--email", default=DEFAULT_ADMIN_EMAIL)
    create_admin.add_argument("--password", required=True)

    create_user_cmd = sub.add_parser("create-user", help="Create a regular user")
    create_user_cmd.add_argument("--email", required=True)
    create_user_cmd.add_argument("--password", required=True)

    reset_cmd = sub.add_parser("reset-password", help="Reset a user's password")
    reset_cmd.add_argument("--email", required=True)
    reset_cmd.add_argument("--password", required=True)

    sub.add_parser("list", help="List users")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "create-admin":
        create_user(args.email, args.password, is_admin=True)
        print(f"Admin created: {args.email}")
        return
    if args.command == "create-user":
        create_user(args.email, args.password, is_admin=False)
        print(f"User created: {args.email}")
        return
    if args.command == "reset-password":
        reset_password(args.email, args.password)
        print(f"Password reset: {args.email}")
        return
    if args.command == "list":
        list_users()
        return


if __name__ == "__main__":
    main()
