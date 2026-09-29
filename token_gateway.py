"""Refresh-token rotation module for an enterprise identity gateway.

Stdlib only. All state lives in a single SQLite file that may be opened
concurrently by many gateway processes; any process may be SIGKILLed at
any moment.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import sqlite3
import time
import uuid
from typing import Callable, Optional


REASON_UNKNOWN = "unknown"
REASON_EXPIRED = "expired"
REASON_REVOKED = "revoked"
REASON_REUSE = "reuse"

ABSOLUTE_LIFETIME_S = 24 * 3600
IDLE_LIFETIME_S = 8 * 3600
RETRY_WINDOW_S = 60

_TOKEN_RAW_BYTES = 32
_BUSY_TIMEOUT_MS = 30_000
_LOCK_RETRIES = 100
_LOCK_RETRY_SLEEP_S = 0.05

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    created_at    REAL NOT NULL,
    last_activity REAL NOT NULL,
    revoked       INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS tokens (
    token_hash     TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL,
    issued_at      REAL NOT NULL,
    rotated_at     REAL,
    successor_hash TEXT,
    successor_mask BLOB
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


class RefreshError(Exception):
    """Raised by refresh(); ``reason`` is one of unknown/expired/revoked/reuse."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _lookup_hash(token: str) -> str:
    return hashlib.sha256(b"gw-lookup\x00" + token.encode("utf-8")).hexdigest()


def _mask_key(token: str) -> bytes:
    return hashlib.sha256(b"gw-mask\x00" + token.encode("utf-8")).digest()


def _keystream(key: bytes, n: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < n:
        out += hmac.new(key, b"gw-stream" + counter.to_bytes(4, "big"),
                        hashlib.sha256).digest()
        counter += 1
    return bytes(out[:n])


def _xor(data: bytes, stream: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(data, stream))


def _new_token() -> tuple[str, bytes]:
    raw = secrets.token_bytes(_TOKEN_RAW_BYTES)
    return base64.urlsafe_b64encode(raw).decode("ascii"), raw


class TokenGateway:
    def __init__(self, db_path: str, clock: Optional[Callable[[], float]] = None):
        self._db_path = db_path
        self._clock = clock or time.time
        self._init_db()

    # -- internal helpers -------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=_BUSY_TIMEOUT_MS / 1000,
                               isolation_level=None)
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        return conn

    def _init_db(self) -> None:
        for _ in range(_LOCK_RETRIES):
            try:
                conn = self._connect()
                try:
                    conn.executescript(_SCHEMA)
                finally:
                    conn.close()
                return
            except sqlite3.OperationalError:
                time.sleep(_LOCK_RETRY_SLEEP_S)
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    def _transact(self, fn):
        """Run fn(conn) inside a BEGIN IMMEDIATE transaction, retrying on
        lock contention so concurrent processes never see busy errors."""
        last_err: Optional[Exception] = None
        for _ in range(_LOCK_RETRIES):
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    result = fn(conn)
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
                conn.execute("COMMIT")
                return result
            except sqlite3.OperationalError as err:
                last_err = err
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                time.sleep(_LOCK_RETRY_SLEEP_S)
            finally:
                conn.close()
        raise last_err  # type: ignore[misc]

    # -- public API -------------------------------------------------------

    def login(self, user_id: str) -> str:
        def op(conn: sqlite3.Connection) -> str:
            now = self._clock()
            session_id = uuid.uuid4().hex
            token, _raw = _new_token()
            conn.execute(
                "INSERT INTO sessions(id, user_id, created_at, last_activity,"
                " revoked) VALUES (?, ?, ?, ?, 0)",
                (session_id, user_id, now, now))
            conn.execute(
                "INSERT INTO tokens(token_hash, session_id, issued_at)"
                " VALUES (?, ?, ?)",
                (_lookup_hash(token), session_id, now))
            return token

        return self._transact(op)

    def refresh(self, token: str) -> str:
        def op(conn: sqlite3.Connection) -> str:
            # Returns the successor token on success, or a RefreshError
            # instance to be raised after the transaction commits (the
            # reuse path must persist the session revocation).
            row = conn.execute(
                "SELECT session_id, rotated_at, successor_hash, successor_mask"
                " FROM tokens WHERE token_hash = ?",
                (_lookup_hash(token),)).fetchone()
            if row is None:
                return RefreshError(REASON_UNKNOWN)
            session_id, rotated_at, successor_hash, successor_mask = row
            sess = conn.execute(
                "SELECT created_at, last_activity, revoked FROM sessions"
                " WHERE id = ?", (session_id,)).fetchone()
            created_at, last_activity, revoked = sess
            now = self._clock()

            if rotated_at is None:
                if revoked:
                    return RefreshError(REASON_REVOKED)
                if (now - created_at >= ABSOLUTE_LIFETIME_S
                        or now - last_activity >= IDLE_LIFETIME_S):
                    return RefreshError(REASON_EXPIRED)
                successor, succ_raw = _new_token()
                mask = _xor(succ_raw, _keystream(_mask_key(token),
                                                 len(succ_raw)))
                conn.execute(
                    "UPDATE tokens SET rotated_at = ?, successor_hash = ?,"
                    " successor_mask = ? WHERE token_hash = ?",
                    (now, _lookup_hash(successor), mask, _lookup_hash(token)))
                conn.execute(
                    "INSERT INTO tokens(token_hash, session_id, issued_at)"
                    " VALUES (?, ?, ?)",
                    (_lookup_hash(successor), session_id, now))
                conn.execute(
                    "UPDATE sessions SET last_activity = ? WHERE id = ?",
                    (now, session_id))
                return successor

            # Token already retired.
            if revoked:
                return RefreshError(REASON_REVOKED)
            succ_row = conn.execute(
                "SELECT rotated_at FROM tokens WHERE token_hash = ?",
                (successor_hash,)).fetchone()
            successor_used = succ_row is not None and succ_row[0] is not None
            if now - rotated_at < RETRY_WINDOW_S and not successor_used:
                succ_raw = _xor(successor_mask,
                                _keystream(_mask_key(token),
                                           len(successor_mask)))
                return base64.urlsafe_b64encode(succ_raw).decode("ascii")

            conn.execute("UPDATE sessions SET revoked = 1 WHERE id = ?",
                         (session_id,))
            return RefreshError(REASON_REUSE)

        result = self._transact(op)
        if isinstance(result, RefreshError):
            raise result
        return result

    def logout(self, token: str) -> None:
        def op(conn: sqlite3.Connection) -> None:
            row = conn.execute(
                "SELECT session_id FROM tokens WHERE token_hash = ?",
                (_lookup_hash(token),)).fetchone()
            if row is None:
                return
            conn.execute("UPDATE sessions SET revoked = 1 WHERE id = ?",
                         (row[0],))

        self._transact(op)

    def revoke_user(self, user_id: str) -> None:
        def op(conn: sqlite3.Connection) -> None:
            conn.execute("UPDATE sessions SET revoked = 1 WHERE user_id = ?",
                         (user_id,))

        self._transact(op)
