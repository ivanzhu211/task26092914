"""企业身份网关 refresh token 轮换模块。

只用标准库，全部状态放在一个 SQLite 文件中，支持多进程并发与 SIGKILL 崩溃安全。

设计要点：
- token 是 32 字节随机数的 urlsafe base64，数据库里只存 SHA-256 哈希，
  文件本身不含任何可直接使用的 token。
- refresh 在同一个 BEGIN IMMEDIATE 事务里完成“退役旧 token + 签发后继”，
  后继的明文用旧 token 派生的密钥异或包裹后存库（succ_wrapped），
  因此崩溃后重试可以还原出逐字节相同的后继，而仅凭数据库文件无法算出它。
- 重试窗口：旧 token 首次轮换后 60 秒内、且后继尚未被用来轮换过时，
  再次出示旧 token 返回同一个后继，不算重用，也不改变任何计时。
- 除此之外出示任何已退役 token：整个会话立即持久作废，返回 reuse。
"""

from __future__ import annotations

import base64
import hashlib
import os
import sqlite3
import time
from typing import Callable

REASON_UNKNOWN = "unknown"
REASON_EXPIRED = "expired"
REASON_REVOKED = "revoked"
REASON_REUSE = "reuse"

ABSOLUTE_TTL_SECONDS = 24 * 3600
IDLE_TTL_SECONDS = 8 * 3600
RETRY_WINDOW_SECONDS = 60

_BUSY_TIMEOUT_MS = 30000
_LOCK_RETRY_LIMIT = 50
_LOCK_RETRY_SLEEP = 0.05

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT    NOT NULL,
    created_at    REAL    NOT NULL,
    last_activity REAL    NOT NULL,
    revoked       INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS tokens (
    token_hash     TEXT PRIMARY KEY,
    session_id     INTEGER NOT NULL REFERENCES sessions(session_id),
    created_at     REAL    NOT NULL,
    rotated_at     REAL,
    successor_hash TEXT,
    succ_wrapped   BLOB
);
CREATE INDEX IF NOT EXISTS idx_tokens_session ON tokens(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


class RefreshError(Exception):
    """refresh/logout 失败，reason 只有 unknown/expired/revoked/reuse 四种。"""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _new_token() -> str:
    return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _wrap_key(token: str) -> bytes:
    return hashlib.sha256(b"refresh-gate-wrap-v1\x00" + token.encode("utf-8")).digest()


def _wrap(token: str, successor: str) -> bytes:
    raw = base64.urlsafe_b64decode(successor.encode("ascii"))
    key = _wrap_key(token)
    return bytes(a ^ b for a, b in zip(raw, key))


def _unwrap(token: str, wrapped: bytes) -> str:
    key = _wrap_key(token)
    raw = bytes(a ^ b for a, b in zip(wrapped, key))
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _is_lock_error(exc: sqlite3.OperationalError) -> bool:
    msg = str(exc).lower()
    return "locked" in msg or "busy" in msg


class _Fail:
    """事务体内返回的失败标记（提交后再抛异常，保证熔断写入持久化）。"""

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason


class TokenGateway:
    """基于单个 SQLite 文件的 refresh token 轮换网关。

    clock: 可注入时钟，返回 Unix 秒（float）。所有过期判断都走它。
    """

    def __init__(self, db_path: str, clock: Callable[[], float] = time.time) -> None:
        self._db_path = str(db_path)
        self._clock = clock
        self._init_db()

    # ---- 内部基础设施 ----

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=_BUSY_TIMEOUT_MS / 1000.0,
                               isolation_level=None)
        conn.execute("PRAGMA busy_timeout=%d" % _BUSY_TIMEOUT_MS)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        for _ in range(_LOCK_RETRY_LIMIT):
            try:
                conn = self._connect()
                try:
                    conn.executescript(_SCHEMA)
                finally:
                    conn.close()
                return
            except sqlite3.OperationalError as exc:
                if _is_lock_error(exc):
                    time.sleep(_LOCK_RETRY_SLEEP)
                    continue
                raise
        raise sqlite3.OperationalError("database initialization keeps failing on locks")

    def _transact(self, fn):
        """在 BEGIN IMMEDIATE 事务里执行 fn(conn)，锁冲突时整体重试。

        fn 返回 _Fail 时事务正常提交（熔断等写入持久化），随后抛出 RefreshError。
        """
        for _ in range(_LOCK_RETRY_LIMIT):
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                result = fn(conn)
                conn.execute("COMMIT")
            except sqlite3.OperationalError as exc:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                if _is_lock_error(exc):
                    time.sleep(_LOCK_RETRY_SLEEP)
                    continue
                raise
            except BaseException:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()
            if isinstance(result, _Fail):
                raise RefreshError(result.reason)
            return result
        raise sqlite3.OperationalError("transaction keeps failing on locks")

    # ---- 四个操作 ----

    def login(self, user_id: str) -> str:
        now = self._clock()
        token = _new_token()

        def fn(conn: sqlite3.Connection):
            cur = conn.execute(
                "INSERT INTO sessions(user_id, created_at, last_activity, revoked)"
                " VALUES (?, ?, ?, 0)",
                (user_id, now, now),
            )
            conn.execute(
                "INSERT INTO tokens(token_hash, session_id, created_at)"
                " VALUES (?, ?, ?)",
                (_token_hash(token), cur.lastrowid, now),
            )

        self._transact(fn)
        return token

    def refresh(self, token: str) -> str:
        now = self._clock()
        token_hash = _token_hash(token)

        def fn(conn: sqlite3.Connection):
            row = conn.execute(
                "SELECT t.session_id, t.rotated_at, t.successor_hash, t.succ_wrapped,"
                "       s.revoked, s.created_at, s.last_activity"
                " FROM tokens t JOIN sessions s ON s.session_id = t.session_id"
                " WHERE t.token_hash = ?",
                (token_hash,),
            ).fetchone()
            if row is None:
                return _Fail(REASON_UNKNOWN)
            (session_id, rotated_at, successor_hash, succ_wrapped,
             revoked, session_created, last_activity) = row

            if revoked:
                # 撤销优先：logout/revoke_user 返回后，会话里任何 token
                # （包括重试窗口内的已退役 token）都不得再换出后继。
                return _Fail(REASON_REVOKED)

            if rotated_at is not None:
                # 已退役 token：先判断是否为允许的重试。
                succ_row = conn.execute(
                    "SELECT rotated_at FROM tokens WHERE token_hash = ?",
                    (successor_hash,),
                ).fetchone()
                succ_unused = succ_row is not None and succ_row[0] is None
                if now - rotated_at < RETRY_WINDOW_SECONDS and succ_unused:
                    return _unwrap(token, succ_wrapped)
                # 重用熔断：立即持久作废整个会话。
                conn.execute(
                    "UPDATE sessions SET revoked = 1 WHERE session_id = ?",
                    (session_id,),
                )
                return _Fail(REASON_REUSE)

            if (now >= session_created + ABSOLUTE_TTL_SECONDS
                    or now >= last_activity + IDLE_TTL_SECONDS):
                return _Fail(REASON_EXPIRED)

            successor = _new_token()
            conn.execute(
                "INSERT INTO tokens(token_hash, session_id, created_at)"
                " VALUES (?, ?, ?)",
                (_token_hash(successor), session_id, now),
            )
            conn.execute(
                "UPDATE tokens SET rotated_at = ?, successor_hash = ?, succ_wrapped = ?"
                " WHERE token_hash = ?",
                (now, _token_hash(successor), _wrap(token, successor), token_hash),
            )
            conn.execute(
                "UPDATE sessions SET last_activity = ? WHERE session_id = ?",
                (now, session_id),
            )
            return successor

        return self._transact(fn)

    def logout(self, token: str) -> None:
        token_hash = _token_hash(token)

        def fn(conn: sqlite3.Connection):
            row = conn.execute(
                "SELECT session_id FROM tokens WHERE token_hash = ?",
                (token_hash,),
            ).fetchone()
            if row is None:
                return _Fail(REASON_UNKNOWN)
            conn.execute(
                "UPDATE sessions SET revoked = 1 WHERE session_id = ?",
                (row[0],),
            )

        self._transact(fn)

    def revoke_user(self, user_id: str) -> None:
        def fn(conn: sqlite3.Connection):
            conn.execute(
                "UPDATE sessions SET revoked = 1 WHERE user_id = ?",
                (user_id,),
            )

        self._transact(fn)
