"""SQLite-backed dedup and history storage."""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_items (
  id TEXT PRIMARY KEY,
  url TEXT NOT NULL,
  source TEXT NOT NULL,
  title TEXT,
  published_at TEXT,
  first_seen TEXT NOT NULL,
  sent_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_seen_first_seen ON seen_items(first_seen);
CREATE INDEX IF NOT EXISTS idx_seen_sent_at ON seen_items(sent_at);
"""

MIGRATIONS = [
    # Add sent_at if missing (idempotent — wraps in try/except via _safe_migrate).
    "ALTER TABLE seen_items ADD COLUMN sent_at TEXT",
]


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            for m in MIGRATIONS:
                try: c.execute(m)
                except sqlite3.OperationalError: pass  # already applied

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def filter_unseen(self, items: Iterable[dict]) -> list[dict]:
        items = list(items)
        if not items:
            return []
        ids = [i["id"] for i in items]
        with self._conn() as c:
            placeholders = ",".join("?" * len(ids))
            cur = c.execute(
                f"SELECT id FROM seen_items WHERE id IN ({placeholders})", ids
            )
            seen = {row[0] for row in cur.fetchall()}
        return [i for i in items if i["id"] not in seen]

    def mark_seen(self, items: Iterable[dict], *, sent: bool = False) -> None:
        now = datetime.now(timezone.utc).isoformat()
        sent_at = now if sent else None
        rows = [
            (i["id"], i["url"], i["source"], i["title"], i.get("published_at"), now, sent_at)
            for i in items
        ]
        with self._conn() as c:
            c.executemany(
                "INSERT OR IGNORE INTO seen_items (id, url, source, title, published_at, first_seen, sent_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            # If items were previously seen (not sent), update sent_at when this is a send.
            if sent:
                ids = [i["id"] for i in items]
                if ids:
                    placeholders = ",".join("?" * len(ids))
                    c.execute(
                        f"UPDATE seen_items SET sent_at = COALESCE(sent_at, ?) WHERE id IN ({placeholders})",
                        [now] + ids,
                    )

    def recent_sent(self, days: int = 14) -> list[dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._conn() as c:
            cur = c.execute(
                "SELECT sent_at, source, title, url FROM seen_items "
                "WHERE sent_at IS NOT NULL AND sent_at >= ? ORDER BY sent_at DESC",
                (cutoff,),
            )
            return [
                {"sent_at": r[0], "source": r[1], "title": r[2], "url": r[3]}
                for r in cur.fetchall()
            ]

    def prune(self, retain_days: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retain_days)).isoformat()
        with self._conn() as c:
            cur = c.execute("DELETE FROM seen_items WHERE first_seen < ?", (cutoff,))
            return cur.rowcount
