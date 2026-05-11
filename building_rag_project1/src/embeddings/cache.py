import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class EmbeddingCache:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    content_hash TEXT PRIMARY KEY,
                    vector_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get_many(self, content_hashes: Iterable[str]) -> Dict[str, List[float]]:
        hashes = list({hash_value for hash_value in content_hashes if hash_value})
        if not hashes:
            return {}

        placeholders = ",".join("?" for _ in hashes)
        query = f"SELECT content_hash, vector_json FROM embedding_cache WHERE content_hash IN ({placeholders})"

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, hashes).fetchall()

        results: Dict[str, List[float]] = {}
        for content_hash, vector_json in rows:
            try:
                results[content_hash] = json.loads(vector_json)
            except json.JSONDecodeError:
                continue
        return results

    def set_many(self, vectors: Dict[str, List[float]]) -> None:
        if not vectors:
            return

        rows = [(content_hash, json.dumps(vector)) for content_hash, vector in vectors.items()]
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache (content_hash, vector_json) VALUES (?, ?)",
                rows,
            )
            conn.commit()
