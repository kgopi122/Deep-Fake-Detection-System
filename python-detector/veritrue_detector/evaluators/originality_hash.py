from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from .base import Evaluator
from ..types import EvaluatorResult, Signal
from ..utils.phash import phash64


class OriginalityHashEvaluator(Evaluator):
    name = "Originality (pHash)"

    def __init__(self, db_path: str | None):
        self._db_path = db_path

    def evaluate(self, image_rgb: np.ndarray, image_path: str) -> EvaluatorResult:
        if not self._db_path:
            return EvaluatorResult(
                evaluator=self.name,
                score_0_100=50.0,
                signals=[
                    Signal(
                        name="phash",
                        score_0_100=50.0,
                        details={"status": "disabled (no db configured)"},
                    )
                ],
            )

        h = phash64(image_rgb)
        self._init_db()
        nearest = self._nearest(h)

        # Interpretation:
        # - If we find close matches (low Hamming distance), it’s likely repost/derivative.
        # - If we find no close matches, content may be unique (which slightly increases AI risk).
        score = 50.0
        if nearest is None:
            score = 58.0
        else:
            dist = nearest["distance"]
            if dist <= 4:
                score = 45.0
            elif dist <= 8:
                score = 48.0
            else:
                score = 55.0

        self._upsert(image_path, h)

        return EvaluatorResult(
            evaluator=self.name,
            score_0_100=float(score),
            signals=[
                Signal(
                    name="phash",
                    score_0_100=float(score),
                    details={"phash": int(h), "nearest": nearest},
                )
            ],
        )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS hashes (
                  path TEXT PRIMARY KEY,
                  phash INTEGER NOT NULL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_hashes_phash ON hashes(phash)")

    def _upsert(self, path: str, phash: int) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO hashes(path, phash) VALUES(?, ?) ON CONFLICT(path) DO UPDATE SET phash=excluded.phash",
                (path, int(phash)),
            )

    def _nearest(self, target: int) -> dict[str, Any] | None:
        # Simple brute scan (OK for small/moderate DB). Replace with BK-tree/FAISS for large scale.
        with self._connect() as con:
            rows = con.execute("SELECT path, phash FROM hashes LIMIT 5000").fetchall()

        best: tuple[int, str] | None = None
        for path, ph in rows:
            dist = _hamming64(int(target), int(ph))
            if best is None or dist < best[0]:
                best = (dist, str(path))

        if best is None:
            return None
        return {"distance": int(best[0]), "path": best[1]}


def _hamming64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())
