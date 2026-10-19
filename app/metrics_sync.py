"""Keeps data/metrics.db across Space restarts by mirroring it to a private HF Dataset.

A free Space's disk is wiped on every restart, so: download the last snapshot at boot,
upload a consistent copy every few minutes when the file changed, and once more on
shutdown. Off unless METRICS_REPO and HF_TOKEN are set, so local dev is untouched.
"""

import logging
import shutil
import sqlite3
import tempfile
import threading
from contextlib import closing
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from config import HF_TOKEN, METRICS_DB, METRICS_REPO, METRICS_SYNC_MINUTES

FILENAME = "metrics.db"
log = logging.getLogger(__name__)


class MetricsSync:
    def __init__(self, repo: str = METRICS_REPO, token: str = HF_TOKEN, db: Path = METRICS_DB,
                 minutes: float = METRICS_SYNC_MINUTES):
        self.repo = repo
        self.db = db
        self.interval = minutes * 60
        self.enabled = bool(repo and token)
        self.api = HfApi(token=token) if self.enabled else None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last: tuple[int, int] | None = None

    def restore(self) -> None:
        """Pull the last snapshot. A local file wins, since it is at least as new."""
        self.api.create_repo(self.repo, repo_type="dataset", private=True, exist_ok=True)
        if self.db.exists():
            return
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = hf_hub_download(self.repo, FILENAME, repo_type="dataset",
                                       token=self.api.token, local_dir=tmp)
                self.db.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, self.db)
            log.info("metrics restored from %s", self.repo)
        except EntryNotFoundError:
            log.info("no metrics snapshot in %s yet", self.repo)
        self._last = self._fingerprint()

    def _fingerprint(self) -> tuple[int, int] | None:
        if not self.db.exists():
            return None
        st = self.db.stat()
        return st.st_mtime_ns, st.st_size

    def push(self) -> bool:
        fp = self._fingerprint()
        if fp is None or fp == self._last:
            return False
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / FILENAME
            # The backup API copies a consistent state even if a request is mid-write.
            with closing(sqlite3.connect(self.db)) as src, closing(sqlite3.connect(snapshot)) as dst:
                src.backup(dst)
            self.api.upload_file(path_or_fileobj=str(snapshot), path_in_repo=FILENAME,
                                 repo_id=self.repo, repo_type="dataset", commit_message="Sync metrics")
        self._last = fp
        return True

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.push()
            except Exception:
                log.exception("metrics sync failed")

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            self.restore()
        except Exception:
            # Pushing after a failed restore would overwrite the history with an empty file.
            log.exception("metrics restore failed, sync disabled")
            self.enabled = False
            return
        self._thread = threading.Thread(target=self._loop, name="metrics-sync", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        try:
            self.push()
        except Exception:
            log.exception("final metrics sync failed")
