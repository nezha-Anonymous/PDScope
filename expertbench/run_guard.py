"""Cooperating CLI runs serialize by canonical results root and host.

The kernel lock is authoritative; its JSON payload is advisory provenance.
Do not unlink the lock file: replacing the inode would permit two owners.
This does not exclude external jobs or direct benchmark function calls.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket


class ExperimentBusy(RuntimeError):
    def __init__(self, record):
        self.record = record
        owner = record.get('owner', {})
        super().__init__(
            f"Experiment lock busy: run={owner.get('run_id', 'unknown')} "
            f"pid={owner.get('pid', 'unknown')} host={record['host']}; "
            "wait for the owning experiment to exit before retrying")


@contextmanager
def experiment_lease(output_root, *, run_id, kind, host=None):
    """Fail immediately on contention, including recursive same-process use."""
    root = Path(output_root).resolve()
    host = socket.gethostname() if host is None else host
    shared = root / '_shared'
    shared.mkdir(parents=True, exist_ok=True)
    path = shared / ('experiment_' + hashlib.sha256(host.encode()).hexdigest()[:24] + '.lock')
    record = dict(protocol='host_results_flock_v1', host=host, output_root=str(root),
                  lock_path=str(path), scope='cooperating_CLI_only', status='acquired')
    # a+ preserves metadata until flock succeeds; closing releases on any error.
    with path.open('a+', encoding='utf-8') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.seek(0)
            try:
                owner = json.loads(stream.read(16384))
                if not isinstance(owner, dict):
                    owner = {}
            except (ValueError, UnicodeError):
                owner = {}
            raise ExperimentBusy(dict(record, status='blocked', owner=owner)) from exc
        owner = dict(run_id=run_id, kind=kind, pid=os.getpid(), host=host,
                     acquired_at=datetime.now(timezone.utc).isoformat())
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps(owner) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
        yield dict(record, owner=owner)
        # Closing the descriptor releases even when the body raises or exits.
