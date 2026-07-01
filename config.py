"""Shared configuration and crash-safe file helpers.

All persistent state (memory.json, soul.md, conversation store, session
snapshots, output.txt) lives inside a single data directory so that the
Docker deployment can mount ONE volume at that directory. Docker named
volumes are directories; mounting one over an individual file (the old
behaviour) silently turned that file into a directory and broke every
read/write. Routing everything through AGENT_DATA_DIR avoids that entirely.

AGENT_DATA_DIR defaults to a ``data`` directory next to this module for local
runs and is set to ``/app/data`` inside the container image.
"""
from __future__ import annotations
import json
import os
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.environ.get('AGENT_DATA_DIR') or os.path.join(_HERE, 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def data_path(name: str) -> str:
    """Return an absolute path for a state file inside the data directory."""
    return os.path.join(DATA_DIR, name)


# Canonical locations for all persistent state.
MEMORY_FILE = data_path('memory.json')
SOUL_FILE = data_path('soul.md')
CONVERSATION_FILE = data_path('conversation.json')
OUTPUT_FILE = data_path('output.txt')


def atomic_write(path: str, text: str) -> None:
    """Write text to path atomically (temp file + os.replace).

    A crash or exception mid-write cannot truncate or corrupt the target:
    the original file stays intact until os.replace swaps in the fully
    written temp file. Raises OSError on failure -- callers decide whether
    to surface that as an error string or a warning.
    """
    directory = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: str, obj) -> None:
    """Serialize obj to JSON and write it atomically."""
    atomic_write(path, json.dumps(obj, indent=2, ensure_ascii=False))
