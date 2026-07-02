"""
State persistence for MiniStack services.
When PERSIST_STATE=1, service state is saved to STATE_DIR on shutdown
and reloaded on startup.
"""

import ast
import json
import logging
import os
import tempfile

from ministack.core.responses import AccountScopedDict

logger = logging.getLogger("persistence")

PERSIST_STATE = os.environ.get("PERSIST_STATE", "0") == "1"
STATE_DIR = os.environ.get("STATE_DIR", "/tmp/ministack-state")


def _json_default(obj):
    """JSON encoder fallback for AccountScopedDict, tuple keys, and bytes.

    Historically, several S3 (and other service) stores held raw request
    bodies as ``bytes``. ``json.dump`` raised ``TypeError`` and
    ``save_state`` silently swallowed the error, leaving ``${service}.json``
    absent on disk (issue #422). Bytes are now serialized as base64 inside a
    tagged dict so round-trip fidelity is preserved even for non-UTF-8
    payloads."""
    if isinstance(obj, AccountScopedDict):
        # Serialize all accounts' data with string keys
        result = {}
        for k, v in obj._data.items():
            # k is (account_id, original_key) tuple
            result[f"{k[0]}\x00{k[1]!r}"] = v
        return {"__scoped__": True, "data": result}
    if isinstance(obj, (bytes, bytearray)):
        import base64
        return {"__bytes__": base64.b64encode(bytes(obj)).decode("ascii")}
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _json_object_hook(obj):
    """JSON decoder hook to restore AccountScopedDict and bytes from serialized form."""
    if obj.get("__scoped__"):
        asd = AccountScopedDict()
        for k, v in obj["data"].items():
            account_id, key_repr = k.split("\x00", 1)
            # Restore the original key (was serialized with repr())
            try:
                original_key = ast.literal_eval(key_repr)
            except (ValueError, SyntaxError):
                original_key = key_repr
            asd._data[(account_id, original_key)] = v
        return asd
    if "__bytes__" in obj:
        import base64
        return base64.b64decode(obj["__bytes__"])
    return obj


def save_state(service: str, data: dict) -> None:
    if not PERSIST_STATE:
        return
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        path = os.path.join(STATE_DIR, f"{service}.json")
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, default=_json_default)
            os.replace(tmp, path)
        except BaseException:
            # Clean up temp file on any failure to avoid stale partial writes
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        logger.info("Persistence: saved %s state to %s", service, path)
    except Exception as e:
        logger.error("Persistence: failed to save %s: %s", service, e)


def load_state(service: str) -> dict | None:
    if not PERSIST_STATE:
        return None
    path = os.path.join(STATE_DIR, f"{service}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f, object_hook=_json_object_hook)
        logger.info("Persistence: loaded %s state from %s", service, path)
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Persistence: failed to load %s: %s", service, e)
        return None


def save_all(services: dict) -> None:
    """Save all service states. services = {name: get_state_fn}"""
    for name, get_state in services.items():
        try:
            save_state(name, get_state())
        except Exception as e:
            logger.error("Persistence: error getting state for %s: %s", name, e)


# ---------------------------------------------------------------------------
# Ungated, path-parameterized primitives. Unlike save_state/load_state these
# do NOT check PERSIST_STATE — they back explicit, on-demand operations
# (admin save endpoint, named checkpoints) where the user has already asked
# for the write. STATE_DIR is intentionally not baked in so checkpoints can
# write the same file format into their own directories.
# ---------------------------------------------------------------------------


def write_state_file(dir_path: str, service: str, data: dict) -> str:
    """Atomically write ``data`` to ``<dir_path>/<service>.json``. Raises on failure."""
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, f"{service}.json")
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, default=_json_default)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


def read_state_file(dir_path: str, service: str) -> dict | None:
    """Read ``<dir_path>/<service>.json``; None if absent or unreadable."""
    path = os.path.join(dir_path, f"{service}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f, object_hook=_json_object_hook)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Persistence: failed to read %s: %s", path, e)
        return None


_GET_STATE_ATTEMPTS = 3


def get_state_with_retry(name: str, get_state):
    """Call ``get_state()``, retrying on RuntimeError.

    get_state() deep-copies live stores; a request mutating a store
    mid-copy raises RuntimeError ("dictionary changed size during
    iteration"). The stores are consistent between requests, so an
    immediate retry almost always succeeds. Returns None when state
    could not be captured."""
    for attempt in range(_GET_STATE_ATTEMPTS):
        try:
            return get_state()
        except RuntimeError as e:
            if attempt == _GET_STATE_ATTEMPTS - 1:
                logger.error(
                    "Persistence: error getting state for %s after %d attempts: %s",
                    name, _GET_STATE_ATTEMPTS, e,
                )
        except Exception as e:
            logger.error("Persistence: error getting state for %s: %s", name, e)
            return None
    return None


def save_all_resilient(services: dict) -> int:
    """Save all service states to STATE_DIR regardless of PERSIST_STATE,
    retrying snapshots that race with in-flight requests. Backs the
    explicit /_ministack/state/save endpoint and the autosave loop.
    Returns the number of services saved."""
    saved = 0
    for name, get_state in services.items():
        data = get_state_with_retry(name, get_state)
        if data is None:
            continue
        try:
            path = write_state_file(STATE_DIR, name, data)
        except Exception as e:
            logger.error("Persistence: failed to save %s: %s", name, e)
            continue
        logger.info("Persistence: saved %s state to %s", name, path)
        saved += 1
    return saved
