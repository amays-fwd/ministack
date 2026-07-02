"""Named state checkpoints for MiniStack.

A checkpoint is an explicit, on-demand snapshot of all service state,
stored under ``${STATE_DIR}/checkpoints/<name>/``:

    <state_key>.json    one file per service (same format as live STATE_DIR)
    lambda-blobs/       Lambda code blobs, hardlinked from the live blob dir
    s3-data/            S3 object bodies (only when S3_PERSIST=1)
    manifest.json       {version, name, created_at, services, s3_data}

Checkpoints work regardless of PERSIST_STATE — that flag governs only the
automatic boot-restore/shutdown-save cycle. Blob files are hardlinked
(copied on cross-device failure) rather than referenced: lambda_svc's
``_prune_orphan_code_blobs`` unlinks live directory entries on every
``get_state()``, and a checkpoint must survive that. Content-addressed
sha256 filenames make hardlinks safe — the same name never means
different bytes.
"""

import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone

from ministack.core import persistence

logger = logging.getLogger("checkpoints")

MANIFEST_VERSION = 1
# Single path segment, no leading dot — blocks traversal and tmp-dir collisions.
CHECKPOINT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

_BLOB_DIR_NAME = "lambda-blobs"
_S3_DATA_DIR_NAME = "s3-data"
_MANIFEST_NAME = "manifest.json"


class CheckpointError(Exception):
    pass


class InvalidCheckpointNameError(CheckpointError):
    pass


class CheckpointExistsError(CheckpointError):
    pass


class CheckpointNotFoundError(CheckpointError):
    pass


def checkpoint_root() -> str:
    return os.path.join(persistence.STATE_DIR, "checkpoints")


def validate_name(name) -> str:
    if not isinstance(name, str) or not CHECKPOINT_NAME_RE.match(name):
        raise InvalidCheckpointNameError(
            "checkpoint name must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}"
        )
    return name


def require_checkpoint(name: str) -> str:
    """Validate ``name`` and return its directory, raising if absent."""
    validate_name(name)
    path = os.path.join(checkpoint_root(), name)
    if not os.path.isdir(path):
        raise CheckpointNotFoundError(f"checkpoint {name!r} not found")
    return path


def _link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _copy_blob_dir(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        src = os.path.join(src_dir, fname)
        if os.path.isfile(src):
            _link_or_copy(src, os.path.join(dst_dir, fname))


def _dir_size(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for fname in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, fname))
            except OSError:
                pass
    return total


def _read_manifest(cp_dir: str) -> dict | None:
    try:
        with open(os.path.join(cp_dir, _MANIFEST_NAME)) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _s3_persist_config() -> tuple[bool, str]:
    return (
        os.environ.get("S3_PERSIST", "0") == "1",
        os.environ.get("S3_DATA_DIR", "/tmp/ministack-data/s3"),
    )


def create_checkpoint(name: str, save_dict: dict, *, overwrite: bool = False) -> dict:
    """Snapshot all service state into a named checkpoint.

    ``save_dict`` is ``{state_key: get_state_fn}`` (the caller builds it from
    the loaded modules and should hold the reset lock so the snapshot is
    near-quiescent). Live ``STATE_DIR/*.json`` files whose key is not in
    ``save_dict`` are copied in as-is — they hold prior-boot state for
    modules that were never imported this run, and skipping them would
    silently drop those services from the checkpoint.

    The checkpoint is built in a temp dir and renamed into place, so a
    half-written checkpoint is never visible under its final name.
    """
    validate_name(name)
    root = checkpoint_root()
    final_dir = os.path.join(root, name)
    if os.path.isdir(final_dir) and not overwrite:
        raise CheckpointExistsError(f"checkpoint {name!r} already exists")

    os.makedirs(root, exist_ok=True)
    tmp_dir = os.path.join(root, f".tmp-{name}-{os.getpid()}")
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    try:
        services = []
        for key, get_state in save_dict.items():
            data = persistence.get_state_with_retry(key, get_state)
            if data is None:
                continue
            persistence.write_state_file(tmp_dir, key, data)
            services.append(key)

        state_dir = persistence.STATE_DIR
        if os.path.isdir(state_dir):
            for fname in sorted(os.listdir(state_dir)):
                if not fname.endswith(".json") or fname.endswith(".tmp"):
                    continue
                key = fname[: -len(".json")]
                if key in save_dict:
                    continue
                shutil.copy2(os.path.join(state_dir, fname), os.path.join(tmp_dir, fname))
                services.append(key)

        live_blobs = os.path.join(state_dir, _BLOB_DIR_NAME)
        if os.path.isdir(live_blobs):
            _copy_blob_dir(live_blobs, os.path.join(tmp_dir, _BLOB_DIR_NAME))

        s3_persist, s3_data_dir = _s3_persist_config()
        has_s3_data = False
        if s3_persist and os.path.isdir(s3_data_dir):
            shutil.copytree(s3_data_dir, os.path.join(tmp_dir, _S3_DATA_DIR_NAME))
            has_s3_data = True

        manifest = {
            "version": MANIFEST_VERSION,
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "services": sorted(services),
            "s3_data": has_s3_data,
        }
        with open(os.path.join(tmp_dir, _MANIFEST_NAME), "w") as f:
            json.dump(manifest, f)

        if os.path.isdir(final_dir):
            old_dir = os.path.join(root, f".old-{name}-{os.getpid()}")
            os.rename(final_dir, old_dir)
            os.rename(tmp_dir, final_dir)
            shutil.rmtree(old_dir, ignore_errors=True)
        else:
            os.rename(tmp_dir, final_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    manifest["size_bytes"] = _dir_size(final_dir)
    logger.info("Checkpoint %r created (%d services)", name, len(services))
    return manifest


def restore_checkpoint_files(name: str) -> list[str]:
    """Replace live STATE_DIR contents with the checkpoint's files.

    Wipes live ``*.json`` and ``lambda-blobs/`` (never ``checkpoints/``),
    copies the checkpoint back, and returns the restored state keys. The
    caller is responsible for resetting in-memory state first and applying
    each service's restore path afterward.
    """
    cp_dir = require_checkpoint(name)
    manifest = _read_manifest(cp_dir)
    state_dir = persistence.STATE_DIR
    os.makedirs(state_dir, exist_ok=True)

    for fname in os.listdir(state_dir):
        if fname.endswith(".json"):
            try:
                os.remove(os.path.join(state_dir, fname))
            except OSError as e:
                logger.warning("restore: failed to remove %s: %s", fname, e)
    live_blobs = os.path.join(state_dir, _BLOB_DIR_NAME)
    if os.path.isdir(live_blobs):
        shutil.rmtree(live_blobs, ignore_errors=True)

    restored = []
    for fname in sorted(os.listdir(cp_dir)):
        if not fname.endswith(".json") or fname == _MANIFEST_NAME:
            continue
        shutil.copy2(os.path.join(cp_dir, fname), os.path.join(state_dir, fname))
        restored.append(fname[: -len(".json")])

    cp_blobs = os.path.join(cp_dir, _BLOB_DIR_NAME)
    if os.path.isdir(cp_blobs):
        _copy_blob_dir(cp_blobs, live_blobs)

    if manifest and manifest.get("s3_data"):
        s3_persist, s3_data_dir = _s3_persist_config()
        if s3_persist:
            if os.path.isdir(s3_data_dir):
                for entry in os.listdir(s3_data_dir):
                    entry_path = os.path.join(s3_data_dir, entry)
                    try:
                        if os.path.isdir(entry_path):
                            shutil.rmtree(entry_path)
                        else:
                            os.remove(entry_path)
                    except OSError as e:
                        logger.warning("restore: failed to remove S3 data %s: %s", entry, e)
            shutil.copytree(
                os.path.join(cp_dir, _S3_DATA_DIR_NAME), s3_data_dir, dirs_exist_ok=True
            )
        else:
            logger.warning(
                "restore: checkpoint %r contains S3 object data but S3_PERSIST is "
                "not enabled — S3 object bodies were NOT restored",
                name,
            )

    logger.info("Checkpoint %r files restored (%d services)", name, len(restored))
    return restored


def list_checkpoints() -> list[dict]:
    root = checkpoint_root()
    if not os.path.isdir(root):
        return []
    out = []
    for entry in sorted(os.listdir(root)):
        if entry.startswith("."):
            continue
        cp_dir = os.path.join(root, entry)
        if not os.path.isdir(cp_dir):
            continue
        manifest = _read_manifest(cp_dir) or {"version": None, "created_at": None, "services": []}
        manifest["name"] = entry
        manifest["size_bytes"] = _dir_size(cp_dir)
        out.append(manifest)
    out.sort(key=lambda m: m.get("created_at") or "")
    return out


def delete_checkpoint(name: str) -> None:
    cp_dir = require_checkpoint(name)
    shutil.rmtree(cp_dir)
    logger.info("Checkpoint %r deleted", name)
