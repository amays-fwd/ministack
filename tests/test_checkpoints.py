"""
Tests for named state checkpoints (/_ministack/state/*), the on-demand save
endpoint, and the durability hardening (autosave loop, atexit final save).

Checkpoint semantics under test:
  - checkpoints work regardless of PERSIST_STATE (the env gate governs only
    the automatic boot-restore/shutdown-save cycle);
  - checkpoints own copies of lambda code blobs, so lambda_svc's
    per-get_state() orphan prune can never corrupt them;
  - hot restore resets in-memory state before merging the checkpoint back
    (restore paths .update(), they don't replace);
  - restoring a nonexistent checkpoint fails BEFORE wiping live state.
"""
import asyncio
import contextlib
import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import time

import pytest

import ministack.app as app
from ministack.core import checkpoints, persistence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    # Default the gate OFF — checkpoints must work without it. Tests that
    # exercise the gate interaction flip it on themselves.
    monkeypatch.setattr(persistence, "PERSIST_STATE", False)
    monkeypatch.setattr(persistence, "STATE_DIR", str(tmp_path))
    # Fresh reset lock per test: asyncio.Lock binds to the event loop that
    # first acquires it, and each asyncio.run() here creates a new loop.
    monkeypatch.setattr(app, "_reset_lock", None)


def _module(mod_name):
    return importlib.import_module(f"ministack.services.{mod_name}")


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "../x", "a/b", ".hidden", "a" * 65, None, "a b", "-lead"])
def test_validate_name_rejects(bad):
    with pytest.raises(checkpoints.InvalidCheckpointNameError):
        checkpoints.validate_name(bad)


@pytest.mark.parametrize("good", ["a", "before-test", "v1.2_rc", "A" * 64])
def test_validate_name_accepts(good):
    assert checkpoints.validate_name(good) == good


# ---------------------------------------------------------------------------
# Checkpoint CRUD
# ---------------------------------------------------------------------------


def test_create_list_overwrite_delete():
    save_dict = {"autoscaling": lambda: {"launch_configs": {"lc": {"x": 1}}}}
    manifest = checkpoints.create_checkpoint("cp1", save_dict)
    assert manifest["name"] == "cp1"
    assert manifest["services"] == ["autoscaling"]
    assert manifest["s3_data"] is False
    assert manifest["size_bytes"] > 0
    assert manifest["created_at"]

    listed = checkpoints.list_checkpoints()
    assert [c["name"] for c in listed] == ["cp1"]
    assert listed[0]["services"] == ["autoscaling"]

    with pytest.raises(checkpoints.CheckpointExistsError):
        checkpoints.create_checkpoint("cp1", save_dict)

    manifest2 = checkpoints.create_checkpoint(
        "cp1", {"eks": lambda: {"clusters": {}}}, overwrite=True
    )
    assert manifest2["services"] == ["eks"]

    checkpoints.delete_checkpoint("cp1")
    assert checkpoints.list_checkpoints() == []
    with pytest.raises(checkpoints.CheckpointNotFoundError):
        checkpoints.delete_checkpoint("cp1")


def test_create_survives_failing_get_state():
    def boom():
        raise ValueError("store exploded")

    manifest = checkpoints.create_checkpoint(
        "cp-partial", {"bad": boom, "good": lambda: {"k": 1}}
    )
    assert manifest["services"] == ["good"]


def test_checkpoint_includes_unloaded_module_state_files():
    """A module never imported this run has no get_state in the save dict,
    but its prior-boot state file in live STATE_DIR must still be captured."""
    persistence.write_state_file(persistence.STATE_DIR, "waf", {"acls": {"a": 1}})
    manifest = checkpoints.create_checkpoint("cp-fallback", {"eks": lambda: {"clusters": {}}})
    assert set(manifest["services"]) == {"eks", "waf"}
    cp_dir = os.path.join(checkpoints.checkpoint_root(), "cp-fallback")
    assert persistence.read_state_file(cp_dir, "waf") == {"acls": {"a": 1}}


# ---------------------------------------------------------------------------
# Lambda code blobs
# ---------------------------------------------------------------------------


def test_checkpoint_blobs_survive_live_prune():
    """Simulates _prune_orphan_code_blobs deleting a live blob after the
    checkpoint was taken — the checkpoint's hardlinked copy must survive
    and restore must bring the live file back."""
    blob_dir = os.path.join(persistence.STATE_DIR, "lambda-blobs")
    os.makedirs(blob_dir)
    blob = os.path.join(blob_dir, "a" * 64 + ".zip")
    with open(blob, "wb") as f:
        f.write(b"zipbytes")

    checkpoints.create_checkpoint("cp-blob", {})
    os.remove(blob)  # what the prune does to unreferenced live blobs

    checkpoints.restore_checkpoint_files("cp-blob")
    with open(blob, "rb") as f:
        assert f.read() == b"zipbytes"


def test_lambda_code_round_trips_through_checkpoint(monkeypatch):
    """End-to-end through the real lambda_svc externalize/prune/internalize
    path: function code must come back as bytes after a hot restore, even
    when the live blob was pruned in between."""
    lam = _module("lambda_svc")
    monkeypatch.setattr(
        lam, "CODE_BLOB_DIR", os.path.join(persistence.STATE_DIR, "lambda-blobs")
    )
    lam.reset()
    code = b"PK\x03\x04 fake zip bytes"
    lam._functions["fn-cp"] = {"FunctionName": "fn-cp", "code_zip": code, "versions": {}}
    try:
        app._get_module("lambda_svc")  # register for _build_persistence_save_dict
        checkpoints.create_checkpoint("cp-lam", app._build_persistence_save_dict())

        # Delete the function and trigger the orphan prune.
        del lam._functions["fn-cp"]
        lam.get_state()

        app._restore_checkpoint_locked("cp-lam")
        assert lam._functions["fn-cp"]["code_zip"] == code
    finally:
        lam.reset()


# ---------------------------------------------------------------------------
# Hot restore
# ---------------------------------------------------------------------------


def test_hot_restore_round_trip():
    auto = _module("autoscaling")
    auto.reset()
    auto._launch_configs["lc-1"] = {"LaunchConfigurationName": "lc-1"}
    try:
        checkpoints.create_checkpoint("cp-hot", {"autoscaling": auto.get_state})

        # Mutate after the checkpoint — restore must return to the snapshot.
        auto._launch_configs["lc-2"] = {"LaunchConfigurationName": "lc-2"}

        applied = app._restore_checkpoint_locked("cp-hot")
        assert "autoscaling" in applied
        assert "lc-1" in auto._launch_configs
        assert "lc-2" not in auto._launch_configs
    finally:
        auto.reset()


def test_hot_restore_works_without_persist_state():
    assert persistence.PERSIST_STATE is False  # fixture default
    eks = _module("eks")
    eks.reset()
    eks._clusters["c1"] = {"name": "c1", "status": "ACTIVE"}
    try:
        checkpoints.create_checkpoint("cp-nogate", {"eks": eks.get_state})
        eks.reset()
        applied = app._restore_checkpoint_locked("cp-nogate")
        assert "eks" in applied
        assert "c1" in eks._clusters
    finally:
        eks.reset()


def test_restore_unknown_checkpoint_leaves_state_untouched():
    auto = _module("autoscaling")
    auto.reset()
    auto._launch_configs["keep-me"] = {"x": 1}
    try:
        with pytest.raises(checkpoints.CheckpointNotFoundError):
            app._restore_checkpoint_locked("does-not-exist")
        assert "keep-me" in auto._launch_configs
    finally:
        auto.reset()


def test_restore_flips_running_stepfunctions_execution_to_failed():
    sfn = _module("stepfunctions")
    sfn.reset()
    exec_arn = "arn:aws:states:us-east-1:000000000000:execution:sm:e1"
    sfn._executions[exec_arn] = {
        "executionArn": exec_arn,
        "stateMachineArn": "arn:aws:states:us-east-1:000000000000:stateMachine:sm",
        "status": "RUNNING",
        "startDate": 0,
        "input": "{}",
    }
    try:
        checkpoints.create_checkpoint("cp-sfn", {"stepfunctions": sfn.get_state})
        app._restore_checkpoint_locked("cp-sfn")
        restored = sfn._executions[exec_arn]
        assert restored["status"] == "FAILED"
        assert restored.get("error") == "States.ServiceRestart"
    finally:
        sfn.reset()


# ---------------------------------------------------------------------------
# /_ministack/reset interaction
# ---------------------------------------------------------------------------


def test_reset_removes_stale_lambda_blobs():
    """With lambda.json wiped by reset, every blob is an orphan — reset must
    remove them or they leak until a Lambda save's orphan prune, which never
    runs if Lambda isn't used again."""
    blob_dir = os.path.join(persistence.STATE_DIR, "lambda-blobs")
    os.makedirs(blob_dir)
    with open(os.path.join(blob_dir, "b" * 64 + ".zip"), "wb") as f:
        f.write(b"orphan")

    app._reset_all_state()
    assert not os.path.isdir(blob_dir), "reset left stale lambda-blobs behind"


def test_reset_wipes_state_files_even_without_gate():
    """/state/save writes files with PERSIST_STATE off; reset must wipe them
    regardless of the gate, or a later checkpoint's unloaded-module fallback
    resurrects pre-reset state."""
    assert persistence.PERSIST_STATE is False  # fixture default
    persistence.write_state_file(persistence.STATE_DIR, "waf", {"acls": {"stale": 1}})

    app._reset_all_state()
    assert not os.path.exists(os.path.join(persistence.STATE_DIR, "waf.json"))

    manifest = checkpoints.create_checkpoint("post-reset", {})
    assert "waf" not in manifest["services"], (
        "checkpoint resurrected pre-reset state from a file reset left behind"
    )


def test_checkpoints_survive_reset():
    """Named checkpoints are user snapshots — snapshot → reset → restore is a
    primary workflow, so reset must not delete them."""
    eks = _module("eks")
    eks.reset()
    eks._clusters["c-reset"] = {"name": "c-reset"}
    try:
        checkpoints.create_checkpoint("pre-reset", {"eks": eks.get_state})

        app._reset_all_state()
        assert [c["name"] for c in checkpoints.list_checkpoints()] == ["pre-reset"]
        assert "c-reset" not in eks._clusters  # reset really wiped live state

        applied = app._restore_checkpoint_locked("pre-reset")
        assert "eks" in applied
        assert "c-reset" in eks._clusters
    finally:
        eks.reset()


# ---------------------------------------------------------------------------
# S3 object data
# ---------------------------------------------------------------------------


def test_checkpoint_captures_and_restores_s3_data(monkeypatch, tmp_path):
    s3_dir = tmp_path / "s3data"
    (s3_dir / "bucket").mkdir(parents=True)
    (s3_dir / "bucket" / "obj").write_bytes(b"body-bytes")
    monkeypatch.setenv("S3_PERSIST", "1")
    monkeypatch.setenv("S3_DATA_DIR", str(s3_dir))

    manifest = checkpoints.create_checkpoint("cp-s3", {})
    assert manifest["s3_data"] is True

    (s3_dir / "bucket" / "obj").unlink()
    checkpoints.restore_checkpoint_files("cp-s3")
    assert (s3_dir / "bucket" / "obj").read_bytes() == b"body-bytes"


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------


def _call(method, path, body=b""):
    return asyncio.run(app._handle_admin_state_request(method, path, body))


def test_state_endpoints_full_flow():
    auto = _module("autoscaling")
    auto.reset()
    auto._launch_configs["lc-ep"] = {"x": 1}
    app._get_module("autoscaling")  # register for _build_persistence_save_dict

    async def flow():
        results = {}
        results["create"] = await app._handle_admin_state_request(
            "POST", "/_ministack/state/checkpoint", json.dumps({"name": "ep1"}).encode()
        )
        results["dup"] = await app._handle_admin_state_request(
            "POST", "/_ministack/state/checkpoint", b'{"name": "ep1"}'
        )
        results["list"] = await app._handle_admin_state_request(
            "GET", "/_ministack/state/checkpoints", b""
        )
        auto._launch_configs.pop("lc-ep")
        results["restore"] = await app._handle_admin_state_request(
            "POST", "/_ministack/state/restore", b'{"name": "ep1"}'
        )
        results["restore_missing"] = await app._handle_admin_state_request(
            "POST", "/_ministack/state/restore", b'{"name": "missing"}'
        )
        results["bad_name"] = await app._handle_admin_state_request(
            "POST", "/_ministack/state/checkpoint", b'{"name": "../evil"}'
        )
        results["delete"] = await app._handle_admin_state_request(
            "DELETE", "/_ministack/state/checkpoints/ep1", b""
        )
        results["delete_again"] = await app._handle_admin_state_request(
            "DELETE", "/_ministack/state/checkpoints/ep1", b""
        )
        results["unknown"] = await app._handle_admin_state_request(
            "GET", "/_ministack/other", b""
        )
        return results

    try:
        r = asyncio.run(flow())
        status, _, body = r["create"]
        assert status == 200
        assert "autoscaling" in json.loads(body)["services"]
        assert r["dup"][0] == 409
        status, _, body = r["list"]
        assert status == 200
        assert [c["name"] for c in json.loads(body)["checkpoints"]] == ["ep1"]
        assert r["restore"][0] == 200
        assert "lc-ep" in auto._launch_configs, "hot restore did not bring state back"
        assert r["restore_missing"][0] == 404
        assert r["bad_name"][0] == 400
        assert r["delete"][0] == 200
        assert r["delete_again"][0] == 404
        assert r["unknown"] is None
    finally:
        auto.reset()


def test_state_save_endpoint_without_gate(monkeypatch):
    eks = _module("eks")
    eks.reset()
    eks._clusters["c-save"] = {"name": "c-save"}
    app._get_module("eks")
    try:
        status, _, body = _call("POST", "/_ministack/state/save")
        assert status == 200
        payload = json.loads(body)
        assert payload["saved"] is True
        assert payload["autoload"] is False
        assert "note" in payload
        # File written to STATE_DIR even with the gate off — explicit action.
        data = persistence.read_state_file(persistence.STATE_DIR, "eks")
        assert data is not None and "c-save" in data.get("clusters", {})
    finally:
        eks.reset()


def test_state_save_endpoint_with_gate(monkeypatch):
    monkeypatch.setattr(persistence, "PERSIST_STATE", True)
    status, _, body = _call("POST", "/_ministack/state/save")
    assert status == 200
    payload = json.loads(body)
    assert payload["autoload"] is True
    assert "note" not in payload


# ---------------------------------------------------------------------------
# save_all_resilient
# ---------------------------------------------------------------------------


def test_save_all_resilient_retries_runtime_error():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("dictionary changed size during iteration")
        return {"ok": 1}

    assert persistence.save_all_resilient({"svc": flaky}) == 1
    assert attempts["n"] == 2
    assert persistence.read_state_file(persistence.STATE_DIR, "svc") == {"ok": 1}


def test_save_all_resilient_gives_up_after_retries():
    def always_racing():
        raise RuntimeError("dictionary changed size during iteration")

    assert persistence.save_all_resilient({"svc": always_racing}) == 0
    assert persistence.read_state_file(persistence.STATE_DIR, "svc") is None


# ---------------------------------------------------------------------------
# Autosave
# ---------------------------------------------------------------------------


def test_persist_interval_parsing(monkeypatch):
    monkeypatch.setenv("PERSIST_INTERVAL", "2.5")
    assert app._persist_interval() == 2.5
    monkeypatch.setenv("PERSIST_INTERVAL", "bogus")
    assert app._persist_interval() == 0.0
    monkeypatch.delenv("PERSIST_INTERVAL")
    assert app._persist_interval() == 0.0


@pytest.mark.parametrize(
    "gate,interval", [(False, "1"), (True, "0"), (False, "0"), (True, "")]
)
def test_autosave_not_started_when_disabled(monkeypatch, gate, interval):
    monkeypatch.setattr(persistence, "PERSIST_STATE", gate)
    monkeypatch.setenv("PERSIST_INTERVAL", interval)
    monkeypatch.setattr(app, "_autosave_task", None)

    async def run():
        app._start_autosave_task()
        assert app._autosave_task is None

    asyncio.run(run())


def test_autosave_loop_saves_periodically(monkeypatch):
    calls = []
    monkeypatch.setattr(
        persistence, "save_all_resilient", lambda d: calls.append(d) or len(d)
    )

    async def run():
        task = asyncio.create_task(app._autosave_loop(0.01))
        for _ in range(200):
            await asyncio.sleep(0.01)
            if calls:
                break
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert calls, "autosave loop never invoked save_all_resilient"


# ---------------------------------------------------------------------------
# atexit final save
# ---------------------------------------------------------------------------


def test_final_save_runs_once(monkeypatch):
    monkeypatch.setattr(persistence, "PERSIST_STATE", True)
    monkeypatch.setattr(app, "_final_save_done", False)
    calls = []
    monkeypatch.setattr(app, "save_all", lambda d: calls.append(d))

    app._final_save()
    assert len(calls) == 1
    assert app._final_save_done is True
    app._final_save()
    assert len(calls) == 1, "atexit save must not run twice"


def test_final_save_noop_without_gate(monkeypatch):
    monkeypatch.setattr(persistence, "PERSIST_STATE", False)
    monkeypatch.setattr(app, "_final_save_done", False)
    calls = []
    monkeypatch.setattr(app, "save_all", lambda d: calls.append(d))

    app._final_save()
    assert not calls
    assert app._final_save_done is False


# ---------------------------------------------------------------------------
# SIGTERM integration: docker-stop shaped shutdown must flush state
# ---------------------------------------------------------------------------


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.serial
def test_sigterm_flushes_state(tmp_path):
    """`python -m ministack` installs a SIGTERM handler that calls
    sys.exit(0), which may bypass the ASGI lifespan shutdown save. The
    atexit safety net must still flush state to STATE_DIR."""
    port = _free_port()
    state_dir = tmp_path / "state"
    env = {
        **os.environ,
        "GATEWAY_PORT": str(port),
        "PERSIST_STATE": "1",
        "STATE_DIR": str(state_dir),
        "LOG_LEVEL": "WARNING",
        "SFTP_ENABLED": "0",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "ministack"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        cwd=REPO_ROOT,
    )
    try:
        import urllib.request

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/_ministack/health", timeout=1)
                break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail("server did not become healthy")

        import boto3

        sqs = boto3.client(
            "sqs",
            endpoint_url=f"http://127.0.0.1:{port}",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        sqs.create_queue(QueueName="sigterm-test-queue")

        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    sqs_file = state_dir / "sqs.json"
    assert sqs_file.exists(), (
        "SIGTERM shutdown did not persist SQS state — neither the lifespan "
        "shutdown save nor the atexit safety net ran."
    )
    assert "sigterm-test-queue" in sqs_file.read_text()
