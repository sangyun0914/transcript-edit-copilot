"""Own the explicitly selected local model for an ordinary editor session."""

import hashlib
import importlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from uuid import uuid4

from .copilot import DisabledAdapter, LocalConfig, LocalLlamaAdapter

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTION = Path("artifacts/local-runtime/gemma-actions-v3/default-deployment.json")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selected_deployment(selection, root=None):
    root = ROOT if root is None else Path(root)
    selection = Path(selection)
    if not selection.is_absolute():
        selection = root / selection
    value = json.loads(selection.read_text())
    if set(value) != {"version", "deployment", "deployment_sha256"} or value["version"] != 1:
        raise ValueError("Unknown default deployment selection format.")
    deployment = (root / value["deployment"]).resolve()
    if not deployment.is_relative_to(root.resolve()):
        raise ValueError("The default local deployment must belong to this workspace.")
    if digest(deployment / "deployment.json") != value["deployment_sha256"]:
        raise ValueError("The selected deployment changed; select its verified replacement.")
    return deployment, value["deployment_sha256"]


def load_launcher():
    scripts = str(ROOT / "server/scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("run_local_copilot_actions_v3")


def default_adapter():
    testing = (
        os.environ.get("NODE_ENV") == "test"
        or "PYTEST_CURRENT_TEST" in os.environ
        or "pytest" in sys.modules
        or "unittest" in sys.modules
    )
    if testing and os.environ.get("AUDAPOLIS_NEP_AUTOSTART_TEST") != "1":
        return DisabledAdapter("Automatic local model startup is disabled in tests.")
    if os.environ.get("AUDAPOLIS_NEP_AUTOSTART") != "1":
        return DisabledAdapter("Automatic local model startup is not enabled for this server.")
    selection = os.environ.get("AUDAPOLIS_NEP_DEFAULT_DEPLOYMENT", str(DEFAULT_SELECTION))
    try:
        deployment, receipt_hash = selected_deployment(selection)
        config = LocalConfig.model_validate(json.loads((deployment / "gateway.json").read_text()))
        return ManagedLocalAdapter(config, deployment, receipt_hash)
    except (ValueError, OSError, KeyError, TypeError) as error:
        return DisabledAdapter(f"The selected local model is unavailable: {error}")


class ManagedLocalAdapter(LocalLlamaAdapter):
    process_lifecycle = "app_owned"

    def __init__(self, config, deployment, receipt_hash):
        super().__init__(config)
        self.deployment = deployment
        self.receipt_hash = receipt_hash
        self._stop = threading.Event()
        self._process_lock = threading.Lock()
        self._model_process = None
        self._thread = None
        self._launcher = None
        self._readiness = ("starting", "Checking the selected local model files.")

    def start(self):
        if self._thread is not None or self._stop.is_set():
            return
        self._thread = threading.Thread(target=self._run, name="local-copilot", daemon=True)
        self._thread.start()

    def health(self):
        status, message = self._readiness
        if status != "ready":
            return {"status": status, "message": message, "model": self.model}
        return super().health()

    def _stop_model(self):
        with self._process_lock:
            if self._model_process is not None:
                self._launcher.lifecycle.stop_owned(self._model_process)
                self._model_process = None

    def stop(self):
        self._stop.set()
        self._readiness = ("unavailable", "The local model session has stopped.")
        self._stop_model()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self):
        try:
            launcher = self._launcher = load_launcher()
            launcher.lifecycle.LOCK.parent.mkdir(parents=True, exist_ok=True)
            with launcher.lifecycle.LOCK.open("a") as lock:
                try:
                    launcher.fcntl.flock(lock, launcher.fcntl.LOCK_EX | launcher.fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ValueError(
                        "Another local model session owns the shared model lock. "
                        "Close it before restarting the editor."
                    ) from error
                try:
                    if digest(self.deployment / "deployment.json") != self.receipt_hash:
                        raise ValueError("The selected deployment changed before startup.")
                    config, verified, command, receipt = launcher.check(self.deployment)
                    if config != self.config:
                        raise ValueError("The selected gateway changed before startup.")
                    if (
                        receipt["historical_tooling_only"]
                        or receipt.get("purpose") != "ui_performance"
                    ):
                        raise ValueError("Select a current deployment prepared for the editor UI.")
                    self._verified_files = verified._verified_files
                    if self._stop.is_set():
                        return
                    launcher.lifecycle.refuse_occupied(config.endpoint)
                    logs = ROOT / "artifacts/local-runtime/gemma-actions-v3/app-sessions"
                    logs.mkdir(parents=True, exist_ok=True)
                    log_path = logs / f"{uuid4()}.log"
                    with log_path.open("x") as log:
                        with self._process_lock:
                            if self._stop.is_set():
                                return
                            model = self._model_process = subprocess.Popen(
                                command,
                                cwd=ROOT,
                                stdout=log,
                                stderr=log,
                                start_new_session=True,
                            )
                        self._readiness = ("starting", "Loading the selected local model.")
                        launcher.lifecycle.wait_model(model, config.endpoint)
                        if self._stop.is_set():
                            return
                        health = super().health()
                        if health["status"] != "ready":
                            raise ValueError(
                                "Loaded model, adapter or native token verification failed."
                            )
                        if self._stop.is_set():
                            return
                        self._readiness = ("ready", "The local model is ready.")
                        while not self._stop.wait(0.25):
                            if model.poll() is not None:
                                raise ValueError(
                                    f"The local model stopped. Runtime log: {log_path}"
                                )
                finally:
                    self._stop_model()
        except Exception as error:
            if not self._stop.is_set():
                self._readiness = ("unavailable", f"Local model startup failed: {error}")
