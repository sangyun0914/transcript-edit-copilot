"""Run the reviewed final11 Gemma deployment and Electron in one foreground session."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT / "artifacts/benchmarks/passive-actions-v1/final11-evaluation-attempt-01"
DEPLOYMENT = EVALUATION / "registration-attempt-01/deployment"
EXECUTION = EVALUATION / "run-attempt-01"
CONDITION = "verified_final_update_11_Q4_K_M"
MODEL_ID = "gemma26-actions11-q4-k-m"
CONTRACT = "passive-nep-actions-v1"
LOCK = ROOT / "artifacts/nep-passive-b1.lock"


def require(value, message):
    if not value:
        raise ValueError(message)


def bound(record):
    path = Path(record["path"])
    require(
        path.is_absolute() and path.resolve() == path and path.is_relative_to(ROOT),
        "Deployment evidence must use its original local path.",
    )
    require(
        path.is_file() and path.stat().st_size < 32 * 1024**2,
        f"Missing or oversized deployment metadata: {path}",
    )
    raw = path.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == record["sha256"], f"Deployment metadata changed: {path}"
    )
    return path, raw


def read(path):
    return json.loads(path.read_text())


def validate_deployment(deployment=DEPLOYMENT, execution=EXECUTION):
    deployment, execution = deployment.resolve(), execution.resolve()
    pipeline = read(execution / "pipeline.json")
    require(
        pipeline["status"] == "completed" and len(pipeline["arms"]) == 1,
        "Finish the registered final14 runtime check before starting the interactive app.",
    )
    arm_result = pipeline["arms"][0]
    require(
        arm_result["condition"] == CONDITION
        and arm_result["status"] == "completed"
        and arm_result["requests_completed"] == 14
        and arm_result["process_exit_code"] == 0,
        "The final14 model process has not completed and stopped.",
    )
    _, raw = bound(pipeline["protocol"])
    protocol = json.loads(raw)
    require(
        protocol["requests"] == 14
        and len(protocol["arms"]) == 1
        and protocol["arms"][0]["condition"] == CONDITION,
        "Wrong registered deployment.",
    )
    arm = protocol["arms"][0]
    for key, filename in (
        ("configuration", "gateway.json"),
        ("launcher", "start-model.sh"),
        ("parity", "native-token-parity.json"),
    ):
        path, _ = bound(arm[key])
        require(path == deployment / filename, "Use the exact evaluated deployment files.")
    config = read(deployment / "gateway.json")
    require(
        config["contract_version"] == CONTRACT and config["model_id"] == MODEL_ID,
        "Use the final11 Gemma all-seven-action deployment.",
    )
    require(
        arm["parity"]
        == {"path": config["token_parity_path"], "sha256": config["token_parity_sha256"]},
        "Gateway native-prefix evidence differs.",
    )
    health = read(execution / CONDITION / "health.json")
    require(
        health["status"] == "ready"
        and health["model"]["training_token_parity"] == "verified_fixed_bilingual_checks",
        "The evaluated native/server prefix checks have not passed.",
    )
    for model_key, config_key in (
        ("id", "model_id"),
        ("sha256", "model_sha256"),
        ("chat_template_sha256", "chat_template_sha256"),
        ("runtime_build", "runtime_build"),
        ("contract_version", "contract_version"),
        ("token_parity_sha256", "token_parity_sha256"),
    ):
        require(
            health["model"][model_key] == config[config_key], "Evaluated runtime identity differs."
        )
    endpoint = urlparse(config["endpoint"])
    require(
        endpoint.scheme == "http"
        and endpoint.hostname == "127.0.0.1"
        and endpoint.port
        and not endpoint.username
        and not endpoint.password
        and endpoint.path in ("", "/")
        and not endpoint.query
        and not endpoint.fragment,
        "Use the dedicated loopback model endpoint.",
    )
    require(Path(config["model_path"]).is_file(), "The evaluated local Gemma model is missing.")
    require(
        all(shutil.which(name) for name in ("node", "npm", "uv")),
        "Node, npm and uv must be available on PATH.",
    )
    require(
        (ROOT / "app/node_modules/electron").is_dir()
        and (ROOT / "server/.venv/bin/python").is_file(),
        "Use the already installed app and server dependencies.",
    )
    return config


def refuse_occupied(endpoint):
    address = urlparse(endpoint)
    try:
        with socket.create_connection((address.hostname, address.port), timeout=1):
            raise ValueError(
                "The model endpoint is occupied; this launcher never replaces an existing process."
            )
    except ConnectionRefusedError:
        return


def stop_owned(process):
    if process is None:
        return
    # Each child owns a new session; its Electron/uv/server descendants share that group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def wait_model(process, endpoint, timeout=180):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        require(process.poll() is None, "The owned model process exited during startup.")
        try:
            with opener.open(endpoint.rstrip("/") + "/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.25)
    raise TimeoutError("Local Gemma startup timed out.")


def existing_gateway_health(config_path):
    sys.path.insert(0, str(ROOT / "server"))
    from app.copilot import LocalConfig, LocalLlamaAdapter

    result = LocalLlamaAdapter(LocalConfig.model_validate(read(config_path))).health()
    require(
        result["status"] == "ready",
        "The existing gateway's actual model/template/prefix check failed.",
    )


def start(deployment=DEPLOYMENT, execution=EXECUTION):
    config = validate_deployment(deployment, execution)
    model = app = None
    with LOCK.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "A benchmark or export owns the shared machine lock. "
                "Finish it before launching the app."
            ) from error
        refuse_occupied(config["endpoint"])
        env = {
            **os.environ,
            "AUDAPOLIS_NEP_CONFIG": str(deployment / "gateway.json"),
            "MODE": "development",
            "NODE_ENV": "development",
            "UV_NO_SYNC": "1",
            "UV_OFFLINE": "1",
        }
        try:
            print(
                "Starting local Gemma; Ctrl+C in this terminal stops the owned app and model.",
                flush=True,
            )
            model = subprocess.Popen(
                ["sh", str(deployment / "start-model.sh")],
                cwd=ROOT,
                env=env,
                start_new_session=True,
            )
            wait_model(model, config["endpoint"])
            existing_gateway_health(deployment / "gateway.json")
            app = subprocess.Popen(
                ["npm", "start"], cwd=ROOT / "app", env=env, start_new_session=True
            )
            print(
                "Enable Local copilot (experimental) in the editor. "
                "Tab accepts, Esc dismisses, Cmd/Ctrl+Z undoes.",
                flush=True,
            )
            while app.poll() is None:
                require(
                    model.poll() is None,
                    "The owned Gemma process stopped; closing its app session.",
                )
                time.sleep(0.5)
            require(app.returncode == 0, f"The app startup exited with code {app.returncode}.")
        except KeyboardInterrupt:
            print("Stopping this local copilot session.", flush=True)
        finally:
            try:
                stop_owned(app)
            finally:
                stop_owned(model)


def interrupted(_signal, _frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "start"))
    args = parser.parse_args()
    try:
        if args.command == "check":
            config = validate_deployment()
            print(
                json.dumps(
                    {
                        "status": "evaluated_runtime_setup_available",
                        "model_id": config["model_id"],
                        "contract_version": config["contract_version"],
                        "quality_threshold_required": False,
                        "model_started": False,
                        "app_started": False,
                    },
                    indent=2,
                )
            )
        else:
            for name in (signal.SIGTERM, signal.SIGHUP):
                signal.signal(name, interrupted)
            start()
    except (OSError, ValueError, TimeoutError) as error:
        parser.exit(1, f"Local copilot: {error}\n")


if __name__ == "__main__":
    main()
