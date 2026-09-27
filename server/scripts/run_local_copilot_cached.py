"""Start the final Gemma prototype with the measured prompt-cache setting."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time

import run_local_copilot as base

DEPLOYMENT = base.ROOT / "artifacts/local-copilot/gemma-final11-cache-v1"
REPORT_SHA = "8a3d0625ac99fffdbba5d2ff81fe96db4a5360b10bc13e747f69596b27e1ee48"
VERIFICATION_SHA = "f6c2ac3367782692edf70c04e9f04222b27f17f83cf0777a947a1cbbdfa8eb7a"


def bound_json(record):
    return json.loads(base.bound(record)[1])


def validate_deployment(deployment=DEPLOYMENT):
    original = base.validate_deployment()
    selection = base.read(deployment / "selection.json")
    base.require(
        selection["version"] == "local-gemma-measured-cache-v1", "Unknown cache selection."
    )
    base.require(selection["measurement"]["sha256"] == REPORT_SHA, "Different cache measurement.")
    base.require(
        selection["independent_verification"]["sha256"] == VERIFICATION_SHA,
        "Different independent cache verification.",
    )
    bound_json(selection["independent_verification"])
    report = bound_json(selection["measurement"])
    base.require(
        report["status"] == "completed"
        and report["selected_cache_prompt"] is True
        and report["normal_requests"] == 24
        and report["all_requests"] == 28
        and len(report["checks"]) == 5
        and all(value is True for value in report["checks"].values()),
        "The completed comparison did not select prompt reuse.",
    )
    protocol = bound_json(report["protocol"])
    for binding in protocol["source_pins"]:
        base.bound(binding)
    measured = bound_json(protocol["configs"]["on"])
    config_path, raw = base.bound(selection["configuration"])
    base.require(config_path == deployment / "gateway.json", "Different cached deployment path.")
    config = json.loads(raw)
    expected = {**original, "cache_prompt": True, "capture_runtime_evidence": False}
    base.require(config == expected, "Only prompt reuse may change the deployed model settings.")
    base.require(
        measured == {**expected, "capture_runtime_evidence": True},
        "Measured runtime differs from the cached deployment.",
    )
    launcher, _ = base.bound(protocol["bindings"]["launcher"])
    base.require(launcher == base.DEPLOYMENT / "start-model.sh", "Different model launcher.")
    return config, config_path, launcher


def start(deployment=DEPLOYMENT):
    config, config_path, launcher = validate_deployment(deployment)
    model = app = None
    with base.LOCK.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("A benchmark or export owns the shared machine lock.") from error
        base.refuse_occupied(config["endpoint"])
        env = {
            **os.environ,
            "AUDAPOLIS_NEP_CONFIG": str(config_path),
            "MODE": "development",
            "NODE_ENV": "development",
            "UV_NO_SYNC": "1",
            "UV_OFFLINE": "1",
        }
        try:
            print("Starting Gemma with prompt reuse; Ctrl+C stops this owned session.", flush=True)
            model = subprocess.Popen(
                ["sh", str(launcher)], cwd=base.ROOT, env=env, start_new_session=True
            )
            base.wait_model(model, config["endpoint"])
            base.existing_gateway_health(config_path)
            app = subprocess.Popen(
                ["npm", "start"], cwd=base.ROOT / "app", env=env, start_new_session=True
            )
            print(
                "Enable Local copilot. Tab accepts, Esc dismisses, Cmd/Ctrl+Z undoes.", flush=True
            )
            while app.poll() is None:
                base.require(model.poll() is None, "The owned Gemma process stopped.")
                time.sleep(0.5)
            base.require(app.returncode == 0, f"App exited with code {app.returncode}.")
        except KeyboardInterrupt:
            print("Stopping this cached copilot session.", flush=True)
        finally:
            try:
                base.stop_owned(app)
            finally:
                base.stop_owned(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "start"))
    args = parser.parse_args()
    try:
        if args.command == "check":
            config, path, _ = validate_deployment()
            print(
                json.dumps(
                    {
                        "status": "measured_cached_runtime_available",
                        "model_id": config["model_id"],
                        "cache_prompt": True,
                        "configuration": str(path),
                        "model_started": False,
                    }
                )
            )
        else:
            for name in (signal.SIGTERM, signal.SIGHUP):
                signal.signal(name, base.interrupted)
            start()
    except (OSError, ValueError, TimeoutError) as error:
        parser.exit(1, f"Cached local copilot: {error}\n")


if __name__ == "__main__":
    main()
