"""Test-only registered UI fixtures and input guard; never supplies model outputs."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
import passive_actions as contract  # noqa: E402
import passive_formats_actions_v3 as formats  # noqa: E402

TOKEN_OPTIONS = {
    "tokenize": True,
    "return_dict": False,
    "add_generation_prompt": True,
    "enable_thinking": False,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def value_sha(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def ordered_sha(value):
    return hashlib.sha256(formats.compact(value).encode()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def session_document(document):
    """Match production app sessionDocument, keeping the original fixture separately."""
    return {
        "content": copy.deepcopy(document["content"]),
        "metadata": {
            key: copy.deepcopy(document["metadata"][key])
            for key in ("transcription", "transcriptions")
            if document["metadata"].get(key)
        },
    }


def projected_transition(transition):
    projected = copy.deepcopy(transition)
    for key in ("before", "after"):
        projected[key] = session_document(projected[key])
    return contract.CommittedTransition.model_validate(projected).model_dump()


def fixture_state(case, document_id):
    runtime = copy.deepcopy(case["runtime"])
    runtime["history"] = [projected_transition(t) for t in runtime["history"]]
    ledger = runtime.get("feedback_journal")
    if ledger:
        contract.PassiveFeedbackLedger.restore(ledger)
        ledger["document_id"] = document_id
        for record in ledger["records"]:
            if record["kind"] == "propose":
                record["document"] = session_document(record["document"])
                record["bundle"]["document_hash"] = contract.session_hash(record["document"])
            elif record.get("transition"):
                record["transition"] = projected_transition(record["transition"])
        contract.PassiveFeedbackLedger.restore(ledger)
    projected = session_document(case["document"])
    require(
        runtime["caption_profile"] == contract.CaptionProfile().model_dump(),
        "The production UI does not expose a custom caption profile.",
    )
    return {
        "case_id": case["id"],
        "document_id": document_id,
        "document": copy.deepcopy(case["document"]),
        "session_document": projected,
        "runtime": runtime,
        "prediction_sequence": max(1, runtime["sequence"]),
        "renderer_sequence_before_request": max(0, runtime["sequence"] - 1),
        "projection": {
            "kind": "production_sessionDocument_metadata_projection",
            "original_document_sha256": value_sha(case["document"]),
            "projected_document_sha256": value_sha(projected),
            "original_runtime_sha256": value_sha(case["runtime"]),
            "projected_runtime_sha256": value_sha(runtime),
            "original_feedback_document_id": (
                case["runtime"]["feedback_journal"]["document_id"]
                if case["runtime"].get("feedback_journal")
                else None
            ),
            "snapshot_as_of": case["runtime"]["sequence"],
            "zero_sequence": (
                "The first production request is sequence 1; no past event is shifted."
            ),
        },
    }


class RegisteredRoster:
    def __init__(self, directory, manifest_sha256, format_directory=None, format_sha256=None):
        self.directory = Path(directory).resolve()
        require(
            sha(self.directory / "manifest.json") == manifest_sha256,
            "Registered roster manifest changed.",
        )
        self.manifest_sha256 = manifest_sha256
        self.format_name = "json"
        self.format_manifest_sha256 = None
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        require(
            self.manifest["version"] == "passive-actions-v3-electron-roster-1",
            "Unknown roster format.",
        )
        for name, expected in self.manifest["files_sha256"].items():
            path = (self.directory / name).resolve()
            require(
                path.is_relative_to(self.directory) and sha(path) == expected,
                f"Registered roster payload changed: {name}",
            )
        self.entries = rows(self.directory / "roster.jsonl")
        cases, requests = (
            rows(self.directory / "cases.jsonl"),
            rows(self.directory / "requests-json.jsonl"),
        )
        self.cases = {case["id"]: case for case in cases}
        self.requests = {request["case_id"]: request for request in requests}
        require(
            len(self.entries)
            == len(self.cases)
            == len(cases)
            == len(requests)
            == len(self.requests)
            == self.manifest["cases"],
            "Duplicate/missing roster case.",
        )
        require(
            [c["id"] for c in cases]
            == [e["case_id"] for e in self.entries]
            == [r["case_id"] for r in requests],
            "Registered order changed.",
        )
        for entry in self.entries:
            case, request = self.cases[entry["case_id"]], self.requests[entry["case_id"]]
            require(
                case["split"] == case["source"]["split"] == request["split"] == "dev",
                "The registered actual-model roster must be development only.",
            )
            require(
                request["format"] == "json" and 0 < len(request["prefix_ids"]) <= 3584,
                "Roster requires the registered JSON native budget.",
            )
            require(
                value_sha(case) == entry["case_payload_sha256"]
                and value_sha(request) == entry["request_payload_sha256"],
                "Roster case/request binding changed.",
            )
            require(
                case["input"] == request["input"]
                and case["messages"]
                == request["messages"]
                == contract.canonical_messages(request["input"]),
                "Registered input/messages changed.",
            )
        if format_directory:
            path = Path(format_directory)
            require(sha(path / "manifest.json") == format_sha256, "Format registration changed.")
            manifest = json.loads((path / "manifest.json").read_text())
            require(
                manifest["roster_manifest_sha256"] == manifest_sha256
                and manifest["format"] == "rewrite",
                "Format counterpart belongs to another roster.",
            )
            require(
                sha(path / "requests-rewrite.jsonl") == manifest["requests_sha256"],
                "Registered format requests changed.",
            )
            requests = rows(path / "requests-rewrite.jsonl")
            require(
                [r["case_id"] for r in requests] == [e["case_id"] for e in self.entries],
                "Format counterpart changed the fixed case roster.",
            )
            for request in requests:
                original = self.requests[request["case_id"]]
                require(
                    request["input"] == original["input"]
                    and request["split"] == "dev"
                    and request["format"] == "rewrite"
                    and request["input_sha256"] == original["input_sha256"]
                    and request["messages"] == formats.messages(original["input"], "rewrite")
                    and ordered_sha(request["messages"]) == request["messages_sha256"]
                    and ordered_sha(request["prefix_ids"]) == request["prefix_ids_sha256"],
                    "Format counterpart altered input or request bindings.",
                )
            self.requests = {r["case_id"]: r for r in requests}
            self.format_name, self.format_manifest_sha256 = "rewrite", format_sha256

    def fixture(self, case_id, document_id):
        require(case_id in self.cases, "Case is absent from the frozen roster.")
        require(
            len(self.requests[case_id]["prefix_ids"]) <= 3584,
            "Fixed visible input exceeds the selected format's 3584 token budget; "
            "no window change is allowed.",
        )
        return fixture_state(self.cases[case_id], document_id)

    def media(self, case_id):
        source = self.cases[case_id]["source"]
        ids = sorted(
            {i["source"] for i in self.cases[case_id]["document"]["content"] if "source" in i}
        )
        require(len(ids) == 1, "This small roster driver requires one bound source per case.")
        if not source["natural_audio"]:
            require(source.get("generated_audio") is False, "Unregistered generated source.")
            return {
                "kind": "symbolic_no_recording",
                "source_ids": ids,
                "bytes": [],
                "acoustic_claim": False,
            }
        evidence = None
        record = source
        if not source.get("media_path"):
            acquisition = (
                ROOT
                / "artifacts/datasets/nep-passive-v1/sources"
                / source["id"]
                / "acquisition.json"
            )
            if acquisition.exists():
                record = json.loads(acquisition.read_text())
                evidence = {"path": str(acquisition), "sha256": sha(acquisition)}
            else:
                manifest = ROOT / "data/manifests/passive-existing-sources-v1.json"
                matches = [
                    s
                    for s in json.loads(manifest.read_text())["sources"]
                    if s["id"] == source["id"] and s["split"] == "dev"
                ]
                require(len(matches) == 1, "Natural source lacks a unique bound local asset.")
                record = matches[0]
                evidence = {"path": str(manifest), "sha256": sha(manifest)}
            require(
                record["id"] == source["id"]
                and record["split"] == "dev"
                and record["source_group_id"] == source["source_group"],
                "Local asset provenance belongs to another source.",
            )
        media_path = (ROOT / record["media_path"]).resolve()
        expected = record.get("media_sha256") or record.get("file_sha256", {}).get(
            record["media_path"]
        )
        require(
            expected and media_path.is_file() and sha(media_path) == expected,
            "Natural recording is absent or its content hash changed.",
        )
        return {
            "kind": "verified_original_recording",
            "source_ids": ids,
            "path": str(media_path),
            "sha256": expected,
            "bytes": media_path.stat().st_size,
            "asset_provenance": evidence,
            "acoustic_claim": False,
        }


class RegisteredAdapter:
    """Delegate generation only after the exact registered messages and tokens were seen."""

    def __init__(self, adapter, audit):
        self.adapter, self.audit = adapter, audit
        self.expected = self.case_id = None
        self.verified = False
        self.started = 0

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def arm(self, case_id, request):
        self.case_id, self.expected, self.verified = case_id, copy.deepcopy(request), False

    def apply_chat_template(self, messages, **kwargs):
        ids = self.adapter.apply_chat_template(messages, **kwargs)
        actual_messages = (
            self.adapter.request_messages(json.loads(messages[-1]["content"]))
            if hasattr(self.adapter, "request_messages")
            else messages
        )
        if self.expected and actual_messages == self.expected["messages"]:
            if ids != self.expected["prefix_ids"]:
                self.audit(
                    {
                        "kind": "input_binding_failure",
                        "case_id": self.case_id,
                        "model_attempted": False,
                        "error": "Registered native prefix differs at the gateway.",
                        "observed_prefix_ids": ids,
                        "expected_prefix_ids_sha256": self.expected["prefix_ids_sha256"],
                    }
                )
                raise ValueError("Registered native prefix differs at the gateway.")
            self.verified = True
        return ids

    async def generate(self, messages, output_schema, timeout):
        try:
            require(
                self.expected is not None and messages == self.expected["messages"],
                "Gateway messages differ from the registered input.",
            )
            require(self.verified, "Registered native prefix was not verified before generation.")
        except ValueError as error:
            self.audit(
                {
                    "kind": "input_binding_failure",
                    "case_id": self.case_id,
                    "error": str(error),
                    "model_attempted": False,
                    "observed_messages_sha256": value_sha(messages),
                }
            )
            raise
        self.started += 1
        scripted = getattr(self.adapter, "scripted_fixture", False)
        self.audit(
            {
                "kind": "fixture_generation_started" if scripted else "model_call_started",
                "case_id": self.case_id,
                "model_attempted": not scripted,
                "model_identity": self.adapter.model,
                "input_sha256": self.expected["input_sha256"],
                "messages_sha256": self.expected["messages_sha256"],
                "prefix_ids_sha256": self.expected["prefix_ids_sha256"],
                "remaining_deadline_seconds": timeout,
            }
        )
        return await self.adapter.generate(messages, output_schema, timeout)


def check_native(roster, tokenizer):
    result = []
    for entry in roster.entries:
        case_id = entry["case_id"]
        row = {"case_id": case_id, "model_attempted": False}
        try:
            fixture = roster.fixture(case_id, f"p6-check-{entry['order']}")
            rt = fixture["runtime"]
            ledger = (
                contract.PassiveFeedbackLedger.restore(rt["feedback_journal"])
                if rt.get("feedback_journal")
                else None
            )

            class FormatTokenizer:
                def apply_chat_template(self, messages, **kwargs):
                    return tokenizer.apply_chat_template(
                        formats.messages(json.loads(messages[-1]["content"]), roster.format_name),
                        **kwargs,
                    )

            bundle = contract.PassiveContextBuilder(FormatTokenizer()).build(
                fixture["session_document"],
                rt["focus_uuid"],
                as_of=fixture["prediction_sequence"],
                selection=rt["selection"],
                history=rt["history"],
                feedback=ledger,
            )
            request = roster.requests[case_id]
            messages = formats.messages(bundle["input"], roster.format_name)
            require(
                bundle["input"] == request["input"] and messages == request["messages"],
                "Production-projected context differs from registration.",
            )
            require(
                tokenizer.apply_chat_template(messages, **TOKEN_OPTIONS) == request["prefix_ids"],
                "Native projected prefix differs from registration.",
            )
            row.update(
                status="passed",
                projection=fixture["projection"],
                media=roster.media(case_id),
                input_sha256=request["input_sha256"],
                messages_sha256=request["messages_sha256"],
                prefix_ids_sha256=request["prefix_ids_sha256"],
            )
        except Exception as error:
            row.update(status="hydration_binding_failure", error=str(error))
        result.append(row)
    return result


def register_rewrite(roster, tokenizer, tokenizer_identity, output):
    output.mkdir(parents=True, exist_ok=False)
    requests = []
    for entry in roster.entries:
        original = roster.requests[entry["case_id"]]
        messages = formats.messages(original["input"], "rewrite")
        ids = tokenizer.apply_chat_template(messages, **TOKEN_OPTIONS)
        requests.append(
            {
                **original,
                "format": "rewrite",
                "messages": messages,
                "messages_sha256": ordered_sha(messages),
                "prefix_ids": ids,
                "prefix_ids_sha256": ordered_sha(ids),
            }
        )
    path = output / "requests-rewrite.jsonl"
    path.write_text("".join(formats.compact(r) + "\n" for r in requests))
    manifest = {
        "version": "p6-fixed-roster-format-counterpart-1",
        "format": "rewrite",
        "roster_manifest_sha256": roster.manifest_sha256,
        "requests_sha256": sha(path),
        "model_attempts": 0,
        "selection_changed": False,
        "visible_inputs_changed": False,
        "planned": len(requests),
        "budget_eligible": sum(len(r["prefix_ids"]) <= 3584 for r in requests),
        "budget_exclusions": [
            {"case_id": r["case_id"], "native_prefix_tokens": len(r["prefix_ids"])}
            for r in requests
            if len(r["prefix_ids"]) > 3584
        ],
        "tokenizer": tokenizer_identity,
        "sources_sha256": {
            str(Path(__file__).relative_to(ROOT)): sha(__file__),
            "ml/passive_actions.py": sha(ROOT / "ml/passive_actions.py"),
            "ml/passive_formats_actions_v3.py": sha(ROOT / "ml/passive_formats_actions_v3.py"),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="CPU tokenizer-only hydration check; no model is loaded."
    )
    parser.add_argument("--roster", type=Path)
    parser.add_argument("--roster-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--register-rewrite", type=Path)
    parser.add_argument("--tokenizer-server", action="store_true")
    args = parser.parse_args()
    import prepare_passive_actions_final as legacy

    tokenizer, identity = legacy.tokenizer()
    if args.tokenizer_server:
        for line in sys.stdin:
            try:
                result = {"ids": tokenizer.apply_chat_template(json.loads(line), **TOKEN_OPTIONS)}
            except Exception as error:
                result = {"error": str(error)}
            print(json.dumps(result), flush=True)
        return
    require(
        args.roster and args.roster_sha256 and args.output,
        "Provide roster, pinned hash and output.",
    )
    roster = RegisteredRoster(args.roster, args.roster_sha256)
    if args.register_rewrite:
        register_rewrite(roster, tokenizer, identity, args.register_rewrite)
        roster = RegisteredRoster(
            args.roster,
            args.roster_sha256,
            args.register_rewrite,
            sha(args.register_rewrite / "manifest.json"),
        )
    checks = check_native(roster, tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(
            {
                "kind": "native_tokenizer_only_hydration_check",
                "model_attempts": 0,
                "roster_sha256": args.roster_sha256,
                "tokenizer": identity,
                "source_sha256": sha(Path(__file__)),
                "planned": len(checks),
                "checks": checks,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    require(
        all(
            c["status"] == "passed" or "exceeds the selected format" in c.get("error", "")
            for c in checks
        ),
        "One or more registered hydration checks failed.",
    )


if __name__ == "__main__":
    main()
