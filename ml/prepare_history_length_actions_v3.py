"""Register eight authored DEV episodes at exact history lengths 0, 3 and 8.

Preparation executes real synthetic transitions and fixes the window at length 8.
Admission requires an independent, prediction-blind review of every visible input.
Neither command runs model inference or reads an existing train/final-test corpus.
"""

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

from grading import transcript_media
import passive_actions as contract
import passive_formats_actions_v3 as formats
import prepare_actions_v3_conditions as conditions
import prepare_passive_actions_v3 as dataset
from replay_session import session_hash

VERSION = "passive-actions-v3-history-length-1"
FIXTURE_VERSION = "history-length-controls-001"
FIXTURE = dataset.FIXTURES / "history-length-controls-001.json"
LENGTHS = (0, 3, 8)
WINDOW_POLICY = "fixed_at_eight_history"
MECHANISMS = {
    "evidence_in_last_three": ("no_action", "correct_text", "correct_text"),
    "evidence_only_in_older_five": ("no_action", "no_action", "correct_text"),
    "unrelated_role_history": ("no_action", "no_action", "no_action"),
    "latest_correction_supersedes_older": ("no_action", "correct_text", "correct_text"),
}
require, digest, sha256 = dataset.require, dataset.digest, dataset.sha256


def arm_name(length):
    require(type(length) is int and length in LENGTHS, "Unregistered history length.")
    return f"history_length_{length}"


def condition_spec(length):
    arm_name(length)
    return conditions.condition_specs([])["context_history" if length else "context_only"]


def without_history(snapshot):
    return {key: value for key, value in snapshot.items() if key != "recent_edits"}


def projection(snapshot, length):
    arm_name(length)
    require(len(snapshot["recent_edits"]) == 8, "Anchor needs eight actual edits.")
    require(snapshot["recent_feedback"] == [], "Feedback must be empty in all arms.")
    result = copy.deepcopy(snapshot)
    result["recent_edits"] = result["recent_edits"][-length:] if length else []
    return contract.parse_snapshot(result).model_dump()


def validate_fixture(fixture):
    require(
        fixture["version"] == FIXTURE_VERSION
        and fixture["history_lengths"] == list(LENGTHS)
        and fixture["window_policy"] == WINDOW_POLICY,
        "Fixture differs from the registered length/window policy.",
    )
    provenance = fixture["provenance"]
    require(
        provenance["actor"] == "synthetic"
        and provenance["author_id"].strip()
        and provenance["model_predictions_consulted"] is False
        and provenance["observed_human_actions"] is False
        and provenance["final_test_bodies_read"] is False
        and provenance["timing"] == "authored_symbolic_not_audio",
        "Controls must be authored, prediction-blind development material.",
    )
    episodes = fixture["episodes"]
    require(
        len(episodes) == 8
        and len({item["id"] for item in episodes}) == 8
        and len({item["source_group"] for item in episodes}) == 8
        and Counter((item["language"], item["mechanism"]) for item in episodes)
        == Counter((language, mechanism) for language in ("en", "ko") for mechanism in MECHANISMS),
        "Register exactly one episode per language and mechanism.",
    )
    for language in ("en", "ko"):
        require(len(fixture["distractors"][language]) == 8, "Need eight distractor positions.")
        require(
            sum(len(sentence.split()) for sentence in fixture["neutral_body"][language]) >= 64,
            "Neutral items must separate prior targets from the current window.",
        )
    for episode in episodes:
        require(
            episode["source_group"].startswith("authored-history-length-")
            and episode["source_group"] not in dataset.RESERVED
            and episode["id"].startswith("history-length-")
            and episode["expected_actions"]
            == dict(zip(map(str, LENGTHS), MECHANISMS[episode["mechanism"]], strict=True))
            and episode["hidden_names"]
            and all(isinstance(name, str) and name for name in episode["hidden_names"]),
            "Episode source, proposed reference or name-leak checks changed.",
        )
        require(
            all(str(int(index)) == index and 1 <= int(index) <= 8 for index in episode["critical_edits"]),
            "Critical edit must refer to an executed position 1 through 8.",
        )
        edits = episode_edits(fixture, episode)
        require(
            all(edit["before"] and edit["after"] and edit["before"] != edit["after"] for edit in edits),
            "Every prior action must actually change display text.",
        )


def episode_edits(fixture, episode):
    result = []
    for index, pair in enumerate(fixture["distractors"][episode["language"]], 1):
        edit = episode["critical_edits"].get(str(index))
        result.append(copy.deepcopy(edit) if edit else {"key": f"history-{index}", "before": pair[0], "after": pair[1]})
    return result


def episode_document(fixture, episode, edits):
    initial = {}
    for edit in edits:
        initial.setdefault(edit["key"], edit["before"])
    paragraphs = [[{"key": key, "text": text, "ms": 5000}] for key, text in initial.items()]
    for index, sentence in enumerate(fixture["neutral_body"][episode["language"]]):
        paragraphs.append([
            {"key": f"neutral-{index}-{word_index}", "text": word, "ms": 500}
            for word_index, word in enumerate(sentence.split())
        ])
    paragraphs.append([{"key": "current", "text": episode["current"], "ms": 5000}])
    return dataset.authored_document({"id": episode["id"], "language": episode["language"], "paragraphs": paragraphs})


def assert_no_visible_name_leak(snapshot, hidden_names):
    rendered = dataset.packed(without_history(snapshot))
    require(
        not any(name in rendered for name in hidden_names),
        "A history-only name leaked into non-history input.",
    )


def validate_request(request):
    conditions.validate_request(request)
    meta = request["history_length"]
    length = meta["length"]
    require(
        set(meta) == {
            "version", "arm", "length", "anchor_length", "window_policy", "episode_id",
            "fixture_sha256", "anchor_input_sha256", "projection_input_sha256",
            "non_history_sha256", "history_chain_sha256",
        }
        and meta["version"] == VERSION
        and meta["arm"] == arm_name(length)
        and meta["anchor_length"] == 8
        and meta["window_policy"] == WINDOW_POLICY
        and meta["episode_id"] == request["case_id"]
        and request["format"] == "json"
        and request["condition"] == condition_spec(length)
        and len(request["input"]["recent_edits"]) == length
        and request["input"]["recent_feedback"] == []
        and set(meta["projection_input_sha256"]) == set(map(str, LENGTHS))
        and meta["anchor_input_sha256"] == meta["projection_input_sha256"]["8"]
        and request["input_sha256"] == meta["projection_input_sha256"][str(length)]
        and digest(without_history(request["input"])) == meta["non_history_sha256"],
        "History-length request differs from its registered projection.",
    )
    hashes = [meta[key] for key in ("fixture_sha256", "anchor_input_sha256", "non_history_sha256", "history_chain_sha256")]
    hashes.extend(meta["projection_input_sha256"].values())
    require(
        all(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value) for value in hashes),
        "History-length bindings must be SHA-256 digests.",
    )


def validate_pair(left, right):
    for request in (left, right):
        validate_request(request)
    a, b = left["history_length"], right["history_length"]
    require(a["length"] != b["length"], "History comparison needs two different lengths.")
    require(
        {key: value for key, value in a.items() if key not in ("length", "arm")}
        == {key: value for key, value in b.items() if key not in ("length", "arm")}
        and without_history(left["input"]) == without_history(right["input"]),
        "History comparison changed its anchor, fixture or non-history state.",
    )
    short, long = sorted((left, right), key=lambda row: row["history_length"]["length"])
    length = short["history_length"]["length"]
    require(
        short["input"]["recent_edits"] == (long["input"]["recent_edits"][-length:] if length else []),
        "History comparison must use the exact chronological suffix.",
    )


def build_episode(fixture, episode, tokenizer, fixture_sha256):
    builder = contract.PassiveContextBuilder(tokenizer, policy=contract.PassivePolicy(history_limit=8, feedback_limit=0))
    edits = episode_edits(fixture, episode)
    document, keys, source_id = episode_document(fixture, episode, edits)
    source = {
        "id": source_id, "source_group": episode["source_group"], "split": "dev",
        "language": episode["language"], "provenance": "authored_symbolic_timing_not_audio",
        "natural_audio": False, "generated_audio": False, "acoustically_verified": False,
    }
    history, history_proofs = [], []
    for sequence, edit in enumerate(edits, 1):
        before = copy.deepcopy(document)
        item = next(item for item in before["content"] if item["uuid"] == keys[edit["key"]])
        require(item["text"] == edit["before"], "Prior correction does not match its actual before state.")
        bundle = builder.build(before, keys[edit["key"]], as_of=sequence - 1, history=history, caption_profile=dataset.PROFILE)
        prediction = dataset.resolve({"action": "correct_text", "targets": [edit["key"]], "text": edit["after"]}, keys, bundle)
        proof = dataset.execution(before, bundle, prediction, source)
        proof["hydrate_start_document_from"] = f"case.runtime.history[{sequence - 1}].before"
        document = contract.apply_passive_edit(before, bundle, prediction)
        history.append(contract.committed_transition(before, document, prediction, sequence))
        history_proofs.append({"sequence": sequence, **proof})
    require(len({item["transaction_id"] for item in history}) == 8, "Eight distinct transactions are required.")
    anchor = builder.build(document, keys["current"], as_of=8, history=history, caption_profile=dataset.PROFILE)
    require(len(anchor["input"]["recent_edits"]) == 8, "Actual runtime did not expose eight edits.")
    require(
        not {keys[edit["key"]] for edit in edits}.intersection(item["uuid"] for item in anchor["mapping"].values()),
        "Prior corrected items remain visible in the fixed current window.",
    )
    assert_no_visible_name_leak(anchor["input"], episode["hidden_names"])
    snapshots = {str(length): projection(anchor["input"], length) for length in LENGTHS}
    metadata = {
        "version": VERSION, "anchor_length": 8, "window_policy": WINDOW_POLICY,
        "episode_id": episode["id"], "fixture_sha256": fixture_sha256,
        "anchor_input_sha256": conditions.input_hash(anchor["input"]),
        "projection_input_sha256": {key: conditions.input_hash(value) for key, value in snapshots.items()},
        "non_history_sha256": digest(without_history(anchor["input"])),
        "history_chain_sha256": digest(history),
    }
    cases, requests, proofs = [], [], []
    for length in LENGTHS:
        snapshot, condition = snapshots[str(length)], condition_spec(length)
        messages = conditions.expected_messages(snapshot, "json", condition)
        prefix = conditions.native_prefix(tokenizer, messages)
        require(len(prefix) <= formats.INPUT_LIMIT, "A registered arm exceeds the native input budget.")
        bundle = copy.deepcopy(anchor)
        bundle["input"] = snapshot
        bundle["context"].update(input_tokens=len(prefix), prompt_hash=session_hash(messages))
        primary = {"action": "no_action"}
        if episode["expected_actions"][str(length)] != "no_action":
            require(episode["replacement"] and episode["replacement"] != episode["current"], "Missing actual replacement.")
            primary = dataset.resolve({"action": "correct_text", "targets": ["current"], "text": episode["replacement"]}, keys, bundle)
        serialized = formats.serialize(snapshot, primary, "json")
        require(formats.parse(serialized, snapshot, "json").model_dump() == primary, "Reference format round trip failed.")
        require(len(tokenizer.encode(serialized, add_special_tokens=False)) + 1 <= formats.OUTPUT_LIMIT, "Reference exceeds the output reserve.")
        runtime = {"focus_uuid": keys["current"], "selection": [], "sequence": 8, "history": history, "feedback_journal": None, "caption_profile": dataset.PROFILE}
        meta = {**metadata, "length": length, "arm": arm_name(length)}
        case = dataset.case_row(
            episode["id"], "dev", episode["language"], source, document, bundle, primary, [], runtime,
            episode["rationale"], {"stratum": "authored_history_length_control", "scenario_family": episode["mechanism"], "selection_method": "preregistered_authored_mechanism_control", "generation_provenance": fixture["provenance"]},
        )
        case.update(bundle=bundle, history_length=meta, condition=condition, condition_sha256=digest(condition))
        proof = dataset.validate_case(case, builder, bundle=bundle)
        case["execution_sha256"] = digest(proof)
        after = contract.apply_passive_edit(document, bundle, primary)
        case["accepted_result_sha256"] = [digest(transcript_media(after))]
        request = {
            "case_id": episode["id"], "split": "dev", "format": "json", "input": snapshot,
            "input_sha256": conditions.input_hash(snapshot), "messages": messages,
            "messages_sha256": digest(messages), "prefix_ids": prefix, "prefix_ids_sha256": digest(prefix),
            "condition": condition, "condition_sha256": digest(condition), "history_length": meta,
        }
        validate_request(request)
        cases.append(case)
        requests.append(request)
        proofs.append({"case_id": episode["id"], "arm": arm_name(length), "proof": proof})
    for request in requests[:-1]:
        validate_pair(request, requests[-1])
    return cases, requests, {"case_id": episode["id"], "history_chain_sha256": digest(history), "history": history_proofs, "references": proofs}


def review_binding(case, request):
    return {
        "case_id": case["id"], "arm": request["history_length"]["arm"],
        "input_sha256": request["input_sha256"], "condition_sha256": request["condition_sha256"],
        "history_length_sha256": digest(request["history_length"]),
        "draft_case_sha256": digest(case), "accepted_predictions_sha256": digest(case["accepted_predictions"]),
        "accepted_result_sha256": sorted(case["accepted_result_sha256"]),
    }


def source_bindings(fixture_path):
    sources = dataset.legacy.source_closure([
        Path(__file__), Path(conditions.__file__), contract.PROMPT_PATH, Path(fixture_path),
        dataset.ROOT / "ml/compare_actions_v3_results.py",
    ])
    return {str((dataset.ROOT / name).resolve()): value for name, value in sources.items()}


def prepare(fixture_path, output, tokenizer, tokenizer_identity):
    fixture_path, output = Path(fixture_path).resolve(), Path(output).resolve()
    require(not output.exists(), "Use a new draft directory.")
    fixture = json.loads(fixture_path.read_text())
    validate_fixture(fixture)
    fixture_binding = conditions.binding(fixture_path)
    cases, requests, proofs = [], [], []
    for episode in fixture["episodes"]:
        episode_cases, episode_requests, proof = build_episode(fixture, episode, tokenizer, fixture_binding["sha256"])
        cases.extend(episode_cases)
        requests.extend(episode_requests)
        proofs.append(proof)
    bindings = [review_binding(case, request) for case, request in zip(cases, requests, strict=True)]
    output.mkdir(parents=True)
    dataset.write_rows(output / "draft-cases.jsonl", cases)
    dataset.write_rows(output / "draft-requests-json.jsonl", requests)
    dataset.write_rows(output / "execution-proofs.jsonl", proofs)
    dataset.write_rows(output / "review-queue.jsonl", [
        {**binding, "input": case["input"], "messages": request["messages"], "proposed_judgment": case["judgment"], "proposed_references": case["accepted_predictions"], "author_rationale": case["rationale"]}
        for binding, case, request in zip(bindings, cases, requests, strict=True)
    ])
    dataset.write(output / "manifest.json", {
        "version": VERSION, "status": "draft_pending_independent_review", "created_at": datetime.now(timezone.utc).isoformat(),
        "fixture": fixture_binding, "author_id": fixture["provenance"]["author_id"], "episodes": 8, "requests": 24,
        "lengths": list(LENGTHS), "window_policy": WINDOW_POLICY, "tokenizer": tokenizer_identity,
        "final_test_bodies_read": False, "sources_sha256": source_bindings(fixture_path),
        "files_sha256": {path.name: sha256(path) for path in output.iterdir() if path.is_file()},
    })
    dataset.write(output / "review-template.json", {
        "version": VERSION, "draft_manifest_sha256": sha256(output / "manifest.json"),
        "reviewer_id": "", "actor": "ai_generated", "model_predictions_consulted": False,
        "decisions": [{**binding, "status": "pending", "whole_visible_window_reviewed": False, "rationale": ""} for binding in bindings],
    })
    return {"status": "draft_pending_independent_review", "episodes": 8, "requests": 24, "output": str(output)}


def validate_review(review, manifest, manifest_sha256, cases, requests):
    require(
        review["version"] == VERSION
        and review["draft_manifest_sha256"] == manifest_sha256
        and review["reviewer_id"].strip()
        and review["reviewer_id"].strip() != manifest["author_id"]
        and review["actor"] in ("human", "ai_generated")
        and review["model_predictions_consulted"] is False,
        "Admission requires an independent, prediction-blind reviewer bound to this draft.",
    )
    expected = {(case["id"], request["history_length"]["arm"]): review_binding(case, request) for case, request in zip(cases, requests, strict=True)}
    decisions = {(item["case_id"], item["arm"]): item for item in review["decisions"]}
    require(len(expected) == len(cases) == len(requests) == len(review["decisions"]) == len(decisions) == 24 and set(expected) == set(decisions), "Every one of the 24 input views needs exactly one review.")
    for key, binding in expected.items():
        decision = decisions[key]
        require(
            all(decision.get(field) == value for field, value in binding.items())
            and decision["status"] == "qualified"
            and decision["whole_visible_window_reviewed"] is True
            and decision["rationale"].strip(),
            "Unqualified, changed or partially reviewed references cannot be admitted.",
        )
    return decisions


def admit(draft, review_path, output):
    draft, review_path, output = Path(draft).resolve(), Path(review_path).resolve(), Path(output).resolve()
    require(not output.exists(), "Use a new admitted preparation directory.")
    manifest_path = draft / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["version"] == VERSION and manifest["status"] == "draft_pending_independent_review" and manifest["final_test_bodies_read"] is False, "Unknown draft scope.")
    for name, expected in manifest["files_sha256"].items():
        require(Path(name).name == name and sha256(draft / name) == expected, "Draft evidence changed.")
    for name, expected in manifest["sources_sha256"].items():
        require(sha256(name) == expected, "Preparation source changed after registration.")
    cases, requests = dataset.rows(draft / "draft-cases.jsonl"), dataset.rows(draft / "draft-requests-json.jsonl")
    review = json.loads(review_path.read_text())
    decisions = validate_review(review, manifest, sha256(manifest_path), cases, requests)
    for request in requests:
        validate_request(request)
    for identity in {request["case_id"] for request in requests}:
        episode_requests = [row for row in requests if row["case_id"] == identity]
        require(sorted(row["history_length"]["length"] for row in episode_requests) == list(LENGTHS), "Episode views are incomplete.")
        anchor = next(row for row in episode_requests if row["history_length"]["length"] == 8)
        for request in episode_requests:
            if request is not anchor:
                validate_pair(request, anchor)
    review_file = conditions.binding(review_path)
    output.mkdir(parents=True)
    for length in LENGTHS:
        directory = output / arm_name(length)
        arm_cases, arm_requests = [], []
        for case, request in zip(cases, requests, strict=True):
            if request["history_length"]["length"] != length:
                continue
            value = copy.deepcopy(case)
            value["judgment"]["independent_review"] = "qualified_full_visible_window"
            value["history_length_qualification"] = {
                "review": review_file, "reviewer_id": review["reviewer_id"],
                "decision": decisions[(case["id"], arm_name(length))],
            }
            arm_cases.append(value)
            arm_requests.append(request)
        dataset.write_rows(directory / "cases.jsonl", arm_cases)
        dataset.write_rows(directory / "requests-json.jsonl", arm_requests)
        dataset.write(directory / "manifest.json", {
            "version": conditions.VERSION, "status": "prepared_not_inferred", "history_length_version": VERSION,
            "study_arm": arm_name(length), "history_length": length, "window_policy": WINDOW_POLICY,
            "condition": condition_spec(length), "condition_sha256": digest(condition_spec(length)),
            "draft_manifest": conditions.binding(manifest_path), "semantic_review": review_file,
            "requests": len(arm_cases), "original_development_cases": 8, "excluded": 0, "eligibility_format": "json",
            "case_ids": [case["id"] for case in arm_cases], "coverage": conditions.coverage(arm_cases),
            "tokenizer": manifest["tokenizer"], "final_test_bodies_read": False, "sources_sha256": manifest["sources_sha256"],
            "files_sha256": {path.name: sha256(path) for path in directory.iterdir() if path.is_file()},
        })
    dataset.write(output / "manifest.json", {
        "version": VERSION, "status": "independently_qualified_not_inferred",
        "draft_manifest": conditions.binding(manifest_path), "semantic_review": review_file,
        "episodes": 8, "requests": 24, "final_test_bodies_read": False,
        "arms": {arm_name(length): conditions.binding(output / arm_name(length) / "manifest.json") for length in LENGTHS},
        "claim_scope": "Authored symbolic mechanism controls; no natural-history or causal training-source claim.",
    })
    return {"status": "independently_qualified_not_inferred", "episodes": 8, "requests": 24, "output": str(output)}


def native_tokenizer(config_path, tokenizer_path):
    from importlib.metadata import version
    from transformers import AutoTokenizer

    config_path, tokenizer_path = Path(config_path).resolve(), Path(tokenizer_path).resolve()
    config = json.loads(config_path.read_text())
    require(config["candidate_id"] == "gemma26" and config["format"] == "json", "Use the selected native JSON configuration.")
    require(version("transformers") == config["libraries"]["transformers"], "Native tokenizer library differs from the frozen configuration.")
    files = config["tokenizer"]["files_sha256"]
    require(set(files) == {"config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"}, "Unexpected tokenizer file set.")
    for name, expected in files.items():
        require(sha256(tokenizer_path / name) == expected, "Native tokenizer file differs from the frozen configuration.")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    return tokenizer, {
        "configuration": conditions.binding(config_path), "path": str(tokenizer_path),
        "repository": config["base"]["repository"], "revision": config["base"]["revision"],
        "files_sha256": files, "transformers": version("transformers"),
        "eos_token_id": tokenizer.eos_token_id, "model_weights_loaded": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--fixture", type=Path, default=FIXTURE)
    preparation.add_argument("--config", type=Path, required=True)
    preparation.add_argument("--tokenizer", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    admission = commands.add_parser("admit")
    admission.add_argument("--draft", type=Path, required=True)
    admission.add_argument("--review", type=Path, required=True)
    admission.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        validate_fixture(json.loads(args.fixture.read_text()))
        tokenizer, identity = native_tokenizer(args.config, args.tokenizer)
        result = prepare(args.fixture, args.output, tokenizer, identity)
    else:
        result = admit(args.draft, args.review, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
