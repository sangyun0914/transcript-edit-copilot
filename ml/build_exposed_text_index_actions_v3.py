"""Index explicitly registered exposed inputs without discovering source files.

Only visible item wording enters the index. References, document bodies, history,
and prompt instructions do not. This is an exact-text screen, not a split policy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
VERSION = "exposed-text-index-actions-v3-2"
HISTORICAL_METADATA = "artifacts/datasets/nep-passive-actions-final-test-v1/exposure-metadata-attempt-01.json"
SPLITS = {"train", "dev", "calibration"}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def unique_object(pairs):
    require(len({key for key, _ in pairs}) == len(pairs), "Ambiguous duplicate JSON field.")
    return dict(pairs)


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(packed(value).encode()).hexdigest()


def corpus_text(value):
    value = value.get("input", value)
    items = value.get("items", [])
    fields = value.get("item_fields")
    if fields and items and isinstance(items[0], list):
        items = [dict(zip(fields, item, strict=True)) for item in items]
    text = " ".join(item.get("text", "") for item in items if isinstance(item, dict))
    return " ".join(unicodedata.normalize("NFC", text).split())


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_path(root, name, *, metadata=False):
    require(isinstance(name, str) and not Path(name).is_absolute(), "Expected relative inventory path.")
    require(".." not in Path(name).parts, "Inventory path traversal.")
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), "Inventory path escaped the workspace.")
    relative = str(path.relative_to(root.resolve()))
    reserved = any(token in relative.lower() for token in ("final-test", "final_test", "holdout", "sealed-test"))
    metadata_exceptions = {HISTORICAL_METADATA, "ml/prepare_passive_actions_final_test.py"}
    require(not reserved or (metadata and relative in metadata_exceptions), "Reserved body path is forbidden.")
    return path


def bound(root, binding, *, metadata=False):
    path = local_path(root, binding["path"], metadata=metadata)
    require(re.fullmatch(r"[0-9a-f]{64}", binding["sha256"]) is not None, "Malformed bound SHA256.")
    require(path.is_file() and sha256(path) == binding["sha256"], "Bound source SHA256 changed: " + binding["path"])
    return path


def select(value, path):
    if not path:
        return [value]
    head, *tail = path
    if head == "*":
        require(isinstance(value, list), "Expected array at registered input selector.")
        return [item for child in value for item in select(child, tail)]
    require(isinstance(value, dict) and head in value, "Missing registered input field: " + str(head))
    return select(value[head], tail)


def validate_input(value, schemas):
    require(isinstance(value, dict) and "input" not in value and isinstance(value.get("items"), list),
            "Expected unambiguous visible input.items array.")
    schema = {key: value[key] for key in ("schema_version", "contract_version", "task") if key in value}
    require(schema in schemas and schema, "Unexpected predictor input schema.")
    items, fields = value["items"], value.get("item_fields")
    if fields is not None:
        require(isinstance(fields, list) and fields and all(isinstance(key, str) for key in fields)
                and len(fields) == len(set(fields)) and "text" in fields, "Invalid compact item_fields.")
    if items and isinstance(items[0], list):
        require(fields and all(isinstance(item, list) and len(item) == len(fields) for item in items),
                "Malformed compact visible items.")
        items = [dict(zip(fields, item, strict=True)) for item in items]
    require(all(isinstance(item, dict) and isinstance(item.get("text", ""), str) for item in items),
            "Malformed visible item or text.")
    return corpus_text(value)


def inputs(row, entry):
    mode = entry["mode"]
    if mode == "visible_inputs":
        return [value for path in entry["input_paths"] for value in select(row, path)]
    if mode == "user_message_inputs":
        messages = row.get("messages")
        require(isinstance(messages, list) and all(isinstance(message, dict) for message in messages),
                "Expected structured messages.")
        require([message.get("role") for message in messages] == entry["message_roles"],
                "Unexpected message role sequence.")
        values = []
        for message in messages:
            if message["role"] == "user":
                require(isinstance(message.get("content"), str), "Expected JSON user-message string.")
                value = json.loads(message["content"], object_pairs_hook=unique_object)
                require(isinstance(value, dict) and sorted(value) == entry["user_input_fields"],
                        "Unexpected user input field schema.")
                values.append(value)
        require(values, "No registered user input.")
        return values
    require(mode == "runtime_without_visible_input", "Unknown extraction mode.")
    require(set(row) == set(entry["row_keys"]) and not {"input", "items", "messages"}.intersection(row)
            and bool(entry.get("exclusion_reason")), "Metadata-only runtime shape changed.")
    return []


def read_rows(path, storage):
    if storage == "json_object":
        yield json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    else:
        require(storage == "jsonl", "Unknown collection storage.")
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line, object_pairs_hook=unique_object)


def split_authority(root, entry):
    if entry.get("split_path") is not None:
        return None
    authority = entry["split_authority"]
    result = {}
    if authority["mode"] == "exact_registered_input":
        for binding in authority["collections"]:
            for row in read_rows(bound(root, binding), "jsonl"):
                require(row["id"] not in result, "Ambiguous split authority ID.")
                result[row["id"]] = row
    else:
        require(authority["mode"] == "bound_attempt_recipe", "Unknown split authority.")
        manifest = json.loads(bound(root, authority["manifest"]).read_text())
        recipe = json.loads(bound(root, authority["recipe"]).read_text())
        require(manifest["inputs_sha256"] == entry["sha256"]
                and manifest["recipe_sha256"] == authority["recipe"]["sha256"]
                and manifest["status"] == "failed_common_format_budget", "Attempt recipe/input binding changed.")
        for row in recipe["scenarios"]:
            require(row["id"] not in result, "Ambiguous attempt recipe ID.")
            result[row["id"]] = row
    return result


def original_split(row, entry, authority):
    if authority is None:
        values = select(row, entry["split_path"])
        require(len(values) == 1, "Ambiguous split field.")
        return values[0]
    identifier = row[entry["split_authority"].get("id_field", "id")]
    require(identifier in authority, "Unknown exposed input ID.")
    registered = authority[identifier]
    if entry["split_authority"]["mode"] == "exact_registered_input":
        require(row["input"] == registered["input"]
                and row["input_sha256"] == registered["metadata"]["input_sha256"],
                "Split authority input or stored canonical-message hash differs.")
        return registered["metadata"]["split"]
    return registered["split"]


def inspect_entry(root, entry):
    path = bound(root, entry)
    counts = {"rows": 0, "text_bearing_rows": 0, "visible_inputs": 0,
              "text_bearing_inputs": 0, "empty_text_inputs": 0, "metadata_only_rows": 0}
    hashes, splits = set(), {}
    require(set(entry["allowed_splits"]) <= SPLITS and entry["allowed_splits"], "Unexposed split registration.")
    authority = split_authority(root, entry)
    for row in read_rows(path, entry["storage"]):
        require(isinstance(row, dict), "Expected object row.")
        counts["rows"] += 1
        split = original_split(row, entry, authority)
        require(split in entry["allowed_splits"], "Unexpected original split.")
        splits[split] = splits.get(split, 0) + 1
        visible = inputs(row, entry)
        counts["visible_inputs"] += len(visible)
        if entry["mode"] == "runtime_without_visible_input":
            counts["metadata_only_rows"] += 1
        else:
            require(visible, "Text-bearing collection contains no selected input.")
        row_has_text = False
        for value in visible:
            text = validate_input(value, entry["input_schemas"])
            if text:
                hashes.add(digest(text))
                counts["text_bearing_inputs"] += 1
                row_has_text = True
            else:
                counts["empty_text_inputs"] += 1
        counts["text_bearing_rows"] += int(row_has_text)
    return {**counts, "original_split_rows": dict(sorted(splits.items())),
            "unique_text_hashes": len(hashes), "per_file_index_sha256": digest({"sha256": sorted(hashes)})}, hashes


def collect(root, inventory):
    require(inventory.get("version") == VERSION and inventory.get("scope") == "exposed_train_dev_diagnostic_only",
            "Unexpected inventory contract.")
    sources = inventory["files"]
    require(sources and len({entry["path"] for entry in sources}) == len(sources), "Duplicate or empty file inventory.")
    for binding in inventory["authorities"]:
        bound(root, binding, metadata=True)
    historical = json.loads(bound(root, inventory["historical_metadata"], metadata=True).read_text())
    registered = {entry["path"]: entry["sha256"] for entry in sources}
    old = historical["historical_exposed_case_collections_sha256"]
    require(all(registered.get(name) == expected for name, expected in old.items()), "Historical exposure collection omitted or changed.")
    all_hashes, reports = set(), []
    for index, entry in enumerate(sources, 1):
        stats, hashes = inspect_entry(root, entry)
        require(stats == entry["expected"], "Registered collection shape/count/index changed: " + entry["path"])
        all_hashes.update(hashes)
        reports.append({"path": entry["path"], "sha256": entry["sha256"], "mode": entry["mode"], **stats})
        if index % 25 == 0:
            print(f"Verified {index}/{len(sources)} explicit exposed collections.", flush=True)
    totals = {key: sum(row[key] for row in reports) for key in
              ("rows", "text_bearing_rows", "visible_inputs", "text_bearing_inputs", "empty_text_inputs", "metadata_only_rows")}
    return {"sha256": sorted(all_hashes)}, {"files": reports, "totals": {
        "files": len(reports), **totals, "historical_bound_files": len(old),
        "unique_text_hashes": len(all_hashes), "repeated_text_occurrences": totals["text_bearing_inputs"] - len(all_hashes)}}


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def build(inventory_path, output, root=ROOT):
    inventory_bytes = inventory_path.read_bytes()
    inventory = json.loads(inventory_bytes)
    index, report = collect(root, inventory)
    require(not output.exists(), "Output already exists; preserve prior attempts.")
    output.mkdir(parents=True)
    (output / "inventory.json").write_bytes(inventory_bytes)
    (output / "builder-source.py").write_bytes(Path(__file__).read_bytes())
    write(output / "index.json", index)
    manifest = {"version": VERSION, "status": "verified_exposed_text_index", "scope": inventory["scope"],
                "normalization": inventory["normalization"], "coverage_limits": inventory["coverage_limits"],
                "files_sha256": {name: sha256(output / name) for name in ("inventory.json", "index.json", "builder-source.py")},
                **report}
    write(output / "manifest.json", manifest)
    return manifest


def verify(output, root=ROOT):
    manifest = json.loads((output / "manifest.json").read_text())
    require(manifest["version"] == VERSION, "Unknown output manifest.")
    for name, expected in manifest["files_sha256"].items():
        require(name in {"inventory.json", "index.json", "builder-source.py"}
                and sha256(output / name) == expected, "Output file changed.")
    require(manifest["files_sha256"]["builder-source.py"] == sha256(__file__), "Verifier differs from saved builder.")
    inventory = json.loads((output / "inventory.json").read_text())
    index, report = collect(root, inventory)
    require(index == json.loads((output / "index.json").read_text()) and report["files"] == manifest["files"]
            and report["totals"] == manifest["totals"], "Index readback differs.")
    return {"status": "passed", "manifest_sha256": sha256(output / "manifest.json"),
            "index_sha256": sha256(output / "index.json"), **report["totals"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        require(args.inventory is not None, "Build needs an explicit inventory.")
        result = build(args.inventory, args.output)["totals"]
    else:
        result = verify(args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
