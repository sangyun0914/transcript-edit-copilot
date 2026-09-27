"""Execute AMI DEV draft references through the common cursor-context builder."""

import argparse
import json
from pathlib import Path

from ami_dev_release import ami_dev_sources
from context_policy import ContextBuilder
from dataset_release import comparison_tokenizers, draft_case
from dataset_release_inventory import ROOT, digest, read, write
from iteration import jsonl
from source_qualification import TOKENIZER


def verify(output):
    from transformers import AutoTokenizer

    evidence_folder = ROOT / "artifacts/datasets/nep-release-v1/ami-dev-evidence-v1"
    fragment_path = evidence_folder / "recipe-fragment.json"
    fragment = read(fragment_path)
    recipe = read(ROOT / "ml/fixtures/dataset-release-v2.json")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "recipe.json", fragment)
    (output / "sessions").mkdir()
    (output / "evidence").mkdir()
    sources, windows = ami_dev_sources()
    source = sources["ami-is1008b"]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    comparisons, _ = comparison_tokenizers(recipe)
    prompt = (ROOT / "ml/prompt-v3.txt").read_text()
    builder = ContextBuilder(tokenizer, prompt, budget_tokenizers=comparisons)
    cases, contexts, executions, outcomes = [], [], [], []
    for raw in fragment["cases"]:
        spec = {**raw, "recipe_version": fragment["version"]}
        try:
            case, context, execution = draft_case(
                spec, source, windows, builder, output, prompt
            )
            cases.append(case)
            contexts.append(context)
            executions.append(execution)
            outcomes.append(
                {
                    "id": spec["id"],
                    "status": "passed_common_context_reference_execution_replay",
                    "schema_version": case["input"]["schema_version"],
                    "visible_items": case["metadata"]["unit_counts"][
                        "visible_timed_items"
                    ],
                    "input_tokens": context["token_count"],
                    "source_context_seconds": case["metadata"][
                        "source_context_seconds"
                    ],
                    "effective_input_sha256": case["metadata"][
                        "effective_input_sha256"
                    ],
                    "accepted_alternatives": len(case["acceptableOutputs"]),
                    "valid_semantic_counterfactuals": len(
                        case["metadata"]["counterfactual_audit"]
                    ),
                }
            )
        except ValueError as error:
            outcomes.append(
                {"id": spec["id"], "status": "failed", "reason": str(error)}
            )
    jsonl(output / "dev-cases.jsonl", cases)
    jsonl(output / "contexts.jsonl", contexts)
    jsonl(output / "executions.jsonl", executions)
    audit = {
        "recipe_sha256": digest(fragment_path),
        "source_import_audit": source["import_audit"],
        "outcomes": outcomes,
        "passed": sum(o["status"].startswith("passed") for o in outcomes),
        "failed": sum(o["status"] == "failed" for o in outcomes),
        "training_eligible": False,
        "independent_or_blinded_review": False,
        "not_covered": [
            "New schema/context profiles added after this verification",
            "Natural cut boundaries, listening or perceptual join quality",
            "Full combined-release inventory, deduplication and caption export",
        ],
    }
    write(output / "verification.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args().output)
