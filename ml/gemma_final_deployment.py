"""Separate-LoRA product identity for the final Gemma family; no process ownership."""

from pathlib import Path
import sys
from urllib.parse import urlsplit

import gemma_final_test_runtime as gemma
import gemma_q8_product as q8

ROOT = gemma.ROOT
PLAN = ROOT / "docs/plans/final-gemma26-runtime-bridge-2026-10-25.md"
POLICY_VERSION = "gemma26-final-deployed-evaluation-policy-1"
require, read, sha, digest, binding, bound = (gemma.require, gemma.read, gemma.sha,
                                            gemma.digest, gemma.binding, gemma.bound)


def deployment_check(directory):
    for path in (ROOT / "server", ROOT / "server/scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import run_local_copilot_actions_v3 as local
    gateway, _, command, report = local.check(directory)
    return gateway.model_dump(), command, report


def verify_loaded_adapter(gateway, observed):
    for path in (ROOT / "server", ROOT / "server/scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from app.copilot import LocalConfig, LocalLlamaAdapter
    require(gateway.get("adapter_path") and gateway.get("adapter_sha256") and gateway["adapter_scale"] == 1.0,
            "Gemma product requires its separate LoRA at scale1.")
    LocalLlamaAdapter(LocalConfig.model_validate(gateway))._verify_loaded_adapter(observed)


def validate_deployment(reference, model, config):
    identity = gemma.evidence.verify_deployment(bound(reference).parent, model["identity"],
                                               preparation=bound(model["inference_configuration"]).parent)
    require(model["runtime_family"] == gemma.FAMILY and config == read(bound(model["inference_configuration"])),
            "Gemma deployment configuration differs from its selected model.")
    return identity


def product_identity(gateway, model):
    return {"runtime_family": "gemma26-final-separate-lora-gguf", "role": "deployed_selected", "model": "gemma26",
            "model_id": gateway["model_id"], "model_sha256": gateway["model_sha256"],
            "adapter_sha256": gateway["adapter_sha256"], "adapter_scale": gateway["adapter_scale"],
            "training_adapter_sha256": model["identity"]["adapter_sha256"], "kind": model["identity"]["kind"],
            "native_identity_sha256": digest(model["identity"])}


def validate_policy(value, model, native_run, config, tokenizer):
    import infer_deployed_task_test_actions_v3 as auxiliary
    core = auxiliary.core
    require(value.get("version") == POLICY_VERSION and value["status"] == "registered_before_test"
            and value["scope"] == "final_test" and value["role"] == "deployed_selected"
            and value["body_access_authorized"] is False and value["final_test_used_for_tuning"] is False
            and model.get("runtime_family") == gemma.FAMILY and model["role"] in {"sft", "sft_dpo"}
            and native_run["role"] == model["role"] and native_run["format"] == "json"
            and native_run["condition"] == core.conditions.condition_specs([])["canonical_full_context"],
            "Unknown Gemma selected-product evaluation policy.")
    require(value["selected_native_run_id"] == native_run["id"]
            and read(bound(value["selected_native_model"])) == model
            and value["selected_native_identity"] == model["identity"]
            and value["selected_native_identity_sha256"] == digest(model["identity"])
            and value["native_inference_configuration"] == model["inference_configuration"]
            and read(bound(value["native_inference_configuration"])) == config
            and value["native_tokenizer"] == tokenizer == gemma.tokenizer_identity(bound(model["inference_configuration"]))
            and value["plan"] == binding(q8.PLAN if value.get("numerical_regime") == q8.REGIME else PLAN)
            and value["source_sha256"] == auxiliary.sources(),
            "Gemma auxiliary model, native tokenizer, plan or implementation changed.")
    bound(value["native_core_registration"])
    validate_deployment(value["deployment"], model, config)
    directory = bound(value["deployment"]).parent
    gateway, command, report = deployment_check(directory)
    require(bound(value["gateway"]) == directory / "gateway.json" and read(bound(value["gateway"])) == gateway
            and bound(value["command"]) == directory / "command.json" and read(bound(value["command"])) == command
            and value["export_receipt"] == report["adapter_export_receipt"]
            and value["base_export_receipt"] == report["base_export_receipt"]
            and value["runtime_registry"] == report["base_registry"]
            and value["runtime_files_sha256"] == report["runtime_files_sha256"],
            "Gemma auxiliary differs from the actual prepared base/adapter deployment.")
    for key, path, expected in (("model_file", gateway["model_path"], gateway["model_sha256"]),
                                ("adapter_file", gateway["adapter_path"], gateway["adapter_sha256"]),
                                ("chat_template", gateway["chat_template_path"], gateway["chat_template_sha256"])):
        require(bound(value[key]) == Path(path).resolve() and value[key]["sha256"] == expected,
                "Gemma auxiliary file identity changed: " + key)
    require(bound(value["runtime_binary"]) == Path(command[0]).resolve()
            and gateway["chat_template_sha256"] == config["tokenizer"]["template_sha256"]
            and value["model_identity"] == product_identity(gateway, model)
            and value["numerical_regime"] == report["numerical_regime"]
            and value["numerical_regime"] in {q8.Q4_REGIME, q8.REGIME}
            and value["canonical_prefix_limit"] == 3584 and value["context_tokens"] == 4096
            and value["output_tokens"] == 512 and value["request_settings"] == auxiliary.request_settings(gateway)
            and value["request_deadline_seconds"] == 30 and value["max_attempts_per_request"] == 1
            and value["overflow_policy"] == "fail_whole_preparation"
            and all(value[k] is True for k in ("schema_constrained", "all_roles_same_case_ids",
                                               "native_prefix_parity_required", "all_request_failures_in_denominator")),
            "Gemma auxiliary decoding, product budget or numerical regime changed.")
    uri = urlsplit(gateway["endpoint"])
    require(uri.scheme == "http" and uri.hostname in {"127.0.0.1", "localhost", "::1"} and uri.port is not None
            and not any((uri.username, uri.password, uri.query, uri.fragment)) and uri.path in {"", "/"},
            "Gemma auxiliary requires its frozen loopback endpoint.")
    return gateway


def policy(context):
    import infer_deployed_task_test_actions_v3 as auxiliary
    registration, ready, role = context["registration"], context["ready"], context["selected_model"]
    model = context["models"][role]
    directory = bound(ready["selected_deployment_receipt"]).parent
    gateway, command, deployment = deployment_check(directory)
    meta = read(bound(model["preparation_manifest"]))
    value = {"version": POLICY_VERSION, "status": "registered_before_test", "scope": "final_test",
        "role": "deployed_selected", "protocol_id": ready["protocol_id"],
        "plan": binding(q8.PLAN if deployment["numerical_regime"] == q8.REGIME else PLAN),
        "native_core_registration": context["registration_binding"], "selected_native_run_id": role + "-json",
        "selected_native_model": registration["models"][role], "selected_native_identity": model["identity"],
        "selected_native_identity_sha256": digest(model["identity"]), "deployment": ready["selected_deployment_receipt"],
        "gateway": binding(directory / "gateway.json"), "export_receipt": deployment["adapter_export_receipt"],
        "base_export_receipt": deployment["base_export_receipt"],
        "native_inference_configuration": model["inference_configuration"], "native_tokenizer": meta["tokenizer"],
        "model_identity": product_identity(gateway, model), "model_file": binding(gateway["model_path"]),
        "adapter_file": binding(gateway["adapter_path"]), "chat_template": binding(gateway["chat_template_path"]),
        "runtime_binary": binding(command[0]), "command": binding(directory / "command.json"),
        "runtime_registry": deployment["base_registry"], "runtime_files_sha256": deployment["runtime_files_sha256"],
        "numerical_regime": deployment["numerical_regime"], "canonical_prefix_limit": 3584,
        "context_tokens": 4096, "output_tokens": 512, "request_settings": auxiliary.request_settings(gateway),
        "request_deadline_seconds": 30, "schema_constrained": True, "all_roles_same_case_ids": True,
        "overflow_policy": "fail_whole_preparation", "max_attempts_per_request": 1,
        "native_prefix_parity_required": True, "all_request_failures_in_denominator": True,
        "final_test_used_for_tuning": False, "source_sha256": registration["deployment_evaluation_sources_sha256"],
        "body_access_authorized": False}
    run = {"id": role + "-json", "role": role, "format": "json",
           "condition": context["conditions"]["conditions"]["canonical_full_context"]}
    validate_policy(value, model, run, read(bound(model["inference_configuration"])), meta["tokenizer"])
    return value
