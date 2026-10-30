"""Explicit inference-only Gemma expert dispatch; training configuration remains immutable."""

OVERLAY = {"version": "gemma26-native-experts-overlay-1", "experts_implementation": "grouped_mm",
           "decode_experts_implementation": "batched_mm", "attention": "sdpa", "cache": "dynamic",
           "compile": False, "batch_size": 1, "cross_request_cache": False}


def require(value, message):
    if not value:
        raise ValueError(message)


def base_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def apply(model, torch):
    require(torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 0)
            and hasattr(torch.nn.functional, "grouped_mm"), "Native CUDA grouped_mm is unavailable; no fallback.")
    base = base_model(model)
    require(base.config.text_config._attn_implementation == "sdpa", "The native attention backend changed.")
    base.set_experts_implementation("grouped_mm")
    require(base.config._experts_implementation == base.config.text_config._experts_implementation == "grouped_mm",
            "Public experts dispatch did not apply the registered inference overlay.")
    return base


class BackendTrace:
    def __init__(self, base):
        self.rows = []
        self.handle = base.register_forward_pre_hook(self.before, with_kwargs=True)

    def before(self, model, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        cache = kwargs.get("past_key_values")
        self.rows.append({"input_tokens": int(ids.shape[-1]) if ids is not None else None,
                          "experts": model.config.text_config._experts_implementation,
                          "cache_type": type(cache).__name__,
                          "cache_length_before": int(cache.get_seq_length()) if cache is not None else None})

    def validate(self, prompt_tokens, output_tokens):
        require(len(self.rows) == output_tokens and output_tokens > 0, "Missing native forward/backend evidence.")
        for index, row in enumerate(self.rows):
            require(row["experts"] == ("grouped_mm" if index == 0 else "batched_mm")
                    and row["input_tokens"] == (prompt_tokens if index == 0 else 1)
                    and row["cache_type"] == "DynamicCache", "Actual native backend/cache differs from overlay.")
        require(self.rows[0]["cache_length_before"] == 0, "Request reused populated KV state.")

    def close(self):
        self.handle.remove()
