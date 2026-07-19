# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os


_FALSEY_ENV_VALUES = {"", "0", "false", "no", "off"}


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in _FALSEY_ENV_VALUES


def _hf_env_repr() -> str:
    """Human-readable HF cache env (for log lines)."""
    hf_home = os.environ.get("HF_HOME")
    hf_hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
    return f"HF_HOME={hf_home} HUGGINGFACE_HUB_CACHE={hf_hub}"


_COSMOS_REASON2_REPO = "nvidia/Cosmos-Reason2-2B"
# Well-known local NVMe HF snapshot used for open-loop / server-client inference
# when the training-server path (/sh/ycb/...) is not available.
_DEFAULT_LOCAL_COSMOS_SNAPSHOT = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "huggingface",
    "hub",
    f"models--{_COSMOS_REASON2_REPO.replace('/', '--')}",
    "snapshots",
    "9ce19a195e423419c349abfc86fd07178b230561",
)


def _latest_cosmos_snapshot(hf_hub_cache: str) -> str | None:
    snap_root = os.path.join(
        hf_hub_cache,
        f"models--{_COSMOS_REASON2_REPO.replace('/', '--')}",
        "snapshots",
    )
    if not os.path.isdir(snap_root):
        return None
    snaps = sorted(
        os.path.join(snap_root, name)
        for name in os.listdir(snap_root)
        if os.path.isdir(os.path.join(snap_root, name))
    )
    return snaps[-1] if snaps else None


def resolve_cosmos_reason2_path() -> str | None:
    """Locate a Cosmos-Reason2-2B directory without rewriting checkpoints.

    Intended split (env overrides; no shell script required for inference):

    - **Training** (``train.sh`` on server): ``COSMOS_REASON2_PATH`` /
      ``HF_HUB_CACHE`` under ``/sh/ycb/.cache/huggingface/...``
    - **Local open-loop / server-client**: ``LOCAL_COSMOS_REASON2_PATH`` or
      ``COSMOS_REASON2_PATH``, else HF cache, else the well-known NVMe snapshot
      under ``~/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/...``

    Resolution order:
    1. ``COSMOS_REASON2_PATH`` / ``LOCAL_COSMOS_REASON2_PATH`` if directory exists
    2. Latest snapshot under ``HF_HUB_CACHE`` / ``HUGGINGFACE_HUB_CACHE`` /
       ``$HF_HOME/hub``
    3. Default local snapshot ``~/.cache/huggingface/.../9ce19a...`` (if present)
    4. Latest snapshot under ``~/.cache/huggingface/hub``

    Returns None when nothing local is found (caller may fall back to Hub id).
    """
    for env_name in ("COSMOS_REASON2_PATH", "LOCAL_COSMOS_REASON2_PATH"):
        env_path = os.environ.get(env_name, "").strip()
        if env_path and os.path.isdir(env_path):
            return env_path

    hf_hub_candidates: list[str] = []
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(key)
        if value:
            hf_hub_candidates.append(value)
    if os.environ.get("HF_HOME"):
        hf_hub_candidates.append(os.path.join(os.environ["HF_HOME"], "hub"))
    # Always consider the user home HF cache (local inference default).
    hf_hub_candidates.append(
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    )

    seen: set[str] = set()
    for hf_hub_cache in hf_hub_candidates:
        if not hf_hub_cache or hf_hub_cache in seen:
            continue
        seen.add(hf_hub_cache)
        found = _latest_cosmos_snapshot(hf_hub_cache)
        if found:
            return found

    if os.path.isdir(_DEFAULT_LOCAL_COSMOS_SNAPSHOT):
        return _DEFAULT_LOCAL_COSMOS_SNAPSHOT
    return None


def _maybe_remap_missing_cosmos_path(name_str: str) -> str | None:
    """Remap a missing absolute Cosmos path baked into a checkpoint config.

    Fine-tuned checkpoints may embed a server-only absolute path such as
    ``/sh/ycb/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/...``.
    When that path is absent on the current machine, resolve a local copy via
    ``resolve_cosmos_reason2_path()`` instead of treating the absolute path as a
    Hub repo id (which raises HFValidationError).
    """
    if not name_str.startswith("/") or os.path.isdir(name_str):
        return None
    if "Cosmos-Reason2" not in name_str:
        return None
    return resolve_cosmos_reason2_path()


def _torch_dtype_from_arg(dtype):
    """Resolve a transformers dtype argument into a torch dtype, if concrete."""
    if dtype is None or dtype == "auto":
        return None

    try:
        import torch
    except Exception:
        return None

    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        name = dtype.removeprefix("torch.")
        candidate = getattr(torch, name, None)
        if isinstance(candidate, torch.dtype):
            return candidate
    return None


def _zero_no_weight_model_parameters(model) -> None:
    """Replace no-init tensor contents with deterministic finite values."""
    try:
        import torch
    except Exception:
        return

    parameters = getattr(model, "parameters", None)
    buffers = getattr(model, "buffers", None)
    tensors = []
    if callable(parameters):
        tensors.extend(parameters())
    if callable(buffers):
        tensors.extend(buffers())

    with torch.no_grad():
        for tensor in tensors:
            if getattr(tensor, "device", None) is not None and tensor.device.type == "meta":
                continue
            try:
                tensor.zero_()
            except Exception:
                pass


def _hf_no_weight_model_call(orig_func, klass, pretrained_model_name_or_path, *args, **kwargs):
    """Instantiate a HF model from config without resolving checkpoint weight files.

    This is intentionally test-only and gated by ``GROOT_SKIP_HF_MODEL_WEIGHTS``.
    It keeps the architecture, processor/config loading, and model code paths
    alive while avoiding multi-GB safetensor reads in tests that only need shape
    and integration coverage.
    """
    if kwargs.get("state_dict") is not None:
        return orig_func(klass, pretrained_model_name_or_path, *args, **kwargs)

    import copy

    import torch
    from transformers import PretrainedConfig
    from transformers.modeling_utils import no_init_weights

    name_str = str(pretrained_model_name_or_path)
    output_loading_info = kwargs.pop("output_loading_info", False)
    config = kwargs.pop("config", None)
    requested_dtype = kwargs.pop("dtype", None)
    requested_torch_dtype = kwargs.pop("torch_dtype", None)
    if requested_dtype is None:
        requested_dtype = requested_torch_dtype
    attn_implementation = kwargs.pop("attn_implementation", None)

    loader_kwargs = {}
    config_loader_keys = {
        "cache_dir",
        "force_download",
        "proxies",
        "local_files_only",
        "token",
        "revision",
        "subfolder",
    }
    loader_only_keys = {
        "_commit_hash",
        "_fast_init",
        "_from_auto",
        "_from_pipeline",
        "adapter_kwargs",
        "adapter_name",
        "device_map",
        "distributed_config",
        "from_flax",
        "from_tf",
        "generation_config",
        "gguf_file",
        "ignore_mismatched_sizes",
        "key_mapping",
        "load_in_4bit",
        "load_in_8bit",
        "low_cpu_mem_usage",
        "max_memory",
        "mirror",
        "offload_buffers",
        "offload_folder",
        "offload_state_dict",
        "quantization_config",
        "resume_download",
        "state_dict",
        "tp_plan",
        "tp_size",
        "trust_remote_code",
        "use_auth_token",
        "use_kernels",
        "use_safetensors",
        "variant",
        "weights_only",
    }
    for key in sorted(config_loader_keys | loader_only_keys):
        if key in kwargs:
            loader_kwargs[key] = kwargs.pop(key)

    token = loader_kwargs.get("token")
    use_auth_token = loader_kwargs.get("use_auth_token")
    if token is None and use_auth_token is not None:
        loader_kwargs["token"] = use_auth_token

    if not isinstance(config, PretrainedConfig):
        config_path = config if config is not None else pretrained_model_name_or_path
        config_kwargs = {k: loader_kwargs[k] for k in config_loader_keys if k in loader_kwargs}
        config, model_kwargs = klass.config_class.from_pretrained(
            config_path,
            return_unused_kwargs=True,
            **config_kwargs,
            **kwargs,
        )
    else:
        config = copy.deepcopy(config)
        model_kwargs = kwargs

    if attn_implementation is not None:
        config._attn_implementation = attn_implementation

    torch_dtype = _torch_dtype_from_arg(requested_dtype)
    previous_dtype = None
    if torch_dtype is not None:
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch_dtype)
    try:
        with no_init_weights():
            model = klass(config, *args, **model_kwargs)
    finally:
        if previous_dtype is not None:
            torch.set_default_dtype(previous_dtype)

    if torch_dtype is not None:
        model.to(dtype=torch_dtype)
    _zero_no_weight_model_parameters(model)
    model.eval()
    print(f"[groot/hf] skip model weights: {name_str} | {_hf_env_repr()}", flush=True)

    if output_loading_info:
        return model, {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        }
    return model


def _hf_local_first_call(
    orig_func,
    klass,
    pretrained_model_name_or_path,
    *args,
    skip_model_weights: bool = False,
    **kwargs,
):
    """Invoke ``orig_func`` (an unwrapped from_pretrained) preferring the local cache.

    Strategy:

    1. Local filesystem path → call ``orig_func`` unchanged.
    2. Caller already passed ``local_files_only=True`` → honor it.
    3. Otherwise probe the cache by calling ``orig_func`` with
       ``local_files_only=True`` first.  If it succeeds, the cache had every
       file ``from_pretrained`` needed — no network traffic at all.  If it
       raises (cache empty, partial snapshot, missing shard), fall through
       to a normal call which lets HF Hub download the missing pieces.

    This mirrors exactly what ``transformers.from_pretrained`` itself uses
    to decide cache hit vs miss, so we never get a false miss from a
    stricter probe (the original implementation called
    ``snapshot_download(local_files_only=True)``, which requires a fully
    populated ``refs/main`` + ``snapshots/<commit>/`` tree that
    ``from_pretrained`` does not always create — observed to cause 100%
    false-miss rate in CI Job 308778931).
    """
    if skip_model_weights and _env_flag_enabled("GROOT_SKIP_HF_MODEL_WEIGHTS"):
        return _hf_no_weight_model_call(
            orig_func, klass, pretrained_model_name_or_path, *args, **kwargs
        )

    name_str = str(pretrained_model_name_or_path)
    if os.path.isdir(name_str):
        print(f"[groot/hf] local path: {name_str} | {_hf_env_repr()}", flush=True)
        return orig_func(klass, pretrained_model_name_or_path, *args, **kwargs)

    remapped = _maybe_remap_missing_cosmos_path(name_str)
    if remapped is not None:
        print(
            f"[groot/hf] remapped missing Cosmos path: {name_str} -> {remapped} | {_hf_env_repr()}",
            flush=True,
        )
        pretrained_model_name_or_path = remapped
        name_str = remapped
        if os.path.isdir(name_str):
            return orig_func(klass, pretrained_model_name_or_path, *args, **kwargs)
    elif name_str.startswith("/") and "Cosmos-Reason2" in name_str:
        raise FileNotFoundError(
            f"Cosmos backbone path from checkpoint does not exist: {name_str}\n"
            "Set COSMOS_REASON2_PATH to a local Cosmos-Reason2-2B directory, "
            "or populate HF_HUB_CACHE with nvidia/Cosmos-Reason2-2B. "
            "Server-side train.sh paths are unchanged."
        )

    if kwargs.get("local_files_only", False):
        return orig_func(klass, pretrained_model_name_or_path, *args, **kwargs)
    try:
        result = orig_func(
            klass,
            pretrained_model_name_or_path,
            *args,
            **{**kwargs, "local_files_only": True},
        )
        print(f"[groot/hf] cache hit: {name_str} | {_hf_env_repr()}", flush=True)
        return result
    except Exception:
        print(
            f"[groot/hf] cache miss (will download): {name_str} | {_hf_env_repr()}",
            flush=True,
        )
        return orig_func(klass, pretrained_model_name_or_path, *args, **kwargs)


def _patch_hf_local_first() -> None:
    """Patch from_pretrained to prefer the local HF cache over network calls.

    When a HF repo ID is passed we first invoke ``from_pretrained`` with
    ``local_files_only=True``; on any error we fall through to a normal
    download.  This avoids the per-file etag roundtrip that
    ``transformers.from_pretrained`` does on every load and the 429
    rate-limit storm when many CI jobs run concurrently.

    Covers: PreTrainedModel, PretrainedConfig, ProcessorMixin, AutoConfig,
    AutoProcessor — every transformers from_pretrained entrypoint.

    Triggered by GROOT_HF_LOCAL_FIRST (set by conftest.py, survives uv run) or
    PYTEST_CURRENT_TEST (set automatically by pytest).
    """

    def _wrap(cls: type) -> None:
        if "from_pretrained" not in cls.__dict__:
            return
        original = cls.from_pretrained
        if getattr(original, "_groot_hf_local_patched", False):
            return

        orig_func = original.__func__

        @classmethod  # type: ignore[misc]
        def patched(klass, pretrained_model_name_or_path, *args, **kwargs):
            return _hf_local_first_call(
                orig_func,
                klass,
                pretrained_model_name_or_path,
                *args,
                skip_model_weights=cls.__name__ == "PreTrainedModel",
                **kwargs,
            )

        patched._groot_hf_local_patched = True  # type: ignore[attr-defined]
        cls.from_pretrained = patched

    try:
        import transformers as _transformers

        for _attr in (
            "PreTrainedModel",
            "PretrainedConfig",
            "ProcessorMixin",
            "AutoConfig",
            "AutoProcessor",
        ):
            _cls = getattr(_transformers, _attr, None)
            if _cls is not None:
                _wrap(_cls)
    except Exception:
        pass


def _patch_mistral() -> None:
    """Suppress 429 / connection errors / hangs from the HuggingFace Hub in mistral regex patching.

    transformers calls model_info() inside a nested is_base_mistral() function
    unconditionally even when loading from a fully local checkpoint. Qwen3VL /
    Cosmos is never Mistral, so returning the tokenizer unchanged on any network
    failure is correct.

    NOTE: is_base_mistral is a *nested* function inside _patch_mistral_regex, so
    it is not accessible as a module-level attribute — we must wrap the classmethod.

    The wrapper short-circuits before _orig for two cases that should never need
    a network roundtrip:

    1. Local filesystem path — can't be a Hub repo ID at all.
    2. Repo ID that doesn't even mention "mistral" — the underlying check exists
       solely to recognize Mistral-derived tokenizers; on any other ID the
       network call is wasted at best.

    The short-circuit also closes a hang the surrounding ``except Exception``
    cannot catch: when a TCP connection to huggingface.co succeeds but the
    server stops sending bytes, ``socket.recv_into()`` blocks in the kernel
    with no Python-level exception, eventually breaching pytest-timeout (e.g.
    job 309584593: ``test_trt_full_pipeline[1]`` hung 600 s here while loading
    the tokenizer for ``nvidia/Cosmos-Reason2-2B``).

    Triggered by GROOT_PATCH_MISTRAL (set by conftest.py, survives uv run) or
    PYTEST_CURRENT_TEST (set automatically by pytest, belt-and-suspenders).
    """
    try:
        import transformers.tokenization_utils_base as _tub

        _cls = _tub.PreTrainedTokenizerBase
        _orig = _cls._patch_mistral_regex.__func__
        if getattr(_orig, "_groot_patched", False):
            return

        def _safe(cls, tokenizer, pretrained_model_name_or_path, **kwargs):
            name_str = str(pretrained_model_name_or_path)
            if os.path.isdir(name_str) or "mistral" not in name_str.lower():
                return tokenizer
            try:
                return _orig(cls, tokenizer, pretrained_model_name_or_path, **kwargs)
            except Exception:
                return tokenizer

        _safe._groot_patched = True  # type: ignore[attr-defined]
        _cls._patch_mistral_regex = classmethod(_safe)
    except Exception:
        pass


if (
    os.environ.get("PYTEST_CURRENT_TEST")
    or os.environ.get("GROOT_HF_LOCAL_FIRST")
    or os.environ.get("HF_HUB_OFFLINE") == "1"
    or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
):
    _patch_hf_local_first()

if (
    os.environ.get("PYTEST_CURRENT_TEST")
    or os.environ.get("GROOT_PATCH_MISTRAL")
    or os.environ.get("HF_HUB_OFFLINE") == "1"
    or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
):
    _patch_mistral()
