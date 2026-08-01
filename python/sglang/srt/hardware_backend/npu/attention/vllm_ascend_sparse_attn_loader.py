from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_DSPARK_VLLM_ASCEND_SO_ENV = "SGLANG_DSPARK_VLLM_ASCEND_SO"
_REQUIRED_OPS = (
    "npu_sparse_attn_sharedkv_metadata",
    "npu_sparse_attn_sharedkv",
)
_loaded_library: Optional[Path] = None


def _missing_ops() -> list[str]:
    namespace = getattr(torch.ops, "_C_ascend", None)
    if namespace is None:
        return list(_REQUIRED_OPS)
    return [name for name in _REQUIRED_OPS if not hasattr(namespace, name)]


def vllm_ascend_sparse_attn_ops_registered() -> bool:
    """Return whether the DSpark sparse-attention ops are already registered."""
    return not _missing_ops()


def _resolve_operator_library() -> Path:
    explicit = os.environ.get(_DSPARK_VLLM_ASCEND_SO_ENV)
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(
                f"{_DSPARK_VLLM_ASCEND_SO_ENV} points to a missing file: {path}"
            )
        return path

    search_roots: list[Path] = []
    source_root = os.environ.get("VLLM_ASCEND_ROOT")
    if source_root:
        search_roots.append(Path(source_root).expanduser())
    search_roots.extend(
        Path(entry).expanduser()
        for entry in sys.path
        if isinstance(entry, str) and entry
    )

    candidates: list[Path] = []
    for root in search_roots:
        candidates.extend(root.glob("vllm_ascend/vllm_ascend_C*.so"))
        candidates.extend(root.glob("vllm_ascend_C*.so"))
    candidates = sorted({path.resolve() for path in candidates if path.is_file()})
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(
            "Found multiple vLLM-Ascend operator libraries. Set "
            f"{_DSPARK_VLLM_ASCEND_SO_ENV} to the exact library to load: "
            + ", ".join(str(path) for path in candidates)
        )
    raise RuntimeError(
        "The DSpark sparse-attention operators are not registered, and no "
        "vllm_ascend_C*.so could be found. Register the operators before "
        "initializing the DSpark worker, or set "
        f"{_DSPARK_VLLM_ASCEND_SO_ENV} to the current standalone library."
    )


def _validate_python_abi(library_path: Path) -> None:
    abi_match = re.search(r"\.cpython-(\d+)-", library_path.name)
    current_abi = f"{sys.version_info.major}{sys.version_info.minor}"
    if abi_match is not None and abi_match.group(1) != current_abi:
        raise RuntimeError(
            f"{library_path} was built for CPython {abi_match.group(1)}, but "
            f"SGLang is running CPython {current_abi}. Rebuild the extension "
            "with the SGLang Python/Torch/torch-npu environment."
        )


def initialize_vllm_ascend_sparse_attn_ops() -> Optional[Path]:
    """Register the sparse-attention ops before backend execution.

    Operator execution deliberately does not call this function. If another
    package or a future SGLang loader has already registered the two operators,
    this function is a no-op. Otherwise it retains the current standalone
    ``vllm_ascend_C*.so`` loading path as a compatibility fallback.

    Returns the fallback library path when this call loaded it, otherwise
    ``None``.
    """
    global _loaded_library

    import torch_npu  # noqa: F401  # registers the PrivateUse1/NPU dispatch

    if vllm_ascend_sparse_attn_ops_registered():
        return None
    if _loaded_library is not None:
        missing = _missing_ops()
        raise RuntimeError(
            f"Loaded {_loaded_library}, but required _C_ascend operators are "
            f"missing: {missing}."
        )

    library_path = _resolve_operator_library()
    _validate_python_abi(library_path)
    try:
        torch.ops.load_library(str(library_path))
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the standalone vLLM-Ascend operator library "
            f"{library_path}. Ensure its dependent CANN/custom-op libraries "
            "are visible through LD_LIBRARY_PATH and the Ascend OPP setup."
        ) from exc

    missing = _missing_ops()
    if missing:
        raise RuntimeError(
            f"Loaded {library_path}, but required _C_ascend operators are "
            f"missing: {missing}. The library version does not include "
            "SparseAttnSharedkv support."
        )

    _loaded_library = library_path
    logger.info(
        "Registered DSpark sparse-attention operators from %s", library_path
    )
    return library_path
