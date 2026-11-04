"""Unified checkpoint I/O for DeepSpeed, Megatron, and safetensors formats.

Supports:
  - model.safetensors (HuggingFace / safetensors format)
  - mp_rank_*_model_states.pt (Megatron-LM / DeepSpeed model states)
  - bf16_zero_pp_rank_*_optim_states.pt (DeepSpeed ZeRO optimizer states)
  - Generic .pt files with tensor dicts

Legacy readers expose FP32 views. Archive extraction below preserves original bits and dtypes.
"""

import hashlib
import io
import json
import os
import pickle
import struct
from collections import OrderedDict

import numpy as np
import torch


class TensorDict(OrderedDict):
    """Ordered dict of {name: np.ndarray(float32)} with metadata."""

    def __init__(self, *args, **kwargs):
        super(TensorDict, self).__init__(*args, **kwargs)
        self.metadata = {}  # name → {"dtype": str, "shape": tuple}

    def add_tensor(self, name, arr_f32, original_dtype, shape):
        self[name] = arr_f32
        self.metadata[name] = {"dtype": original_dtype, "shape": tuple(shape)}


def load_pt_checkpoint(filepath):
    torch = _get_torch()
    return torch.load(filepath, map_location="cpu", weights_only=False)


def save_pt_checkpoint(state, filepath):
    torch = _get_torch()
    torch.save(state, filepath, _use_new_zipfile_serialization=True, pickle_protocol=5)


def _path_step(kind, key):
    return {"kind": kind, "key": key}


def _display_child(prefix, key, indexed):
    if indexed:
        return "%s[%d]" % (prefix, key) if prefix else "[%d]" % key
    return "%s.%s" % (prefix, key) if prefix else str(key)


def iter_tensor_leaves(obj, prefix="", path=None):
    torch = _get_torch()
    if path is None:
        path = []

    if isinstance(obj, torch.Tensor):
        yield {
            "name": prefix.rstrip("."),
            "path": list(path),
            "tensor": obj,
            "shape": list(obj.shape),
            "dtype": _torch_dtype_to_str(obj.dtype),
            "numel": int(obj.numel()),
        }
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = _display_child(prefix, key, indexed=False)
            yield from iter_tensor_leaves(value, child_prefix, path + [_path_step("dict", key)])
        return

    if isinstance(obj, list):
        for index, value in enumerate(obj):
            child_prefix = _display_child(prefix, index, indexed=True)
            yield from iter_tensor_leaves(value, child_prefix, path + [_path_step("list", index)])
        return

    if isinstance(obj, tuple):
        for index, value in enumerate(obj):
            child_prefix = _display_child(prefix, index, indexed=True)
            yield from iter_tensor_leaves(value, child_prefix, path + [_path_step("tuple", index)])


def get_by_typed_path(obj, path):
    current = obj
    for step in path:
        current = current[step["key"]]
    return current


def set_by_typed_path(obj, path, value):
    if not path:
        return value

    step = path[0]
    key = step["key"]
    if step["kind"] == "dict":
        obj[key] = set_by_typed_path(obj[key], path[1:], value)
        return obj
    if step["kind"] == "list":
        obj[key] = set_by_typed_path(obj[key], path[1:], value)
        return obj
    if step["kind"] == "tuple":
        items = list(obj)
        items[key] = set_by_typed_path(items[key], path[1:], value)
        return tuple(items)
    raise ValueError("unknown path step kind: %s" % step["kind"])


def tensor_to_float32_array(tensor):
    return _tensor_to_float32(tensor)


def float32_array_to_tensor(arr_f32, dtype_str, shape):
    torch = _get_torch()
    shaped = np.asarray(arr_f32, dtype=np.float32).reshape(shape)
    tensor = torch.from_numpy(shaped.copy())
    if dtype_str == "BF16":
        return tensor.to(dtype=torch.bfloat16)
    if dtype_str == "F16":
        return tensor.to(dtype=torch.float16)
    return tensor.to(dtype=torch.float32)


def bytes_to_uint8_tensor(data):
    torch = _get_torch()
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()


def uint8_tensor_to_bytes(tensor):
    return bytes(tensor.detach().cpu().numpy().tobytes())


def _to_float32(raw_bytes, dtype_str, shape):
    """Convert raw bytes to flat float32 numpy array."""
    if dtype_str in ("F32", "float32"):
        return np.frombuffer(raw_bytes, dtype=np.float32).ravel().copy()
    elif dtype_str in ("BF16", "bfloat16"):
        u16 = np.frombuffer(raw_bytes, dtype=np.uint16).ravel()
        u32 = u16.astype(np.uint32) << np.uint32(16)
        return u32.view(np.float32).copy()
    elif dtype_str in ("F16", "float16"):
        return np.frombuffer(raw_bytes, dtype=np.float16).ravel().astype(np.float32).copy()
    else:
        raise ValueError("Unsupported dtype: %s" % dtype_str)


def _torch_dtype_to_str(dtype):
    """Convert torch.dtype to string."""
    dtype_map = {
        "torch.float32": "F32",
        "torch.bfloat16": "BF16",
        "torch.float16": "F16",
        "torch.float": "F32",
        "torch.half": "F16",
    }
    key = str(dtype)
    return dtype_map.get(key, key)


def _tensor_to_float32(tensor):
    """Convert a torch tensor to flat float32 numpy array.

    Handles tensors with requires_grad=True by explicitly detaching and
    clearing gradient tracking before converting to numpy. This is necessary
    for full DeepSpeed/Megatron checkpoints where model parameters retain
    requires_grad from training.
    """
    torch = _get_torch()
    safe_tensor = tensor.detach().cpu()
    if safe_tensor.requires_grad:
        safe_tensor = safe_tensor.clone()
        safe_tensor.requires_grad_(False)
    dtype_str = _torch_dtype_to_str(safe_tensor.dtype)
    if dtype_str == "BF16":
        f32_tensor = safe_tensor.float()
        arr = f32_tensor.numpy().ravel().copy()
        return arr, dtype_str
    elif dtype_str == "F16":
        return safe_tensor.numpy().ravel().astype(np.float32).copy(), dtype_str
    else:
        return safe_tensor.float().numpy().ravel().copy(), dtype_str


_torch = None

def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def read_safetensors(filepath):
    """Read a .safetensors file into a TensorDict.

    Supports F32, BF16, F16 dtypes. No external dependencies beyond numpy.
    """
    result = TensorDict()
    with open(filepath, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
        data_offset = 8 + header_len

        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype_str = meta["dtype"]
            shape = meta["shape"]
            start, end = meta["data_offsets"]
            fh.seek(data_offset + start)
            raw = fh.read(end - start)
            arr_f32 = _to_float32(raw, dtype_str, shape)
            result.add_tensor(name, arr_f32, dtype_str, shape)

    return result


def read_pt_model_states(filepath):
    """Read a Megatron/DeepSpeed model_states .pt file.

    Handles:
      - dict with 'module' key containing tensor dict (Megatron/DS)
      - dict directly containing tensors
      - Nested state dicts
    """
    torch = _get_torch()
    state = torch.load(filepath, map_location="cpu", weights_only=False)

    result = TensorDict()

    tensor_dict = _extract_tensor_dict(state)

    for name, tensor in tensor_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.numel() == 0:
            continue
        arr_f32, dtype_str = _tensor_to_float32(tensor)
        result.add_tensor(name, arr_f32, dtype_str, list(tensor.shape))

    return result


def read_pt_optim_states(filepath):
    """Read a DeepSpeed ZeRO optimizer states .pt file.

    Optimizer states contain exp_avg, exp_avg_sq, etc. per parameter.
    These are typically fp32 even when the model is bf16.
    """
    torch = _get_torch()

    try:
        import deepspeed  # noqa: F401 -- legacy reader requires its pickle classes
        state = torch.load(filepath, map_location="cpu", weights_only=False)
    except ImportError:
        state = _load_pt_safe(filepath)

    result = TensorDict()
    _flatten_optim_state(state, result, prefix="")
    return result


def read_pt_generic(filepath):
    """Read a generic .pt file, auto-detecting structure."""
    torch = _get_torch()
    state = torch.load(filepath, map_location="cpu", weights_only=False)

    result = TensorDict()

    if isinstance(state, torch.Tensor):
        arr_f32, dtype_str = _tensor_to_float32(state)
        result.add_tensor("tensor", arr_f32, dtype_str, list(state.shape))
        return result

    if isinstance(state, dict):
        if "module" in state:
            return read_pt_model_states(filepath)
        if "optimizer_state_dict" in state or "exp_avg" in str(state.keys()):
            return read_pt_optim_states(filepath)
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                arr_f32, dtype_str = _tensor_to_float32(value)
                result.add_tensor(name, arr_f32, dtype_str, list(value.shape))
            elif isinstance(value, dict):
                for sub_name, sub_val in value.items():
                    if isinstance(sub_val, torch.Tensor) and sub_val.numel() > 0:
                        full_name = "%s.%s" % (name, sub_name)
                        arr_f32, dtype_str = _tensor_to_float32(sub_val)
                        result.add_tensor(full_name, arr_f32, dtype_str, list(sub_val.shape))

    return result


def read_checkpoint(filepath):
    """Auto-detect format and read a checkpoint file into TensorDict.

    Supported formats:
      - *.safetensors → safetensors reader
      - *model_states*.pt → Megatron/DeepSpeed model states
      - *optim_states*.pt → DeepSpeed optimizer states
      - *.pt → generic PyTorch loader
    """
    basename = os.path.basename(filepath).lower()

    if filepath.endswith(".safetensors"):
        return read_safetensors(filepath)
    elif "optim_states" in basename:
        return read_pt_optim_states(filepath)
    elif "model_states" in basename:
        return read_pt_model_states(filepath)
    elif filepath.endswith(".pt") or filepath.endswith(".pth"):
        return read_pt_generic(filepath)
    else:
        raise ValueError("Unsupported checkpoint format: %s" % filepath)


def _extract_tensor_dict(state):
    """Extract the flat tensor dict from various checkpoint wrappers."""
    torch = _get_torch()

    if isinstance(state, dict):
        if "module" in state and isinstance(state["module"], dict):
            return state["module"]
        has_tensors = any(isinstance(v, torch.Tensor) for v in state.values())
        if has_tensors:
            return state
        for value in state.values():
            if isinstance(value, dict):
                inner_has_tensors = any(isinstance(v, torch.Tensor) for v in value.values())
                if inner_has_tensors:
                    return value

    return {}


def _load_pt_safe(filepath):
    """Load a .pt file without requiring custom classes (e.g., deepspeed)."""
    import io
    import pickle
    import zipfile

    torch = _get_torch()

    class SafeUnpickler(pickle.Unpickler):
        """Unpickler that replaces unknown classes with plain dicts/lists."""
        def find_class(self, module, name):
            try:
                return super(SafeUnpickler, self).find_class(module, name)
            except (ImportError, AttributeError):
                # Return a factory that creates a dict-like object
                return type(name, (dict,), {})

    with zipfile.ZipFile(filepath) as zf:
        pkl_names = [n for n in zf.namelist() if n.endswith("data.pkl")]
        if not pkl_names:
            raise ValueError("No data.pkl found in %s" % filepath)
        pkl_data = zf.read(pkl_names[0])
        return SafeUnpickler(io.BytesIO(pkl_data)).load()


def _flatten_optim_state(state, result, prefix):
    """Recursively extract tensors from optimizer state dicts."""
    torch = _get_torch()

    if isinstance(state, torch.Tensor):
        if state.numel() > 0:
            arr_f32, dtype_str = _tensor_to_float32(state)
            name = prefix.rstrip(".")
            result.add_tensor(name, arr_f32, dtype_str, list(state.shape))
        return

    if isinstance(state, dict):
        for key, value in state.items():
            new_prefix = "%s.%s" % (prefix, key) if prefix else key
            _flatten_optim_state(value, result, new_prefix)
        return

    if isinstance(state, (list, tuple)):
        for idx, value in enumerate(state):
            new_prefix = "%s[%d]" % (prefix, idx)
            _flatten_optim_state(value, result, new_prefix)


def discover_checkpoints(directory):
    """Discover checkpoint files in a directory tree.

    Returns list of (filepath, category) tuples where category is one of:
      'safetensors', 'model_states', 'optim_states', 'other'
    """
    results = []
    for root, dirs, files in os.walk(directory):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            if fname.endswith(".safetensors"):
                results.append((fpath, "safetensors"))
            elif "model_states" in fname and fname.endswith(".pt"):
                results.append((fpath, "model_states"))
            elif "optim_states" in fname and fname.endswith(".pt"):
                results.append((fpath, "optim_states"))
            elif fname.endswith(".pt") or fname.endswith(".pth"):
                results.append((fpath, "other"))
    return results


# Bit-preserving archive metadata and tensor interface.

def raw_view(tensor):
    if (
        type(tensor) not in (torch.Tensor, torch.nn.Parameter)
        or tensor.layout != torch.strided
        or tensor.is_quantized
    ):
        raise ValueError("ALC currently supports dense, unquantized tensors")
    array = (
        tensor.detach()
        .cpu()
        .resolve_conj()
        .resolve_neg()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
    )
    return memoryview(array).cast("B")


def array(tensor):
    safe = tensor.detach().cpu().contiguous().reshape(-1)
    if tensor.dtype == torch.bfloat16:
        return safe.view(torch.int16).numpy().view(np.uint16)
    if tensor.dtype in (torch.float32, torch.float64):
        return safe.numpy()
    return np.frombuffer(raw_view(safe), np.uint8)


def extract(state, hash_tensors=True):
    entries, by_object, containers = {}, {}, set()

    def register(tensor, path):
        if id(tensor) in by_object:
            return by_object[id(tensor)]
        raw = raw_view(tensor)
        key = hashlib.sha256(pickle.dumps(path, protocol=4)).hexdigest()
        spec = {
            "key": key,
            "name": ".".join(str(x[1]) for x in path) or "tensor",
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
            "original_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest() if hash_tensors else None,
            "requires_grad": tensor.requires_grad,
            "parameter": isinstance(tensor, torch.nn.Parameter),
        }
        entries[key] = {"spec": spec, "tensor": tensor, "path": path}
        by_object[id(tensor)] = key
        return key

    def walk(obj, path):
        if isinstance(obj, torch.Tensor):
            register(obj, path)
        elif isinstance(obj, (dict, list, tuple)) and id(obj) not in containers:
            containers.add(id(obj))
            children = obj.items() if isinstance(obj, dict) else enumerate(obj)
            for key, value in children:
                walk(
                    value,
                    path + [("dict" if isinstance(obj, dict) else "sequence", key)],
                )

    walk(state, [])

    class Writer(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, torch.Tensor):
                key = by_object.get(id(obj))
                if key is None:
                    key = register(obj, [("object_tensor", len(entries))])
                return ("ALC_TENSOR", key)
            return None

    output = io.BytesIO()
    Writer(output, protocol=5).dump(state)
    return output.getvalue(), entries


def rebuild(template, load_tensor):
    class Reader(pickle.Unpickler):
        def persistent_load(self, reference):
            if (
                not isinstance(reference, tuple)
                or len(reference) != 2
                or reference[0] != "ALC_TENSOR"
            ):
                raise ValueError("Invalid tensor metadata reference")
            return load_tensor(reference[1])

    return Reader(io.BytesIO(template)).load()


def from_bytes(data, spec):
    dtype = getattr(torch, spec["dtype"])
    if len(data) != spec["original_bytes"]:
        raise ValueError("Tensor byte count mismatch")
    storage = data if isinstance(data, bytearray) else bytearray(data)
    tensor = (
        torch.frombuffer(storage, dtype=dtype).reshape(spec["shape"])
        if data
        else torch.empty(spec["shape"], dtype=dtype)
    )
    if spec["parameter"]:
        return torch.nn.Parameter(tensor, requires_grad=spec["requires_grad"])
    if spec["requires_grad"]:
        tensor.requires_grad_(True)
    return tensor
