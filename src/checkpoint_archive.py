"""Reverse-chain archives with Direct group anchors and a single Raw head.

Inputs are ordered oldest to newest. Each input is a trusted PyTorch checkpoint
file or an in-memory checkpoint object. No training model, server path, checkpoint
count, rank count, or experiment configuration is embedded in the implementation.
"""

import hashlib
import json
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import bf16
import checkpoint_io as _tree
import tensor_compress as codec

FORMAT = "ALC_CHECKPOINT_ARCHIVE_1"
MASTER_KEYS = {
    "master",
    "masters",
    "fp32",
    "fp32_master",
    "fp32_master_weights",
    "master_weights",
    "fp32_from_fp16_params",
    "fp32_groups",
    "fp32_flat_groups",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_checkpoint(source, mmap=False):
    if isinstance(source, (str, os.PathLike)):
        if mmap:
            try:
                return torch.load(
                    source, map_location="cpu", weights_only=False, mmap=True
                )
            except (RuntimeError, TypeError) as exc:
                if "mmap" not in str(exc):
                    raise
        return torch.load(source, map_location="cpu", weights_only=False)
    return source


def anchor(version, group_size, count):
    if not 0 <= version < count or group_size < 1:
        raise ValueError("Invalid version or group size")
    if version == count - 1 or group_size >= count:
        return count - 1
    # Match the paper diagram: if the last group contains the newest version,
    # its anchor is that version's Raw, without an extra historical anchor.
    return min((version // group_size + 1) * group_size - 1, count - 1)


def _lookup(entries, name):
    if name in entries:
        return entries[name]
    matches = [entry for entry in entries.values() if entry["spec"]["name"] == name]
    if len(matches) != 1:
        raise ValueError("Tensor name is missing or ambiguous: " + str(name))
    return matches[0]


def _derivations(entries, pairs):
    """A name/offset proposes a pairing; only a full bitwise match permits dedup."""
    result = {}
    for key, entry in entries.items():
        spec = entry["spec"]
        if spec["dtype"] != "bfloat16" or not spec["numel"]:
            continue
        candidates = []
        explicit = pairs.get(spec["name"], pairs.get(key))
        if explicit is not None:
            source_name = explicit if isinstance(explicit, str) else explicit["source"]
            offset = 0 if isinstance(explicit, str) else explicit.get("start", 0)
            candidates.append((_lookup(entries, source_name), offset))
        else:
            leaf = entry["path"][-1][1] if entry["path"] else None
            for candidate in entries.values():
                if (
                    candidate["spec"]["dtype"] == "float32"
                    and candidate["path"]
                    and candidate["path"][-1][1] == leaf
                    and any(
                        step[1] in MASTER_KEYS
                        for step in candidate["path"]
                        if isinstance(step[1], str)
                    )
                    and candidate["spec"]["shape"] == spec["shape"]
                ):
                    candidates.append((candidate, 0))
        for candidate, offset in candidates:
            if (
                candidate["spec"]["dtype"] != "float32"
                or type(offset) is not int
                or offset < 0
                or offset + spec["numel"] > candidate["spec"]["numel"]
            ):
                raise ValueError("Invalid explicit BF16 master span")
            master = _tree.array(candidate["tensor"])[offset : offset + spec["numel"]]
            rule = bf16.verified_rule(master, bytes(_tree.raw_view(entry["tensor"])))
            if rule is not None:
                result[key] = {
                    "source_key": candidate["spec"]["key"],
                    "start": offset,
                    "rule": rule,
                }
                break
    return result


def _encode_one(target, reference, path, raw=False, byte_stream=False):
    source = target.tobytes()
    frame = (
        source
        if raw
        else codec.require_native().tans_encode(source, 8)
        if byte_stream
        else codec.encode_frame(target, reference, path)
    )
    restored = (
        source
        if raw
        else codec.require_native().tans_decode(frame)
        if byte_stream
        else codec.decode_frame(frame, reference).tobytes()
    )
    if restored != source:
        raise ValueError("Build-time bitwise round trip failed")
    return frame


def compress(
    checkpoints,
    output,
    group_size=None,
    chunk_elems=2**18,
    sample_size=65536,
    compression_workers=8,
    compression_memory_bytes=512 * 2**20,
    bf16_pairs=None,
    retain_latest_raw=True,
):
    """Create an immutable archive; verify every frame before accepting it.

    bf16_pairs maps BF16 names to FP32 master names, or to {source, start} spans.
    A failed derivation is encoded through I-Align Direct without float casting.
    The optimized adaptive FP32/FP64 path is selected once per tensor.
    group_size=None disables grouping by using the entire input cohort.
    """
    if not checkpoints:
        raise ValueError("A nonempty oldest-to-newest sequence is required")
    if group_size is None:
        group_size = len(checkpoints)
    if type(group_size) is not int or group_size < 1:
        raise ValueError(
            "A nonempty oldest-to-newest sequence and positive group size are required"
        )
    if type(chunk_elems) is not int or not 1 <= chunk_elems < 2**32:
        raise ValueError("Invalid codec chunk size")
    if type(compression_workers) is not int or compression_workers < 1:
        raise ValueError("Positive compression worker count required")
    if not 1 <= sample_size <= 65536 or compression_memory_bytes < 1:
        raise ValueError("Invalid sampling or compression memory budget")
    if not retain_latest_raw and len(checkpoints) != 1:
        raise ValueError("Standalone Direct mode accepts exactly one checkpoint")
    codec.require_native()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    count = len(checkpoints)
    manifest = {
        "format": FORMAT,
        "group_size": group_size,
        "chunk_elems": chunk_elems,
        "sample_size": sample_size,
        "table_log": 8,
        "latest": count - 1,
        "latest_raw": bool(retain_latest_raw),
        "versions": [None] * count,
    }
    root_specs = None
    head_entries = None
    reference_entries = {}
    for version in reversed(range(count)):
        state = load_checkpoint(checkpoints[version])
        template, entries = _tree.extract(state)
        latest = version == count - 1
        raw_latest = latest and retain_latest_raw
        if raw_latest:
            if isinstance(checkpoints[version], (str, os.PathLike)):
                shutil.copyfile(checkpoints[version], output / "latest.pt")
            else:
                torch.save(state, output / "latest.pt")
            manifest["latest_file_sha256"] = file_sha(output / "latest.pt")
        else:
            (output / f"v{version:06d}.metadata.pkl").write_bytes(template)
        derived = {} if raw_latest else _derivations(entries, bf16_pairs or {})
        current = {"template_sha256": sha(template), "tensors": {}}
        direct_anchor = not raw_latest and anchor(version, group_size, count) == version
        pack_path = output / f"v{version:06d}.pack"
        with (
            pack_path.open("wb") as pack,
            ThreadPoolExecutor(max_workers=compression_workers) as pool,
        ):
            for key, entry in entries.items():
                spec = dict(entry["spec"])
                current["tensors"][key] = spec
                if key in derived:
                    spec.update(mode="DerivedBF16", derivation=derived[key], chunks=[])
                    continue
                frozen = (
                    not latest
                    and key in root_specs
                    and all(
                        spec[field] == root_specs[key][field]
                        for field in ("dtype", "shape", "sha256")
                    )
                )
                if frozen:
                    if head_entries is None:
                        _, head_entries = _tree.extract(
                            load_checkpoint(output / "latest.pt", mmap=True),
                            hash_tensors=False,
                        )
                    frozen = np.array_equal(
                        np.frombuffer(_tree.raw_view(entry["tensor"]), np.uint8),
                        np.frombuffer(
                            _tree.raw_view(head_entries[key]["tensor"]), np.uint8
                        ),
                    )
                reference = reference_entries.get(key)
                if reference is not None and (
                    reference["spec"]["dtype"] != spec["dtype"]
                    or reference["spec"]["shape"] != spec["shape"]
                ):
                    reference = None
                target = _tree.array(entry["tensor"])
                adaptive = spec["dtype"] in ("float32", "float64")
                coding = adaptive or spec["dtype"] == "bfloat16"
                base = (
                    _tree.array(reference["tensor"])
                    if reference is not None and adaptive and not direct_anchor
                    else None
                )
                choice = (
                    codec.choose_path(target, base, sample_size)
                    if coding
                    else {"path": "Bytes", "estimated_bits": {}}
                )
                path = (
                    "Raw" if raw_latest else "FrozenRef" if frozen else choice["path"]
                )
                spec.update(
                    mode=path,
                    chunks=[],
                    selection={
                        **choice,
                        "estimated_bits": {
                            name: value if math.isfinite(value) else None
                            for name, value in choice["estimated_bits"].items()
                        },
                    },
                )
                if path in ("FM-Delta*", "RotDelta"):
                    spec["reference_version"] = version + 1
                elif frozen:
                    spec["reference_version"] = count - 1
                if raw_latest:
                    spec["selection"] = {"path": "Raw"}
                elif direct_anchor and not frozen:
                    spec["selection"]["forced_group_anchor"] = True
                fragments = []
                for target_key, derivation in derived.items():
                    if derivation["source_key"] == key:
                        fragments.append((target_key, derivation, entries[target_key]))
                unit = target.dtype.itemsize
                limit = min(
                    compression_workers,
                    compression_memory_bytes
                    // (min(len(target), chunk_elems) * unit * 12 + 65536),
                )
                if len(target) and not limit:
                    raise MemoryError(
                        "One whole codec chunk exceeds compression memory budget"
                    )
                pending = []
                for first in range(0, len(target), chunk_elems):
                    selected = target[first : first + chunk_elems]
                    ref = None if base is None else base[first : first + chunk_elems]
                    future = (
                        None
                        if raw_latest or frozen
                        else pool.submit(
                            _encode_one, selected, ref, path, False, not coding
                        )
                    )
                    pending.append((first, selected, future))
                    if len(pending) < limit and first + chunk_elems < len(target):
                        continue
                    for start, values, future in pending:
                        source = values.tobytes()
                        frame = source if raw_latest or frozen else future.result()
                        row = {
                            "part": start // chunk_elems,
                            "start": start,
                            "n": len(values),
                            "sha": sha(source),
                            "frame_sha": sha(frame),
                            "size": len(frame),
                            "fragments": [],
                        }
                        if not raw_latest and not frozen:
                            row.update(file=pack_path.name, offset=pack.tell())
                            pack.write(frame)
                        wire = [source]
                        for target_key, derivation, bf_entry in fragments:
                            begin = max(start, derivation["start"])
                            end = min(
                                start + len(values),
                                derivation["start"] + bf_entry["spec"]["numel"],
                            )
                            if begin < end:
                                offset = begin - derivation["start"]
                                original = bytes(
                                    _tree.raw_view(bf_entry["tensor"])[
                                        offset * 2 : (offset + end - begin) * 2
                                    ]
                                )
                                row["fragments"].append(
                                    {
                                        "start": begin - start,
                                        "n": end - begin,
                                        "rule": derivation["rule"],
                                        "sha": sha(original),
                                        "target_key": target_key,
                                        "target_start": offset,
                                    }
                                )
                                wire.append(original)
                        row.update(
                            wire_bytes=sum(map(len, wire)), wire_sha=sha(b"".join(wire))
                        )
                        spec["chunks"].append(row)
                    pending.clear()
        if not pack_path.stat().st_size:
            pack_path.unlink()
        manifest["versions"][version] = current
        if latest:
            root_specs = current["tensors"]
        reference_entries = entries
        del state, template
    manifest["original_tensor_bytes"] = sum(
        s["original_bytes"] for v in manifest["versions"] for s in v["tensors"].values()
    )
    temp = output / "manifest.json.partial"
    temp.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temp.replace(output / "manifest.json")
    return Archive(output)


class Archive:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest.get("format") != FORMAT:
            raise ValueError("Unsupported or incomplete ALC archive")

    def version(self, index):
        if type(index) is not int or not 0 <= index <= self.manifest["latest"]:
            raise ValueError("Version index out of range")
        return self.manifest["versions"][index]

    def template(self, version):
        current = self.version(version)
        if version == self.manifest["latest"] and self.manifest.get("latest_raw", True):
            template, _ = _tree.extract(
                load_checkpoint(self.path / "latest.pt", mmap=True), hash_tensors=False
            )
        else:
            template = (self.path / f"v{version:06d}.metadata.pkl").read_bytes()
        if sha(template) != current["template_sha256"]:
            raise ValueError("Checkpoint metadata checksum mismatch")
        return template

    def stats(self):
        payload = sum(p.stat().st_size for p in self.path.iterdir() if p.is_file())
        total = self.manifest["original_tensor_bytes"]
        counts = {}
        for version in self.manifest["versions"]:
            for spec in version["tensors"].values():
                counts[spec["mode"]] = counts.get(spec["mode"], 0) + 1
        return {
            "versions": len(self.manifest["versions"]),
            "archive_bytes": payload,
            "original_tensor_bytes": total,
            "ratio_percent": 100 * payload / total if total else None,
            "modes": counts,
            "ratio_scope": "All archive files / original tensor bytes, including BF16; metadata is charged to the numerator",
        }

    def jobs(self, version):
        """Metadata-only query plan; no compressed frame is decoded here."""
        result = []
        for key, spec in self.version(version)["tensors"].items():
            if spec["mode"] == "DerivedBF16":
                continue
            for chunk in spec["chunks"]:
                layers, current, visited = [], version, set()
                while True:
                    if current in visited:
                        raise ValueError("Archive dependency cycle")
                    visited.add(current)
                    tensor = self.version(current)["tensors"][key]
                    mode = tensor["mode"]
                    if mode == "FrozenRef":
                        current = tensor["reference_version"]
                        continue
                    layer = dict(tensor["chunks"][chunk["part"]])
                    layer.update(
                        mode=mode, version=current, key=key, dtype=spec["dtype"]
                    )
                    layers.append(layer)
                    if mode not in ("FM-Delta*", "RotDelta"):
                        break
                    current = tensor["reference_version"]
                result.append(
                    {
                        "key": key,
                        "dtype": spec["dtype"],
                        "chunk": chunk,
                        "layers": layers,
                        "fragments": chunk["fragments"],
                    }
                )
        return result

    def preload(self, jobs):
        """Force actual reads/copies into anonymous RAM before resident timing."""
        from contextlib import ExitStack

        result, streams, head = [], {}, None
        with ExitStack() as stack:
            for job in jobs:
                layers = []
                for original in job["layers"]:
                    layer = dict(original)
                    if layer["mode"] == "Raw":
                        if head is None:
                            _, head = _tree.extract(
                                load_checkpoint(self.path / "latest.pt", mmap=True),
                                hash_tensors=False,
                            )
                        values = _tree.array(head[layer["key"]]["tensor"])
                        data = values[
                            layer["start"] : layer["start"] + layer["n"]
                        ].tobytes()
                    else:
                        name = layer["file"]
                        if name not in streams:
                            streams[name] = stack.enter_context(
                                (self.path / name).open("rb")
                            )
                        stream = streams[name]
                        stream.seek(layer["offset"])
                        data = stream.read(layer["size"])
                    if len(data) != layer["size"] or sha(data) != layer["frame_sha"]:
                        raise ValueError(
                            "Compressed frame checksum mismatch during preload"
                        )
                    layer["data"] = data
                    layers.append(layer)
                result.append({**job, "layers": layers})
        return result

    def restore(self, version):
        """Recover a complete loadable state, including metadata and BF16 fallback."""
        self.version(version)
        if version == self.manifest["latest"] and self.manifest.get("latest_raw", True):
            if file_sha(self.path / "latest.pt") != self.manifest["latest_file_sha256"]:
                raise ValueError("Latest Raw checkpoint checksum mismatch")
            return load_checkpoint(self.path / "latest.pt")
        from serving import decode_layer

        jobs = {(j["key"], j["chunk"]["part"]): j for j in self.jobs(version)}
        cache = {}

        def tensor(key):
            if key in cache:
                return cache[key]
            spec = self.version(version)["tensors"][key]
            if spec["mode"] == "DerivedBF16":
                d = spec["derivation"]
                master = _tree.array(tensor(d["source_key"]))[
                    d["start"] : d["start"] + spec["numel"]
                ]
                data = bf16.convert(master, d["rule"])
            else:
                values = []
                for chunk in spec["chunks"]:
                    job = self.preload([jobs[key, chunk["part"]]])[0]
                    base = None
                    for layer in reversed(job["layers"]):
                        base = decode_layer(layer, base)
                    data = base.tobytes()
                    if sha(data) != chunk["sha"]:
                        raise ValueError("Restored chunk checksum mismatch")
                    values.append(data)
                data = b"".join(values)
            if sha(data) != spec["sha256"]:
                raise ValueError("Restored tensor checksum mismatch")
            cache[key] = _tree.from_bytes(data, spec)
            return cache[key]

        return _tree.rebuild(self.template(version), tensor)
