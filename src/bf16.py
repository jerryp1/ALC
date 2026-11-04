"""Verified RNE/truncation derivation and server-side BF16 reconstruction."""

import hashlib

import numpy as np
import torch


def convert(values, rule):
    if rule == "rne":
        return (
            torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
            .to(torch.bfloat16)
            .view(torch.int16)
            .numpy()
            .view(np.uint16)
            .tobytes()
        )
    if rule == "truncate":
        return (values.view(np.uint32) >> np.uint32(16)).astype(np.uint16).tobytes()
    raise ValueError("Unknown BF16 derivation rule")


def verified_rule(master, original_bits):
    for rule in ("rne", "truncate"):
        if convert(master, rule) == original_bits:
            return rule
    return None


def assemble_wire(values, fragments, wire_bytes, wire_sha):
    """Same native finite fast path and verified NaN handling as the paper code."""
    from tensor_compress import require_native

    result = require_native().pack_wire(
        values,
        [(f["start"], f["n"], {"rne": 0, "truncate": 1}[f["rule"]]) for f in fragments],
    )
    view = memoryview(result).cast("B")
    offset = values.nbytes
    for fragment in fragments:
        size = fragment["n"] * 2
        if hashlib.sha256(view[offset : offset + size]).hexdigest() != fragment["sha"]:
            selected = values[fragment["start"] : fragment["start"] + fragment["n"]]
            if fragment["rule"] != "rne" or not np.isnan(selected).any():
                raise ValueError("Rebuilt BF16 differs from source bits")
            exact = convert(selected, fragment["rule"])
            if hashlib.sha256(exact).hexdigest() != fragment["sha"]:
                raise ValueError("Source NaN rule is not reproducible")
            result[offset : offset + size] = np.frombuffer(exact, np.uint8)
        offset += size
    if len(view) != wire_bytes or hashlib.sha256(view).hexdigest() != wire_sha:
        raise ValueError("Reconstructed wire chunk checksum mismatch")
    return view
