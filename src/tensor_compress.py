"""Paper cost model and tANS frames, without trial compression or raw fallback.

FP32 uses the original ALA1 native encoding. BF16 words and FP64 use a compact
generic frame with the same I-Align/PCMap/RotMap and joint-(S,K) construction.
Their Python bit packing is a correctness-oriented extension, not the native
FP32 throughput implementation measured in the paper.
"""

import hashlib
import importlib
import math
import struct
import time
from dataclasses import dataclass

import numpy as np

PATHS = ("Direct", "FM-Delta*", "RotDelta")
HEADER = struct.Struct("<4sBIQ")
GENERIC = struct.Struct("<4sBBIQ")
LENGTH = struct.Struct("<Q")
_NATIVE = None


def require_native():
    global _NATIVE
    if _NATIVE is None:
        try:
            _NATIVE = importlib.import_module("cpp.alc_tans_cpp_codec")
        except ImportError as exc:
            raise RuntimeError(
                "Build ALC first: python -m pip install -e '.[test]'"
            ) from exc
    return _NATIVE


def word_dtype(array):
    dtype = np.asarray(array).dtype
    if dtype not in (np.dtype("uint16"), np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Expected BF16 uint16 words, FP32 or FP64")
    return np.dtype("u" + str(dtype.itemsize))


def words(array):
    array = np.asarray(array)
    if array.ndim != 1 or not array.flags.c_contiguous:
        raise ValueError("Expected a contiguous, flat array")
    return array.view(word_dtype(array))


def rotmap(array):
    u = words(array)
    bits = 8 * u.dtype.itemsize
    return (u << 1) | (u >> (bits - 1))


def inverse_rotmap(u):
    bits = 8 * u.dtype.itemsize
    return (u >> 1) | (u << (bits - 1))


def pcmap(array):
    u = words(array)
    mask = np.array(1 << (8 * u.dtype.itemsize - 1), dtype=u.dtype)
    return np.where((u & mask) == 0, u ^ mask, ~u).astype(u.dtype)


def inverse_pcmap(u):
    mask = np.array(1 << (8 * u.dtype.itemsize - 1), dtype=u.dtype)
    return np.where((u & mask) != 0, u ^ mask, ~u).astype(u.dtype)


def joint_symbols(reference, target):
    """Exact integer bit lengths, including uint64 extrema and zero residuals."""
    if reference.dtype != target.dtype or reference.shape != target.shape:
        raise ValueError("Matching mapped words required")
    bits = 8 * target.dtype.itemsize
    positive, negative = reference < target, reference > target
    magnitude = np.zeros_like(target)
    magnitude[positive] = target[positive] - reference[positive]
    magnitude[negative] = reference[negative] - target[negative]
    widths = np.fromiter(
        (max(0, int(value).bit_length() - 1) for value in magnitude),
        dtype=np.uint8,
        count=len(target),
    )
    symbols = np.full(len(target), bits, dtype=np.uint8)
    symbols[positive] = bits + 1 + widths[positive]
    symbols[negative] = bits - 1 - widths[negative]
    return symbols, magnitude, widths


def entropy(values):
    if not len(values):
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def choose_path(target, reference=None, sample_size=65536):
    words(target)
    if type(sample_size) is not int or not 1 <= sample_size <= 65536:
        raise ValueError("sample_size must be between 1 and 65536")
    if reference is not None and (
        reference.dtype != target.dtype or reference.shape != target.shape
    ):
        reference = None
    n = len(target)
    indices = np.random.default_rng(0).choice(n, min(n, sample_size), replace=False)
    sample = target[indices]
    matrix = rotmap(sample).view(np.uint8).reshape(-1, target.dtype.itemsize)
    costs = {
        "Direct": sum(entropy(matrix[:, i]) for i in range(target.dtype.itemsize)),
        "FM-Delta*": math.inf,
        "RotDelta": math.inf,
    }
    if reference is not None:
        for name, mapping in [("FM-Delta*", pcmap), ("RotDelta", rotmap)]:
            symbols, _, widths = joint_symbols(
                mapping(reference[indices]), mapping(sample)
            )
            costs[name] = entropy(symbols) + (float(widths.mean()) if n else 0.0)
    return {
        "path": min(PATHS, key=costs.__getitem__),
        "estimated_bits": costs,
        "sample_elements": len(indices),
        "sample_indices_sha256": hashlib.sha256(
            indices.astype("<u8").tobytes()
        ).hexdigest(),
    }


def encode_frame(target, reference, path):
    u = words(target)
    if not len(target) or len(target) >= 2**32 or path not in PATHS:
        raise ValueError("Invalid nonempty codec chunk/path")
    if reference is None:
        if path != "Direct":
            raise ValueError("Delta frame requires a reference")
    elif reference.dtype != target.dtype or reference.shape != target.shape:
        raise ValueError("Mismatching reference")
    native = require_native()
    if path == "Direct":
        matrix = rotmap(target).view(np.uint8).reshape(-1, u.dtype.itemsize)
        parts = []
        for column in range(u.dtype.itemsize):
            encoded = native.tans_encode(
                np.ascontiguousarray(matrix[:, column]).tobytes(), 8
            )
            parts.extend([LENGTH.pack(len(encoded)), encoded])
        body = b"".join(parts)
    elif target.dtype == np.float32:
        function = (
            native.encode_pcdelta_tans_f32_buffer
            if path == "FM-Delta*"
            else native.encode_rotdelta_tans_f32_buffer
        )
        body = function(reference, target, 8)
    else:
        mapping = pcmap if path == "FM-Delta*" else rotmap
        symbols, magnitude, widths = joint_symbols(mapping(reference), mapping(target))
        encoded = native.tans_encode(symbols.tobytes(), 8)
        bit_count = 0
        accumulator = 0
        payload = bytearray()
        for value, width in zip(magnitude, widths):
            width = int(width)
            accumulator = (accumulator << width) | (int(value) & ((1 << width) - 1))
            bit_count += width
            while bit_count >= 8:
                bit_count -= 8
                payload.append((accumulator >> bit_count) & 255)
                accumulator &= (1 << bit_count) - 1
        tail = bit_count
        if tail:
            payload.append(accumulator << (8 - tail))
        body = (
            LENGTH.pack(len(encoded))
            + encoded
            + LENGTH.pack(sum(int(x) for x in widths))
            + payload
        )
    if target.dtype == np.float32:
        return HEADER.pack(b"ALA1", PATHS.index(path), len(target), len(body)) + body
    return (
        GENERIC.pack(
            b"ALG1", PATHS.index(path), target.dtype.itemsize, len(target), len(body)
        )
        + body
    )


def decode_frame(frame, reference=None):
    if len(frame) < HEADER.size:
        raise ValueError("Truncated frame")
    native = require_native()
    if frame[:4] == b"ALA1":
        magic, path, n, size = HEADER.unpack_from(frame)
        if path >= len(PATHS) or size != len(frame) - HEADER.size:
            raise ValueError("Invalid FP32 frame")
        if path and (
            reference is None or reference.dtype != np.float32 or len(reference) != n
        ):
            raise ValueError("Delta reference mismatch")
        return native.decode_frame(
            memoryview(frame), np.empty(n, np.float32) if path == 0 else reference
        )
    if len(frame) < GENERIC.size:
        raise ValueError("Truncated generic frame")
    magic, path, width, n, size = GENERIC.unpack_from(frame)
    if (
        magic != b"ALG1"
        or path >= len(PATHS)
        or width not in (2, 8)
        or size != len(frame) - GENERIC.size
    ):
        raise ValueError("Invalid generic frame")
    dtype = np.dtype("uint16" if width == 2 else "float64")
    body = memoryview(frame)[GENERIC.size :]
    if path == 0:
        matrix = np.empty((n, width), np.uint8)
        offset = 0
        for column in range(width):
            if offset + LENGTH.size > len(body):
                raise ValueError("Truncated direct column")
            (length,) = LENGTH.unpack_from(body, offset)
            offset += LENGTH.size
            if offset + length > len(body):
                raise ValueError("Truncated direct payload")
            decoded = native.tans_decode(bytes(body[offset : offset + length]))
            if len(decoded) != n:
                raise ValueError("Invalid decoded column length")
            matrix[:, column] = np.frombuffer(decoded, np.uint8)
            offset += length
        if offset != len(body):
            raise ValueError("Trailing direct bytes")
        return inverse_rotmap(matrix.view("u" + str(width)).reshape(-1)).view(dtype)
    if reference is None or reference.dtype != dtype or len(reference) != n:
        raise ValueError("Generic delta reference mismatch")
    if len(body) < 16:
        raise ValueError("Truncated delta frame")
    (length,) = LENGTH.unpack_from(body)
    if length + 16 > len(body):
        raise ValueError("Truncated symbol stream")
    symbols = native.tans_decode(bytes(body[8 : 8 + length]))
    (bits,) = LENGTH.unpack_from(body, 8 + length)
    payload = body[16 + length :]
    if len(symbols) != n or len(payload) != (bits + 7) // 8:
        raise ValueError("Invalid generic delta lengths")
    mapping = pcmap if path == 1 else rotmap
    base = mapping(reference)
    output = np.empty(n, dtype=base.dtype)
    cursor = 0
    word_bits = width * 8
    for index, symbol in enumerate(symbols):
        delta = int(symbol) - word_bits
        if not -word_bits <= delta <= word_bits:
            raise ValueError("Invalid joint symbol")
        if delta:
            k = abs(delta) - 1
            if cursor + k > bits:
                raise ValueError("Truncated remainder bits")
            remainder = 0
            for position in range(cursor, cursor + k):
                remainder = (remainder << 1) | (
                    (payload[position // 8] >> (7 - position % 8)) & 1
                )
            cursor += k
            value = int(base[index]) + (1 if delta > 0 else -1) * ((1 << k) | remainder)
        else:
            value = int(base[index])
        if not 0 <= value < (1 << word_bits):
            raise ValueError("Decoded word overflow")
        output[index] = value
    if cursor != bits:
        raise ValueError("Unconsumed remainder bits")
    return (inverse_pcmap(output) if path == 1 else inverse_rotmap(output)).view(dtype)


# Original tensor-level API and path spelling, backed by the paper codec above.
CHUNK_ELEMS = 2**18
DEFAULT_SAMPLE = 65536
DEFAULT_ENCODING = "TANS"
PATH_IALIGN = "I-Align"
PATH_PCDELTA = "pcdelta"
PATH_ROTDELTA = "rotdelta"
_PAPER_PATH = {
    PATH_IALIGN: "Direct",
    PATH_PCDELTA: "FM-Delta*",
    PATH_ROTDELTA: "RotDelta",
}


@dataclass
class CompressStrategy:
    path: str
    encoding: str = DEFAULT_ENCODING
    ent_ialign: float = math.inf
    ent_pcdelta: float = math.inf
    ent_rotdelta: float = math.inf

    @property
    def chosen_entropy(self):
        return {
            PATH_IALIGN: self.ent_ialign,
            PATH_PCDELTA: self.ent_pcdelta,
            PATH_ROTDELTA: self.ent_rotdelta,
        }[self.path]


def analyze_strategy(
    ft_f32, base_f32=None, sample=DEFAULT_SAMPLE, encoding=DEFAULT_ENCODING
):
    """Keep the original API, using bounded sampling and whole-sample Shannon costs."""
    if encoding != DEFAULT_ENCODING:
        raise ValueError("The ALC paper paths use the shared TANS backend")
    result = choose_path(ft_f32, base_f32, sample)
    costs = result["estimated_bits"]
    path = next(name for name, paper in _PAPER_PATH.items() if paper == result["path"])
    return CompressStrategy(
        path, encoding, costs["Direct"], costs["FM-Delta*"], costs["RotDelta"]
    )


def compress_tensor(ft_f32, strategy, base_f32=None, chunk_elems=None):
    """Return (encoded bytes, independent chunk frames, encode seconds)."""
    if ft_f32.dtype != np.float32:
        raise ValueError(
            "The original tensor API expects FP32; use encode_frame for other word widths"
        )
    if strategy.encoding != DEFAULT_ENCODING or strategy.path not in _PAPER_PATH:
        raise ValueError("Unsupported ALC tensor strategy")
    chunk_elems = CHUNK_ELEMS if chunk_elems is None else chunk_elems
    if type(chunk_elems) is not int or not 1 <= chunk_elems < 2**32:
        raise ValueError("Invalid chunk size")
    path = _PAPER_PATH[strategy.path]
    if path != "Direct" and (
        base_f32 is None
        or base_f32.shape != ft_f32.shape
        or base_f32.dtype != ft_f32.dtype
    ):
        raise ValueError("Delta tensor requires a matching reference")
    start = time.perf_counter()
    chunks = []
    for first in range(0, len(ft_f32), chunk_elems):
        target = ft_f32[first : first + chunk_elems]
        reference = None if base_f32 is None else base_f32[first : first + len(target)]
        chunks.append((encode_frame(target, reference, path), {"n": len(target)}))
    return sum(len(frame) for frame, _ in chunks), chunks, time.perf_counter() - start


def decompress_tensor(all_chunks, strategy, n_elems, base_f32=None):
    """Reconstruct exact FP32 bits; each stored chunk carries its own size."""
    if strategy.encoding != DEFAULT_ENCODING or strategy.path not in _PAPER_PATH:
        raise ValueError("Unsupported ALC tensor strategy")
    start = time.perf_counter()
    values, cursor = [], 0
    for frame, metadata in all_chunks:
        count = metadata["n"]
        if count <= 0 or cursor + count > n_elems:
            raise ValueError("Invalid tensor chunk coverage")
        if HEADER.unpack_from(frame)[1] != PATHS.index(_PAPER_PATH[strategy.path]):
            raise ValueError("Tensor strategy/frame mismatch")
        reference = None if base_f32 is None else base_f32[cursor : cursor + count]
        value = decode_frame(frame, reference)
        if len(value) != count:
            raise ValueError("Invalid decoded chunk length")
        values.append(value)
        cursor += count
    if cursor != n_elems:
        raise ValueError("Missing tensor chunks")
    result = np.concatenate(values) if values else np.empty(0, np.float32)
    return result, time.perf_counter() - start


def compress_and_verify(
    ft_f32, base_f32=None, sample=DEFAULT_SAMPLE, encoding=DEFAULT_ENCODING
):
    strategy = analyze_strategy(ft_f32, base_f32, sample, encoding)
    size, chunks, enc_sec = compress_tensor(ft_f32, strategy, base_f32)
    recovered, dec_sec = decompress_tensor(chunks, strategy, len(ft_f32), base_f32)
    return dict(
        strategy=strategy,
        compressed_size=size,
        ratio=size / ft_f32.nbytes if ft_f32.nbytes else 0.0,
        bitwise_ok=np.array_equal(recovered.view(np.uint32), ft_f32.view(np.uint32)),
        enc_sec=enc_sec,
        dec_sec=dec_sec,
        ent_ialign=strategy.ent_ialign,
        ent_pcdelta=strategy.ent_pcdelta,
        ent_rotdelta=strategy.ent_rotdelta,
    )


def pcmap_f32(array):
    return pcmap(array)


def pcmap_inverse_f32(mapped):
    return inverse_pcmap(mapped).view(np.float32)


def rotl1(u32):
    return (u32 << np.uint32(1)) | (u32 >> np.uint32(31))


def rotr1(u32):
    return (u32 >> np.uint32(1)) | (u32 << np.uint32(31))
