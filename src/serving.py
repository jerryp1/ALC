"""Ordered server-side recovery, bounded pipelining and verifiable state streams.

query_local uses the original native paced transport to validate the algorithm.
send_query/receive_query expose a real socket interface returning a loadable
PyTorch state. Streaming includes archive reads; resident mode excludes frame IO.
"""

import base64
import json
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import bf16
import checkpoint_io as _tree
import tensor_compress as codec
from checkpoint_archive import Archive, sha
from serving_transport import MemoryTCP, clock


def numpy_dtype(name):
    return {"float32": np.float32, "float64": np.float64, "bfloat16": np.uint16}.get(
        name, np.uint8
    )


def decode_layer(layer, base):
    if layer["mode"] == "Raw":
        return np.frombuffer(layer["data"], numpy_dtype(layer["dtype"])).copy()
    if layer["mode"] == "Bytes":
        return np.frombuffer(
            codec.require_native().tans_decode(layer["data"]), np.uint8
        ).copy()
    return codec.decode_frame(layer["data"], base)


def finish(job, values):
    chunk = job["chunk"]
    if job["dtype"] == "float32":
        data = bf16.assemble_wire(
            values, job["fragments"], chunk["wire_bytes"], chunk["wire_sha"]
        )
    else:
        data = values.tobytes()
        if (
            job["fragments"]
            or len(data) != chunk["wire_bytes"]
            or sha(data) != chunk["wire_sha"]
        ):
            raise ValueError("Reconstructed wire chunk checksum mismatch")
    return data, len(job["layers"]) - 1


def estimate_memory(jobs, workers):
    if not jobs:
        return 0
    count = min(len(jobs), workers)
    sources = sum(layer["size"] for job in jobs for layer in job["layers"])
    work = max(
        job["chunk"]["n"] * np.dtype(numpy_dtype(job["dtype"])).itemsize for job in jobs
    )
    wire = max(job["chunk"]["wire_bytes"] for job in jobs)
    metadata = sum(
        2048 * (1 + len(job["layers"]) + len(job["fragments"])) for job in jobs
    )
    return (
        sources + metadata + 6 * count * work + (min(len(jobs), 2 * workers) + 2) * wire
    )


def resident_segments(jobs, workers, budget):
    """Keep complete codec chunks and align boundaries with worker groups."""
    if budget <= 0 or workers < 1:
        raise ValueError("Positive memory budget and worker count required")
    windows, first = [], 0
    while first < len(jobs):
        end = first
        # A binary search avoids quadratic planning on large state streams.
        lo, hi = first + 1, len(jobs)
        while lo <= hi:
            middle = (lo + hi) // 2
            if estimate_memory(jobs[first:middle], workers) <= budget:
                end, lo = middle, middle + 1
            else:
                hi = middle - 1
        if end == first:
            raise MemoryError(
                "One whole codec chunk and workspace exceed query memory budget"
            )
        aligned = first + ((end - first) // workers) * workers
        if end < len(jobs) and aligned > first:
            end = aligned
        windows.append((first, end))
        first = end
    return windows


def query_local(
    archive,
    version,
    workers=8,
    bandwidth_gbps=8,
    resident=False,
    memory_budget_bytes=2**30,
    backend="unix_paced",
):
    """Validate actual bytes over a native paced stream with bounded verification.

    The local receiver verifies and discards payloads. Use receive_query to
    retain a complete checkpoint. Timed major faults are reported, never hidden.
    No model-specific warmup/repetition or experimental results are bundled.
    """
    from serving_scheduler import grouped_decode

    archive = archive if isinstance(archive, Archive) else Archive(archive)
    if type(workers) is not int or workers < 1 or bandwidth_gbps <= 0:
        raise ValueError("Positive workers and bandwidth required")
    wall = clock()
    jobs = archive.jobs(version)
    if not jobs:
        raise ValueError("No tensor payloads to serve")
    if resident:
        windows = resident_segments(jobs, workers, memory_budget_bytes)
    else:
        peak = max(
            2 * estimate_memory(jobs[first : first + workers], workers)
            for first in range(0, len(jobs), workers)
        )
        if peak > memory_budget_bytes:
            raise MemoryError("Streaming chunk group exceeds query memory budget")
        windows = [(0, len(jobs))]
    rows = []
    transport = MemoryTCP(bandwidth_gbps, backend)
    try:
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="alc-decode"
        ) as pool:
            for first, end in windows:
                preload_start = clock()
                selected = (
                    archive.preload(jobs[first:end]) if resident else jobs[first:end]
                )
                preload_seconds = clock() - preload_start if resident else 0
                events = []
                producer = grouped_decode(
                    selected,
                    workers,
                    events,
                    pool,
                    True,
                    load_group=None if resident else archive.preload,
                )
                transfer = transport.transfer(
                    [job["chunk"] for job in selected], producer
                )
                rows.append(
                    {
                        "start_chunk": first,
                        "end_chunk": end,
                        "active_seconds": transfer["completion_seconds"],
                        "preload_seconds_excluded": preload_seconds,
                        "events": [
                            {
                                **event,
                                "start": event["start"] - transport.last_start,
                                "end": event["end"] - transport.last_start,
                            }
                            for event in events
                        ],
                        **transfer,
                    }
                )
                del selected, producer
    finally:
        transport.close()
    return {
        "version": version,
        "workers": workers,
        "chunk_elems": archive.manifest["chunk_elems"],
        "checkpoint_group_size": archive.manifest["group_size"],
        "buffer_groups": 2,
        "scheduler": "version_barriers_then_target_group_send_double_buffer",
        "first_payload_byte_seconds": rows[0]["first_payload_byte_seconds"],
        "first_complete_chunk_seconds": rows[0]["first_complete_chunk_seconds"],
        "completion_seconds": sum(row["completion_seconds"] for row in rows),
        "query_wall_seconds": clock() - wall,
        "timed_major_page_faults": sum(row["timed_major_page_faults"] for row in rows),
        "received_bytes": sum(row["received_bytes"] for row in rows),
        "expected_bytes": sum(job["chunk"]["wire_bytes"] for job in jobs),
        "verified_chunks": sum(row["verified_chunks"] for row in rows),
        "received_sha256": transport.received_digest,
        "received_digest_kind": "ordered_length_and_payload_sha256_manifest",
        "disk_io_in_timing": not resident,
        "segment_count": len(rows),
        "segments": rows,
        "measurement_kind": "segmented_active_time_sum"
        if resident and len(rows) > 1
        else "continuous_active_query",
        "network_backend": backend,
        "bandwidth_gbps": bandwidth_gbps,
        "network_scope": "Actual local socket bytes with application pacing; not a physical NIC benchmark",
    }


def _read_exact(sock, size, on_first=None):
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        received = sock.recv_into(view[offset:])
        if not received:
            raise EOFError("Checkpoint stream ended early")
        if offset == 0 and on_first is not None:
            on_first()
        offset += received
    return data


def send_query(archive, version, sock, workers=8, bandwidth_gbps=None):
    """Serve one checkpoint to a trusted client over a connected TCP/Unix socket.

    The header carries only types, checksums, fragment placement and checkpoint
    metadata. BF16 values are regenerated on the server while processing the
    same master chunk, and transmitted as ordinary output bytes.
    """
    from serving_scheduler import grouped_decode

    if type(workers) is not int or workers < 1:
        raise ValueError("Positive workers required")
    if bandwidth_gbps is not None and bandwidth_gbps <= 0:
        raise ValueError("Positive bandwidth required")
    archive = archive if isinstance(archive, Archive) else Archive(archive)
    jobs = archive.jobs(version)
    template = archive.template(version)
    chunks = []
    for job in jobs:
        chunk = job["chunk"]
        width = np.dtype(numpy_dtype(job["dtype"])).itemsize
        chunks.append(
            {
                "bytes": chunk["wire_bytes"],
                "sha256": chunk["wire_sha"],
                "pieces": [
                    {
                        "key": job["key"],
                        "offset": chunk["start"] * width,
                        "bytes": chunk["n"] * width,
                    }
                ]
                + [
                    {
                        "key": f["target_key"],
                        "offset": f["target_start"] * 2,
                        "bytes": f["n"] * 2,
                    }
                    for f in job["fragments"]
                ],
            }
        )
    header = {
        "format": "ALC_STATE_STREAM_1",
        "version": version,
        "template": base64.b64encode(template).decode("ascii"),
        "template_sha256": sha(template),
        "tensors": archive.version(version)["tensors"],
        "chunks": chunks,
    }
    encoded = json.dumps(header, allow_nan=False).encode("utf-8")
    sock.sendall(struct.pack("!Q", len(encoded)) + encoded)
    if _read_exact(sock, 1) != b"R":
        raise ValueError("Client was not ready")
    start = clock()
    sender = codec.require_native().Sender(
        1e6 if bandwidth_gbps is None else bandwidth_gbps
    )
    events = []
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="alc-decode"
    ) as pool:
        producer = (
            grouped_decode(jobs, workers, events, pool, True, archive.preload)
            if jobs
            else ()
        )
        for data, _, ready_at in producer:
            sender.send(sock.fileno(), memoryview(data).cast("B"), ready_at)
    if _read_exact(sock, 1) != b"V":
        raise ValueError("Client did not verify the complete checkpoint")
    return {
        "version": version,
        "server_seconds_through_ack": clock() - start,
        "sent_bytes": sum(c["bytes"] for c in chunks),
        "events": events,
        "disk_io_in_timing": True,
    }


def receive_query(sock, maximum_header_bytes=512 * 2**20):
    """Receive, verify, and reconstruct a full checkpoint from a trusted server.

    Remote latency is measured on this client's clock from its ready signal;
    it is not presented as a cross-host synchronized server-start timestamp.
    """
    (length,) = struct.unpack("!Q", _read_exact(sock, 8))
    if length > maximum_header_bytes:
        raise ValueError("Checkpoint stream header exceeds limit")
    header = json.loads(_read_exact(sock, length))
    if header.get("format") != "ALC_STATE_STREAM_1":
        raise ValueError("Unknown checkpoint stream")
    template = base64.b64decode(header["template"], validate=True)
    if sha(template) != header["template_sha256"]:
        raise ValueError("Stream metadata checksum mismatch")
    specs = header["tensors"]
    buffers = {key: bytearray(spec["original_bytes"]) for key, spec in specs.items()}
    coverage = {key: [] for key in specs}
    first = [None]
    first_checked = None

    def arrived():
        if first[0] is None:
            first[0] = clock()

    start = clock()
    sock.sendall(b"R")
    total = 0
    for chunk in header["chunks"]:
        (received,) = struct.unpack("!Q", _read_exact(sock, 8))
        if received != chunk["bytes"]:
            raise ValueError("Stream chunk size mismatch")
        payload = _read_exact(sock, received, arrived)
        if sha(payload) != chunk["sha256"]:
            raise ValueError("Stream payload checksum mismatch")
        if first_checked is None:
            first_checked = clock()
        cursor = 0
        for piece in chunk["pieces"]:
            key, offset, size = piece["key"], piece["offset"], piece["bytes"]
            if (
                key not in buffers
                or offset < 0
                or size < 0
                or offset + size > len(buffers[key])
            ):
                raise ValueError("Invalid tensor placement in stream")
            if cursor + size > len(payload):
                raise ValueError("Truncated tensor fragment")
            buffers[key][offset : offset + size] = payload[cursor : cursor + size]
            coverage[key].append((offset, offset + size))
            cursor += size
        if cursor != received:
            raise ValueError("Unassigned stream payload bytes")
        total += received
    for key, intervals in coverage.items():
        cursor = 0
        for begin, end in sorted(intervals):
            if begin != cursor:
                raise ValueError("Missing or overlapping tensor fragments")
            cursor = end
        if cursor != len(buffers[key]) or sha(buffers[key]) != specs[key]["sha256"]:
            raise ValueError("Incomplete or corrupted tensor")
    tensors = {key: _tree.from_bytes(data, specs[key]) for key, data in buffers.items()}
    state = _tree.rebuild(template, tensors.__getitem__)
    finished = clock()
    sock.sendall(b"V")
    return state, {
        "first_payload_byte_seconds": 0 if first[0] is None else first[0] - start,
        "first_complete_chunk_seconds": 0
        if first_checked is None
        else first_checked - start,
        "completion_seconds": finished - start,
        "received_bytes": total,
        "verified_chunks": len(header["chunks"]),
        "timing_origin": "client_ready; includes network and reconstruction, excludes output-file writing",
    }
