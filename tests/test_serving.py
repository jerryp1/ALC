import hashlib
import json
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch
from conftest import equal, states

from checkpoint_archive import Archive, compress
from serving import (
    estimate_memory,
    query_local,
    receive_query,
    resident_segments,
    send_query,
)


@pytest.mark.parametrize("resident", [False, True])
def test_ordered_complete_state_stream_and_bf16(tmp_path, resident):
    original = states()
    archive = compress(
        original, tmp_path / "archive", chunk_elems=128, compression_workers=2
    )
    result = query_local(
        archive, 1, workers=2, resident=resident, memory_budget_bytes=2**20
    )
    assert result["received_bytes"] == result["expected_bytes"]
    assert result["received_bytes"] == sum(
        s["original_bytes"] for s in archive.version(1)["tensors"].values()
    )
    assert result["disk_io_in_timing"] is not resident
    assert (
        0
        <= result["first_payload_byte_seconds"]
        <= result["first_complete_chunk_seconds"]
        <= result["completion_seconds"]
    )
    assert result["buffer_groups"] == 2
    expected = hashlib.sha256()
    for job in archive.jobs(1):
        expected.update(job["chunk"]["wire_bytes"].to_bytes(8, "big"))
        expected.update(bytes.fromhex(job["chunk"]["wire_sha"]))
    assert result["received_sha256"] == expected.hexdigest()
    assert result["verified_chunks"] == len(archive.jobs(1))
    for row in result["segments"]:
        groups = [e for e in row["events"] if e["kind"] == "decode_group"]
        for send in row["send_intervals"]:
            group = next(
                g for g in groups if g["start_chunk"] <= send["chunk"] < g["end_chunk"]
            )
            assert send["start"] >= group["end"]
    equal(original[1], archive.restore(1))


def test_resident_complete_chunks_under_fixed_memory_budget(tmp_path):
    archive = compress(states(elements=1025), tmp_path / "archive", chunk_elems=128)
    jobs = archive.jobs(0)
    budget = (
        max(
            estimate_memory(jobs[first : first + 2], 2)
            for first in range(0, len(jobs), 2)
        )
        + 1024
    )
    windows = resident_segments(jobs, 2, budget)
    assert len(windows) > 1 and windows[0][0] == 0 and windows[-1][1] == len(jobs)
    result = query_local(
        archive, 0, workers=2, resident=True, memory_budget_bytes=budget
    )
    assert result["segment_count"] > 1
    assert result["measurement_kind"] == "segmented_active_time_sum"
    assert result["completion_seconds"] == pytest.approx(
        sum(s["active_seconds"] for s in result["segments"])
    )
    assert result["query_wall_seconds"] >= result["completion_seconds"]
    with pytest.raises(MemoryError):
        resident_segments(jobs, 2, 1)


@pytest.mark.parametrize("version", [0, 1, 4])
def test_real_socket_receiver_returns_a_loadable_complete_state(tmp_path, version):
    original = states()
    archive = compress(original, tmp_path / "archive", chunk_elems=128)
    server, client = socket.socketpair()
    server.settimeout(10)
    client.settimeout(10)
    errors = []

    def serve():
        try:
            with server:
                send_query(archive, version, server, workers=2, bandwidth_gbps=8)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=serve)
    thread.start()
    with client:
        state, result = receive_query(client)
    thread.join(10)
    assert not thread.is_alive() and not errors
    equal(original[version], state)
    assert state["tied"] is state["model"]["weight"]
    assert result["received_bytes"] == sum(
        s["original_bytes"] for s in archive.version(version)["tensors"].values()
    )
    assert (
        0
        <= result["first_payload_byte_seconds"]
        <= result["first_complete_chunk_seconds"]
        <= result["completion_seconds"]
    )


def test_cli_example_archive_restore_and_tcp_fetch(tmp_path):
    root = Path(__file__).resolve().parents[1]
    inputs = tmp_path / "inputs"
    subprocess.run(
        [
            sys.executable,
            str(root / "src/synthetic_checkpoints.py"),
            "--output",
            str(inputs),
        ],
        check=True,
        capture_output=True,
    )
    archive = tmp_path / "archive"

    def command(script, *args):
        return subprocess.run(
            [sys.executable, str(root / script), *map(str, args)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

    command(
        "compress_checkpoint.py",
        "--checkpoints",
        *sorted(inputs.glob("*.pt")),
        "--output",
        archive,
        "--chunk-elements",
        128,
        "--workers",
        2,
    )
    assert Archive(archive).manifest["group_size"] == 5
    output = tmp_path / "restored.pt"
    command(
        "decompress_checkpoint.py",
        "--input",
        archive,
        "--version",
        1,
        "--output",
        output,
    )
    expected = torch.load(inputs / "checkpoint-01.pt", weights_only=False)
    equal(expected, torch.load(output, weights_only=False))
    process = subprocess.Popen(
        [
            sys.executable,
            str(root / "serve_checkpoint.py"),
            "serve",
            str(archive),
            "--port",
            "0",
            "--workers",
            "2",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        first = process.stdout.readline()
        assert first, process.stderr.read()
        port = json.loads(first)["listening"][1]
        fetched = tmp_path / "received.pt"
        command(
            "serve_checkpoint.py",
            "fetch",
            "--port",
            port,
            "--version",
            1,
            "--output",
            fetched,
        )
        process.communicate(timeout=30)
        assert process.returncode == 0
        equal(expected, torch.load(fetched, weights_only=False))
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_metadata_only_checkpoint_socket_round_trip(tmp_path):
    original = {"step": 10, "empty": torch.empty(0), "config": [None, 0.1]}
    archive = compress([original, original], tmp_path / "archive")
    server, client = socket.socketpair()
    errors = []

    def serve():
        try:
            with server:
                send_query(archive, 0, server, workers=2)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=serve)
    thread.start()
    with client:
        state, result = receive_query(client)
    thread.join(10)
    assert not thread.is_alive() and not errors
    equal(original, state)
    assert result["received_bytes"] == 0
