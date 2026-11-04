"""Validate barriers and backpressure independently of codec timing."""

import threading

import pytest

import serving_scheduler as scheduler


def test_version_barriers_with_mixed_adaptive_depths_and_two_slots(monkeypatch):
    depths = [3, 1, 2, 0, 1, 3, 0, 2]
    jobs = [
        {"index": i, "layers": [{"index": i, "depth": d} for d in range(h + 1)]}
        for i, h in enumerate(depths)
    ]
    done, finished = set(), set()
    lock, second_ready = threading.Lock(), threading.Event()

    def layer_decode(layer, base):
        i, d = layer["index"], layer["depth"]
        start = (i // 2) * 2
        with lock:
            for j in range(start, start + 2):
                if depths[j] >= d + 1:
                    assert (j, d + 1) in done
            if d < depths[i]:
                assert base == (i, d + 1)
            else:
                assert base is None
            done.add((i, d))
        return i, d

    def finish(job, base):
        i = job["index"]
        assert base == (i, 0)
        with lock:
            finished.add(i)
            if {2, 3} <= finished:
                second_ready.set()
        return bytes([i]), depths[i]

    monkeypatch.setattr(scheduler, "decode_layer", layer_decode)
    monkeypatch.setattr(scheduler, "finish", finish)
    events = []
    stream = scheduler.grouped_decode(jobs, 2, events)
    first = next(stream)
    assert second_ready.wait(5)
    with lock:
        assert finished == {0, 1, 2, 3}, (
            "third group must wait for the sender to release group zero"
        )
    assert [first, *stream] == [(bytes([i]), d) for i, d in enumerate(depths)]
    assert [
        e["active_chunks"]
        for e in events
        if e["kind"] == "decode_version" and e["group"] == 0
    ] == [1, 1, 2, 2]
    assert not any(t.name == "alc-group-decoder" for t in threading.enumerate())


def test_decoder_failure_reaches_consumer_and_stops(monkeypatch):
    def fail(layer, base):
        raise ValueError("injected codec failure")

    monkeypatch.setattr(scheduler, "decode_layer", fail)
    with pytest.raises(ValueError, match="injected codec failure"):
        list(scheduler.grouped_decode([{"layers": [{}]}], 2))
    assert not any(t.name == "alc-group-decoder" for t in threading.enumerate())
