"""Bounded, ordered chunk groups with a barrier at each reverse-delta stage."""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

from serving import decode_layer, finish
from tensor_compress import require_native


def grouped_decode(
    jobs, workers, events=None, executor=None, with_ready_time=False, load_group=None
):
    """Version barriers within a group, two-group decode/send pipeline.

    Adaptive Direct paths can terminate different chunks at different versions.
    A stage operates on the active corresponding chunks only; other chunks join
    at their own Direct/Raw anchor. Only target-version payloads are yielded.
    """
    if not jobs or workers <= 0:
        raise ValueError("positive workers and a nonempty resident segment required")
    events = events if events is not None else []
    ready, slots, stopped = queue.Queue(), threading.Semaphore(2), threading.Event()

    def produce():
        try:
            with (
                nullcontext(executor)
                if executor
                else ThreadPoolExecutor(max_workers=workers)
            ) as pool:
                for first in range(0, len(jobs), workers):
                    while not slots.acquire(timeout=0.1):
                        if stopped.is_set():
                            return
                    if stopped.is_set():
                        return
                    batch, group = jobs[first : first + workers], first // workers
                    if load_group is not None:
                        before = require_native().clock()
                        batch = load_group(batch)
                        events.append(
                            {
                                "kind": "read_group",
                                "group": group,
                                "start": before,
                                "end": require_native().clock(),
                            }
                        )
                    start = require_native().clock()
                    bases = [None] * len(batch)
                    for depth in reversed(range(max(len(j["layers"]) for j in batch))):
                        before = require_native().clock()
                        active = [
                            i
                            for i, job in enumerate(batch)
                            if depth < len(job["layers"])
                        ]
                        pending = [
                            (
                                i,
                                pool.submit(
                                    decode_layer, batch[i]["layers"][depth], bases[i]
                                ),
                            )
                            for i in active
                        ]
                        for i, future in pending:
                            bases[i] = future.result()
                        events.append(
                            {
                                "kind": "decode_version",
                                "group": group,
                                "version_offset": depth,
                                "active_chunks": len(active),
                                "start": before,
                                "end": require_native().clock(),
                            }
                        )
                    pending = [
                        pool.submit(finish, job, base)
                        for job, base in zip(batch, bases)
                    ]
                    payloads = [f.result() for f in pending]
                    events.append(
                        {
                            "kind": "decode_group",
                            "group": group,
                            "start_chunk": first,
                            "end_chunk": first + len(batch),
                            "start": start,
                            "end": require_native().clock(),
                        }
                    )
                    ready.put((payloads, require_native().clock()))
                    # Drop worker references before waiting for the next slot.
                    del pending, bases, payloads
                ready.put(None)
        except BaseException as exc:
            ready.put(exc)

    thread = threading.Thread(target=produce, name="alc-group-decoder", daemon=True)
    thread.start()
    try:
        while True:
            payloads = ready.get(timeout=3600)
            if payloads is None:
                break
            if isinstance(payloads, BaseException):
                raise payloads
            payloads, ready_at = payloads
            if with_ready_time:
                for data, depth in payloads:
                    yield data, depth, ready_at
            else:
                yield from payloads
            # Consumer resumes after the group's final chunk has been sent.
            slots.release()
            del payloads
    finally:
        stopped.set()
        slots.release()
        thread.join(timeout=3600)
        if thread.is_alive():
            raise RuntimeError("group decoder did not stop")
