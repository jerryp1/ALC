"""GIL-free native socket IO and pacing, with bounded pipelined verification."""

import queue
import resource
import socket
import struct
import threading

from tensor_compress import require_native


def clock():
    return require_native().clock()


class MemoryTCP:
    """One connection/query; native send and recv do not execute Python loops.

    Two receive buffers permit SHA verification to overlap the next receive.
    Every payload is SHA256 verified. The whole-query digest hashes the ordered
    (length, payload-SHA256) manifest, avoiding a second full payload hash pass.
    Buffers are allocated and touched before the timed request starts.
    """

    def __init__(self, gbps, backend="tcp"):
        if backend not in ("tcp", "unix_paced"):
            raise ValueError("unknown fixed-bandwidth transport backend")
        self.backend = backend
        self.native = require_native()
        self.sender = self.native.Sender(gbps)
        self.requests, self.errors = queue.Queue(), []
        self.received_digest = None
        listener = None
        if backend == "unix_paced":
            server_socket, client_socket = socket.socketpair()
        else:
            listener = socket.socket()
            self._socket_options(listener)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            address = listener.getsockname()
            server_socket = client_socket = None

        def client():
            try:
                with (
                    client_socket if client_socket is not None else socket.socket()
                ) as sock:
                    self._socket_options(sock)
                    sock.settimeout(60)
                    if backend == "tcp":
                        sock.connect(address)
                    sock.settimeout(None)  # native blocking IO; kernel timeout below
                    self._socket_options(sock)
                    receiver = self.native.Receiver()
                    while True:
                        request = self.requests.get()
                        if request is None:
                            break
                        chunks = request["chunks"]
                        receiver.prepare(
                            [c.get("wire_bytes", c["n"] * 4) for c in chunks],
                            [c.get("wire_sha", c["sha"]) for c in chunks],
                        )
                        request["ready"].set()
                        request["begin"].wait()
                        result = receiver.receive(sock.fileno())
                        start = request["start"]
                        self.received_digest = result["digest"]
                        request["result"] = {
                            "first_payload_byte_seconds": result["first"] - start,
                            "first_complete_chunk_seconds": result["first_checked"]
                            - start,
                            "completion_seconds": result["finished"] - start,
                            "arrival_completion_seconds": result["last_arrival"]
                            - start,
                            "received_bytes": result["bytes"],
                            "verified_chunks": result["verified"],
                        }
                        receiver.clear_buffers()
                        sock.sendall(b"V")

            except BaseException as exc:
                self.errors.append(repr(exc))

        self.thread = threading.Thread(target=client, daemon=True, name="alc-client")
        self.thread.start()
        if listener is not None:
            listener.settimeout(60)
            self.sock, _ = listener.accept()
            listener.close()
        else:
            self.sock = server_socket
        self.sock.settimeout(None)
        self._socket_options(self.sock)

    @staticmethod
    def _socket_options(sock):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for option in (socket.SO_SNDBUF, socket.SO_RCVBUF):
            try:
                sock.setsockopt(socket.SOL_SOCKET, option, 4 * 1024 * 1024)
            except OSError:
                pass  # retain kernel limit; actual buffer sizes are recorded

        if hasattr(socket, "SO_NOSIGPIPE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_NOSIGPIPE, 1)
        # Native calls bypass Python's timeout polling. Set real kernel timeouts.
        timeout = struct.pack("ll", 120, 0)
        for opt in (socket.SO_RCVTIMEO, socket.SO_SNDTIMEO):
            sock.setsockopt(socket.SOL_SOCKET, opt, timeout)

    def transfer(self, chunks, producer, predecoded=False, before_timing=None):
        request = {
            "chunks": chunks,
            "ready": threading.Event(),
            "begin": threading.Event(),
        }
        self.requests.put(request)
        import time

        deadline = time.monotonic() + 120
        while not request["ready"].wait(0.1):
            if self.errors or time.monotonic() > deadline:
                raise RuntimeError("resident client not ready: " + repr(self.errors))
        if before_timing is not None:
            try:
                before_timing()
            except BaseException:
                request["start"] = clock()
                request["begin"].set()
                raise
        faults = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
        request["start"] = self.last_start = clock()
        request["begin"].set()
        depths, sends, waits = [], [], []
        iterator = iter(producer)
        try:
            while True:
                wait_start = clock()
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                ready = clock()
                data, depth = item[:2]
                group_ready = (
                    item[2]
                    if len(item) == 3
                    else request["start"]
                    if predecoded
                    else ready
                )
                waits.append(
                    {
                        "chunk": len(sends),
                        "start": wait_start - request["start"],
                        "end": ready - request["start"],
                    }
                )
                begin, end, sleep, syscall = self.sender.send(
                    self.sock.fileno(), memoryview(data).cast("B"), group_ready
                )
                sends.append(
                    {
                        "chunk": len(sends),
                        "start": begin - request["start"],
                        "end": end - request["start"],
                        "data_ready_at": group_ready - request["start"],
                        "wire_bytes": memoryview(data).nbytes,
                        "service_clock_deadline": self.sender.service_clock_deadline
                        - request["start"],
                        "pacing_sleep_seconds": sleep,
                        "socket_call_seconds": syscall,
                    }
                )
                depths.append(depth)
            if len(sends) != len(chunks) or self.sock.recv(1) != b"V":
                raise ValueError("missing verified client acknowledgement")
            result = request["result"]
        except BaseException as exc:
            self.sock.close()
            raise RuntimeError({"server": repr(exc), "client": self.errors}) from exc
        return {
            **result,
            "max_actual_delta_hops": max(depths, default=0),
            "received_digest_kind": "ordered_length_and_payload_sha256_manifest_v5",
            "network_backend": self.backend,
            "native_verification_workers": 2,
            "socket_sndbuf_bytes": self.sock.getsockopt(
                socket.SOL_SOCKET, socket.SO_SNDBUF
            ),
            "socket_rcvbuf_bytes": self.sock.getsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF
            ),
            "send_intervals": sends,
            "producer_wait_intervals": waits,
            "producer_wait_seconds": sum(w["end"] - w["start"] for w in waits),
            "post_start_producer_wait_seconds": sum(
                w["end"] - w["start"] for w in waits[1:]
            ),
            "sender_elapsed_seconds": sum(s["end"] - s["start"] for s in sends),
            "pacing_sleep_seconds": sum(s["pacing_sleep_seconds"] for s in sends),
            "socket_call_seconds": sum(s["socket_call_seconds"] for s in sends),
            "timed_major_page_faults": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_majflt
            - faults,
        }

    def close(self):
        self.requests.put(None)
        self.sock.close()
        self.thread.join(timeout=10)
        if self.thread.is_alive() or self.errors:
            raise RuntimeError({"client": self.errors, "alive": self.thread.is_alive()})
