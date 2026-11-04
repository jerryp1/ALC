#!/usr/bin/env python3
"""Inspect/query archives or transfer a complete checkpoint over TCP."""

import argparse
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch

from checkpoint_archive import Archive
from serving import query_local, receive_query, send_query


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("archive")
    query = commands.add_parser(
        "query", help="Local paced socket with payload verification"
    )
    query.add_argument("archive")
    query.add_argument("--version", type=int, required=True)
    query.add_argument("--workers", type=int, default=8)
    query.add_argument("--bandwidth-gbps", type=float, default=8)
    query.add_argument("--resident", action="store_true")
    query.add_argument("--memory-mib", type=float, default=1024)
    serve = commands.add_parser("serve", help="Serve one request over TCP")
    serve.add_argument("archive")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=9000)
    serve.add_argument("--workers", type=int, default=8)
    serve.add_argument("--bandwidth-gbps", type=float)
    fetch = commands.add_parser(
        "fetch", help="Receive a complete, verified PyTorch state"
    )
    fetch.add_argument("--host", default="127.0.0.1")
    fetch.add_argument("--port", type=int, default=9000)
    fetch.add_argument("--version", type=int, required=True)
    fetch.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        result = Archive(args.archive).stats()
    elif args.command == "query":
        result = query_local(
            args.archive,
            args.version,
            args.workers,
            args.bandwidth_gbps,
            args.resident,
            int(args.memory_mib * 2**20),
        )
    elif args.command == "serve":
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.host, args.port))
            listener.listen(1)
            print(json.dumps({"listening": listener.getsockname()}), flush=True)
            with listener.accept()[0] as connection:
                with connection.makefile("rb", buffering=0) as stream:
                    request = stream.readline(4096)
                if not request.endswith(b"\n"):
                    raise ValueError("Invalid version request")
                result = send_query(
                    args.archive,
                    int(json.loads(request)["version"]),
                    connection,
                    args.workers,
                    args.bandwidth_gbps,
                )
    else:
        with socket.create_connection((args.host, args.port), timeout=60) as connection:
            connection.sendall(json.dumps({"version": args.version}).encode() + b"\n")
            state, result = receive_query(connection)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, args.output)
        result["output"] = args.output
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
