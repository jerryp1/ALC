#!/usr/bin/env python3
"""Compress checkpoints through the original file/directory command interface."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from checkpoint_archive import compress


def compress_pt_checkpoint(
    filepath, base_path=None, output_path=None, verbose=True, **options
):
    """Archive a target and optional newer reference; version zero is the target."""
    paths = [filepath] if base_path is None else [filepath, base_path]
    output = output_path or ("compressed_" + Path(filepath).name)
    archive = compress(
        paths, output, retain_latest_raw=base_path is not None, **options
    )
    report = {"output_path": str(archive.path), **archive.stats()}
    if verbose:
        print(json.dumps(report, indent=2))
    return report


def find_matching_base(base_dir, target, input_dir):
    if base_dir is None:
        return None
    relative = Path(target).relative_to(input_dir)
    for candidate in [Path(base_dir) / relative, Path(base_dir) / relative.name]:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No matching base checkpoint for " + str(relative))


def main(argv=None):
    parser = argparse.ArgumentParser(description="ALC checkpoint compression")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", help="One trusted PyTorch checkpoint")
    inputs.add_argument(
        "--input_dir", help="Batch of independent PyTorch checkpoint files"
    )
    inputs.add_argument(
        "--checkpoints", nargs="+", help="One version chain, ordered oldest to newest"
    )
    parser.add_argument(
        "--base", help="Newer reference checkpoint; stored once as latest Raw"
    )
    parser.add_argument(
        "--base_dir", help="Matching newer references for directory mode"
    )
    parser.add_argument("--output", help="Single archive directory")
    parser.add_argument("--output_dir", default="compressed_output")
    parser.add_argument("--filter", help="Only batch filenames containing this text")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json_report")
    parser.add_argument(
        "--group_size",
        "--group-size",
        type=int,
        default=None,
        help="Versions per group; default: no grouping (all input versions)",
    )
    parser.add_argument("--chunk_elements", "--chunk-elements", type=int, default=2**18)
    parser.add_argument(
        "--sample_elements", "--sample-elements", type=int, default=65536
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--compression_memory_mib", type=float, default=512)
    parser.add_argument(
        "--bf16_pairs", help="JSON mapping BF16 names to FP32 master names/spans"
    )
    args = parser.parse_args(argv)
    if args.checkpoints and (args.base or args.base_dir):
        parser.error("--checkpoints already includes its latest reference")
    if args.input_dir and args.output:
        parser.error("Use --output_dir for a directory batch")
    if not args.input_dir and args.base_dir:
        parser.error("--base_dir requires --input_dir")
    if args.base and args.base_dir:
        parser.error("Use either --base or --base_dir")
    pairs = json.loads(Path(args.bf16_pairs).read_text()) if args.bf16_pairs else None
    options = dict(
        group_size=args.group_size,
        chunk_elems=args.chunk_elements,
        sample_size=args.sample_elements,
        compression_workers=args.workers,
        compression_memory_bytes=int(args.compression_memory_mib * 2**20),
        bf16_pairs=pairs,
    )
    if args.checkpoints:
        archive = compress(args.checkpoints, args.output or args.output_dir, **options)
        reports = [{"output_path": str(archive.path), **archive.stats()}]
    elif args.input:
        if args.base and not Path(args.base).is_file():
            parser.error("--base must be a checkpoint file in single-file mode")
        output = args.output or str(
            Path(args.output_dir) / ("compressed_" + Path(args.input).name)
        )
        reports = [
            compress_pt_checkpoint(args.input, args.base, output, False, **options)
        ]
    else:
        root = Path(args.input_dir)
        files = sorted(
            p for p in root.rglob("*") if p.suffix in (".pt", ".pth") and p.is_file()
        )
        files = [p for p in files if not args.filter or args.filter in p.name]
        if not files:
            parser.error("No matching .pt/.pth checkpoint files")
        reports = []
        base_dir = args.base_dir or (
            args.base if args.base and Path(args.base).is_dir() else None
        )
        for path in files:
            base = find_matching_base(base_dir, path, root) if base_dir else args.base
            relative = path.relative_to(root)
            output = (
                Path(args.output_dir)
                / relative.parent
                / ("compressed_" + relative.name)
            )
            reports.append(compress_pt_checkpoint(path, base, output, False, **options))
    if args.json_report:
        Path(args.json_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_report).write_text(json.dumps(reports, indent=2) + "\n")
    if not args.quiet:
        print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
