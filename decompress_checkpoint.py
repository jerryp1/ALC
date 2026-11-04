#!/usr/bin/env python3
"""Restore checkpoints through the original file/directory command interface."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch

from checkpoint_archive import Archive


def _default_restored_path(input_path, output_dir):
    name = Path(input_path).name.removeprefix("compressed_")
    if not name.endswith((".pt", ".pth")):
        name += ".pt"
    return Path(output_dir) / ("restored_" + name)


def restore_pt_checkpoint(
    input_path,
    metadata_path=None,
    base_path=None,
    output_path=None,
    verbose=True,
    version=0,
):
    """Restore an archive version, including BF16 and non-tensor metadata."""
    archive = Archive(input_path)
    if (
        metadata_path is not None
        and Path(metadata_path).resolve() != (archive.path / "manifest.json").resolve()
    ):
        raise ValueError(
            "This archive uses its own manifest.json; old PT_V1 sidecars are unsupported"
        )
    if base_path and verbose:
        print(
            "Reference is already retained in the archive; --base is unnecessary",
            file=sys.stderr,
        )
    output = (
        Path(output_path) if output_path else _default_restored_path(input_path, ".")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(archive.restore(version), output)
    report = {"output_path": str(output), "version": version}
    if verbose:
        print(json.dumps(report, indent=2))
    return report


def discover_compressed_pt_checkpoints(input_dir):
    return sorted(path.parent for path in Path(input_dir).rglob("manifest.json"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="ALC checkpoint restoration")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", help="One archive directory")
    inputs.add_argument("--input_dir", help="Directory containing independent archives")
    parser.add_argument(
        "--metadata", help="Optional path to the archive's manifest.json"
    )
    parser.add_argument(
        "--base", help="Accepted for the old CLI; references are inside the archive"
    )
    parser.add_argument(
        "--base_dir", help="Accepted for the old CLI; references are inside archives"
    )
    parser.add_argument("--output")
    parser.add_argument("--output_dir", default="restored_output")
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.input_dir and (args.output or args.metadata):
        parser.error("Use --output_dir and each archive's own manifest for a batch")
    if args.base or args.base_dir:
        print(
            "References are already retained in the archive; external base arguments are unused",
            file=sys.stderr,
        )
    if args.input:
        output = args.output or _default_restored_path(args.input, args.output_dir)
        reports = [
            restore_pt_checkpoint(
                args.input,
                args.metadata,
                output_path=output,
                verbose=False,
                version=args.version,
            )
        ]
    else:
        archives = discover_compressed_pt_checkpoints(args.input_dir)
        if not archives:
            parser.error("No ALC archive manifests found")
        reports = []
        for archive in archives:
            relative = archive.relative_to(args.input_dir)
            output = _default_restored_path(
                archive, Path(args.output_dir) / relative.parent
            )
            reports.append(
                restore_pt_checkpoint(
                    archive, output_path=output, verbose=False, version=args.version
                )
            )
    if not args.quiet:
        print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
