#!/usr/bin/env python3

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from checkpoint_io import (
    TensorDict,
    bytes_to_uint8_tensor,
    discover_checkpoints,
    iter_tensor_leaves,
    load_pt_checkpoint,
    read_checkpoint,
    save_pt_checkpoint,
    set_by_typed_path,
    tensor_to_float32_array,
)
import tensor_compress as tc


SUPPORTED_DTYPES = {"F32", "BF16", "F16"}


def _serialize_chunks(all_chunks):
    """Serialize compress_tensor output to bytes for file storage."""
    import pickle
    return pickle.dumps(all_chunks, protocol=4)


def _strategy_to_dict(strategy):
    return {
        "path": strategy.path,
        "encoding": strategy.encoding,
        "ent_ialign": strategy.ent_ialign,
        "ent_pcdelta": strategy.ent_pcdelta,
        "ent_rotdelta": strategy.ent_rotdelta,
    }


def _default_output_path(input_path, output_dir):
    return os.path.join(output_dir, "compressed_" + os.path.basename(input_path))


def _find_base_array(base_tensors, name, n_elems):
    if base_tensors is None or name not in base_tensors:
        return None
    base_arr = base_tensors[name]
    if len(base_arr) != n_elems:
        return None
    return base_arr


def _add_timing(timings, key, seconds):
    timings[key] = timings.get(key, 0.0) + float(seconds)


def _time_call(timings, key, fn, *args, **kwargs):
    t0 = time.time()
    value = fn(*args, **kwargs)
    _add_timing(timings, key, time.time() - t0)
    return value


def compress_pt_checkpoint(filepath, base_tensors=None, output_path=None, verbose=True):
    timings = {"per_tensor": []}
    total_t0 = time.time()

    state = _time_call(timings, "target_load_pt_seconds", load_pt_checkpoint, filepath)
    t0 = time.time()
    leaves = list(iter_tensor_leaves(state))
    _add_timing(timings, "iter_tensor_leaves_seconds", time.time() - t0)

    if output_path is None:
        output_path = "compressed_" + os.path.basename(filepath)

    if verbose:
        header = "%-50s %-20s %8s %8s %6s %9s" % (
            "Tensor", "Shape", "Size(MB)", "Path", "Ratio", "Enc MB/s")
        print(header)
        print("=" * 110)

    tensor_entries = []
    skipped_entries = []
    total_orig = 0
    total_comp = 0
    compressed_count = 0

    for leaf in leaves:
        leaf_timing = {"name": leaf["name"]}
        leaf_total_t0 = time.time()
        name = leaf["name"]
        tensor = leaf["tensor"]
        dtype_str = leaf["dtype"]
        shape = leaf["shape"]
        n_elems = leaf["numel"]
        original_bytes = int(tensor.numel() * tensor.element_size())
        total_orig += original_bytes

        if n_elems < 100 or dtype_str not in SUPPORTED_DTYPES:
            total_comp += original_bytes
            skipped_entries.append({
                "name": name,
                "path": leaf["path"],
                "shape": shape,
                "dtype": dtype_str,
                "n_elems": n_elems,
                "original_bytes": original_bytes,
                "reason": "too_small" if n_elems < 100 else "unsupported_dtype",
            })
            leaf_timing["skipped"] = True
            leaf_timing["total_seconds"] = time.time() - leaf_total_t0
            timings["per_tensor"].append(leaf_timing)
            continue

        t0 = time.time()
        ft_arr, dtype_str = tensor_to_float32_array(tensor)
        dt = time.time() - t0
        _add_timing(timings, "target_tensor_to_float32_seconds", dt)
        leaf_timing["target_tensor_to_float32_seconds"] = dt

        t0 = time.time()
        base_arr = _find_base_array(base_tensors, name, len(ft_arr))
        dt = time.time() - t0
        _add_timing(timings, "base_lookup_seconds", dt)
        leaf_timing["base_lookup_seconds"] = dt

        t0 = time.time()
        strategy = tc.analyze_strategy(ft_arr, base_arr)
        dt = time.time() - t0
        _add_timing(timings, "analyze_strategy_seconds", dt)
        leaf_timing["analyze_strategy_seconds"] = dt

        t0 = time.time()
        codec_bytes, chunks, _ = tc.compress_tensor(ft_arr, strategy, base_arr)
        enc_time = time.time() - t0
        _add_timing(timings, "codec_encode_seconds", enc_time)
        leaf_timing["codec_encode_seconds"] = enc_time

        t0 = time.time()
        chunk_bytes = _serialize_chunks(chunks)
        dt = time.time() - t0
        _add_timing(timings, "pickle_serialize_seconds", dt)
        leaf_timing["pickle_serialize_seconds"] = dt
        compressed_bytes = len(chunk_bytes)

        t0 = time.time()
        payload_tensor = bytes_to_uint8_tensor(chunk_bytes)
        dt = time.time() - t0
        _add_timing(timings, "bytes_to_uint8_tensor_seconds", dt)
        leaf_timing["bytes_to_uint8_tensor_seconds"] = dt

        t0 = time.time()
        state = set_by_typed_path(state, leaf["path"], payload_tensor)
        dt = time.time() - t0
        _add_timing(timings, "set_by_typed_path_seconds", dt)
        leaf_timing["set_by_typed_path_seconds"] = dt

        ratio = compressed_bytes / float(original_bytes) if original_bytes else 0.0
        enc_mbs = original_bytes / enc_time / 1e6 if enc_time > 0 else float("inf")
        total_comp += compressed_bytes
        compressed_count += 1

        entry = {
            "name": name,
            "path": leaf["path"],
            "shape": shape,
            "dtype": dtype_str,
            "n_elems": int(len(ft_arr)),
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "codec_bytes": int(codec_bytes),
            "ratio": ratio,
            "strategy": _strategy_to_dict(strategy),
            "requires_base": strategy.path in (tc.PATH_PCDELTA, tc.PATH_ROTDELTA),
            "payload": "torch.uint8_pickle_chunks",
            "encode_mbps": enc_mbs,
        }
        tensor_entries.append(entry)

        leaf_timing.update({
            "skipped": False,
            "path": strategy.path,
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "total_seconds": time.time() - leaf_total_t0,
        })
        timings["per_tensor"].append(leaf_timing)

        if verbose:
            print("%-50s %-20s %8.2f %8s %6.4f %9.1f" % (
                name[:50], str(shape)[:20], original_bytes / 1e6,
                strategy.path, ratio, enc_mbs))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    t0 = time.time()
    save_pt_checkpoint(state, output_path)
    _add_timing(timings, "save_compressed_pt_seconds", time.time() - t0)

    metadata = {
        "format": "ALC_COMPRESSED_PT_V1",
        "source_checkpoint": filepath,
        "compressed_checkpoint": output_path,
        "metadata_path": output_path + ".json",
        "tensor_count": len(leaves),
        "compressed_tensor_count": compressed_count,
        "skipped_tensor_count": len(skipped_entries),
        "total_original_bytes": total_orig,
        "total_compressed_payload_bytes": total_comp,
        "overall_payload_ratio": total_comp / float(total_orig) if total_orig > 0 else 0,
        "tensors": tensor_entries,
        "skipped_tensors": skipped_entries,
    }

    timings["total_seconds_before_metadata_write"] = time.time() - total_t0
    metadata["timings"] = timings

    t0 = time.time()
    with open(output_path + ".json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    _add_timing(timings, "metadata_write_seconds", time.time() - t0)
    timings["total_seconds"] = time.time() - total_t0

    if verbose:
        print("=" * 110)
        print("%-50s %-20s %8.2f %8s %6.4f" % (
            "TOTAL", "", total_orig / 1e6, "",
            total_comp / float(total_orig) if total_orig > 0 else 0))
        print("Tensors: %d compressed, %d skipped" % (compressed_count, len(skipped_entries)))
        print("Saved: %s" % output_path)
        print("Metadata: %s" % (output_path + ".json"))

    return metadata


def compress_tensor_dict(ft_tensors, base_tensors=None, verbose=True, output_dir=None):
    results = []
    total_orig = 0
    total_comp = 0

    if verbose:
        header = "%-50s %-20s %8s %8s %6s %9s" % (
            "Tensor", "Shape", "Size(MB)", "Path", "Ratio", "Enc MB/s")
        print(header)
        print("=" * 110)

    for name in ft_tensors:
        ft_arr = ft_tensors[name]
        shape = ft_tensors.metadata[name]["shape"]
        dtype_str = ft_tensors.metadata[name]["dtype"]
        n = len(ft_arr)

        if n < 100:
            continue

        orig_bytes = n * 4
        total_orig += orig_bytes

        base_arr = None
        if base_tensors is not None and name in base_tensors:
            base_arr = base_tensors[name]

        strategy = tc.analyze_strategy(ft_arr, base_arr)

        t0 = time.time()
        comp_sz, chunks, _ = tc.compress_tensor(ft_arr, strategy, base_arr)
        enc_time = time.time() - t0
        chunk_bytes = _serialize_chunks(chunks)

        ratio = len(chunk_bytes) / float(orig_bytes)
        enc_mbs = orig_bytes / enc_time / 1e6 if enc_time > 0 else float("inf")
        total_comp += len(chunk_bytes)

        if verbose:
            print("%-50s %-20s %8.2f %8s %6.4f %9.1f" % (
                name[:50], str(shape)[:20], orig_bytes / 1e6,
                strategy.path, ratio, enc_mbs))

        results.append({
            "name": name,
            "shape": list(shape),
            "dtype": dtype_str,
            "original_bytes": orig_bytes,
            "compressed_bytes": len(chunk_bytes),
            "codec_bytes": comp_sz,
            "ratio": ratio,
            "path": strategy.path,
            "encode_mbps": enc_mbs,
        })

    if verbose:
        print("=" * 110)
        overall_ratio = total_comp / float(total_orig) if total_orig > 0 else 0
        print("%-50s %-20s %8.2f %8s %6.4f" % (
            "TOTAL", "", total_orig / 1e6, "", overall_ratio))
        print("Tensors: %d" % len(results))

    return results, total_orig, total_comp


def find_matching_base(base_dir, ft_filename):
    """Find the matching base file for a given finetuned file."""
    if base_dir is None:
        return None

    candidate = os.path.join(base_dir, ft_filename)
    if os.path.exists(candidate):
        return candidate

    for root, dirs, files in os.walk(base_dir):
        if ft_filename in files:
            return os.path.join(root, ft_filename)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="ALC compression for PyTorch .pt checkpoints with preserved container structure")
    parser.add_argument("--input", type=str, help="Single input checkpoint file")
    parser.add_argument("--input_dir", type=str, help="Input directory to scan for checkpoints")
    parser.add_argument("--base", type=str, help="Base checkpoint file or directory (for delta)")
    parser.add_argument("--base_dir", type=str, help="Base checkpoint directory (for delta)")
    parser.add_argument("--output", type=str, help="Output file (single-file mode)")
    parser.add_argument("--output_dir", type=str, default="./compressed_output",
                        help="Output directory (default: ./compressed_output)")
    parser.add_argument("--filter", type=str, default=None,
                        help="Glob filter for files (e.g., '*.pt')")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-tensor output")
    parser.add_argument("--json_report", type=str, default=None,
                        help="Save JSON report to this path")
    args = parser.parse_args()

    verbose = not args.quiet

    input_files = []
    if args.input:
        input_files.append(args.input)
    elif args.input_dir:
        if args.filter:
            pattern = os.path.join(args.input_dir, "**", args.filter)
            input_files = sorted(glob.glob(pattern, recursive=True))
        else:
            discovered = discover_checkpoints(args.input_dir)
            input_files = [fp for fp, cat in discovered]
    else:
        parser.error("Must specify --input or --input_dir")

    if not input_files:
        print("No checkpoint files found.")
        sys.exit(1)

    if args.output and len(input_files) != 1:
        parser.error("--output can only be used with a single --input file")

    base_dir = args.base_dir or (os.path.dirname(args.base) if args.base else None)
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    grand_orig = 0
    grand_comp = 0

    for filepath in input_files:
        if not (filepath.endswith(".pt") or filepath.endswith(".pth")):
            raise ValueError("PT container output requires a .pt or .pth input: %s" % filepath)

        filename = os.path.basename(filepath)
        if verbose:
            print("\n>>> %s" % filepath)

        base_tensors = None
        if args.base and os.path.isfile(args.base):
            base_tensors = read_checkpoint(args.base)
        elif base_dir:
            base_path = find_matching_base(base_dir, filename)
            if base_path and base_path != filepath:
                if verbose:
                    print("  Base: %s" % base_path)
                base_tensors = read_checkpoint(base_path)

        output_path = args.output if args.output else _default_output_path(filepath, args.output_dir)
        metadata = compress_pt_checkpoint(
            filepath, base_tensors=base_tensors, output_path=output_path, verbose=verbose)

        all_results[filepath] = metadata
        grand_orig += metadata["total_original_bytes"]
        grand_comp += metadata["total_compressed_payload_bytes"]

    print("\n" + "=" * 80)
    print("GRAND TOTAL: %.2f MB → %.2f MB (payload ratio: %.4f)" % (
        grand_orig / 1e6, grand_comp / 1e6,
        grand_comp / float(grand_orig) if grand_orig > 0 else 0))
    print("Files processed: %d" % len(input_files))

    if args.json_report:
        report = {
            "format": "ALC_COMPRESSED_PT_REPORT_V1",
            "files": all_results,
            "grand_total_original_bytes": grand_orig,
            "grand_total_compressed_payload_bytes": grand_comp,
            "grand_overall_payload_ratio": grand_comp / float(grand_orig) if grand_orig > 0 else 0,
        }
        with open(args.json_report, "w") as fh:
            json.dump(report, fh, indent=2)
        print("Report saved to: %s" % args.json_report)


if __name__ == "__main__":
    main()
