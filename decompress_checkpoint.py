#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from checkpoint_io import (
    float32_array_to_tensor,
    get_by_typed_path,
    load_pt_checkpoint,
    save_pt_checkpoint,
    set_by_typed_path,
    tensor_to_float32_array,
    uint8_tensor_to_bytes,
)
import tensor_compress as tc


def _deserialize_chunks(data):
    """Deserialize chunks from bytes."""
    import pickle
    return pickle.loads(data)


def _strategy_from_dict(strategy_dict):
    return tc.CompressStrategy(
        path=strategy_dict["path"],
        encoding=strategy_dict.get("encoding", "TANS"),
        ent_ialign=strategy_dict.get("ent_ialign", 0),
        ent_pcdelta=strategy_dict.get("ent_pcdelta", 0),
        ent_rotdelta=strategy_dict.get("ent_rotdelta", 0),
    )


def _default_restored_path(input_path, output_dir):
    filename = os.path.basename(input_path)
    if filename.startswith("compressed_"):
        filename = filename[len("compressed_"):]
    return os.path.join(output_dir, "restored_" + filename)


def _add_timing(timings, key, seconds):
    timings[key] = timings.get(key, 0.0) + float(seconds)


def _time_call(timings, key, fn, *args, **kwargs):
    t0 = time.time()
    value = fn(*args, **kwargs)
    _add_timing(timings, key, time.time() - t0)
    return value


def _base_array_for_entry(base_state, entry):
    if base_state is None:
        return None
    base_tensor = get_by_typed_path(base_state, entry["path"])
    base_arr, _ = tensor_to_float32_array(base_tensor)
    if len(base_arr) != entry["n_elems"]:
        raise ValueError("base tensor shape does not match compressed tensor: %s" % entry["name"])
    return base_arr


def restore_pt_checkpoint(input_path, metadata_path=None, base_path=None, output_path=None, verbose=True):
    timings = {"per_tensor": []}
    total_t0 = time.time()

    if metadata_path is None:
        metadata_path = input_path + ".json"
    if output_path is None:
        output_path = _default_restored_path(input_path, os.path.dirname(os.path.abspath(input_path)))

    t0 = time.time()
    with open(metadata_path, "r") as fh:
        metadata = json.load(fh)
    _add_timing(timings, "metadata_read_seconds", time.time() - t0)

    if metadata.get("format") != "ALC_COMPRESSED_PT_V1":
        raise ValueError("unsupported metadata format: %s" % metadata.get("format"))

    state = _time_call(timings, "compressed_load_pt_seconds", load_pt_checkpoint, input_path)
    base_state = None
    if base_path:
        base_state = _time_call(timings, "base_load_pt_seconds", load_pt_checkpoint, base_path)

    if verbose:
        header = "%-50s %-20s %8s %9s" % (
            "Tensor", "Shape", "Path", "Dec MB/s")
        print(header)
        print("=" * 95)

    total_dec_time = 0.0
    restored_count = 0

    for entry in metadata["tensors"]:
        tensor_timing = {"name": entry["name"], "path": entry["strategy"]["path"]}
        tensor_total_t0 = time.time()

        t0 = time.time()
        payload_tensor = get_by_typed_path(state, entry["path"])
        dt = time.time() - t0
        _add_timing(timings, "payload_lookup_seconds", dt)
        tensor_timing["payload_lookup_seconds"] = dt

        t0 = time.time()
        chunk_bytes = uint8_tensor_to_bytes(payload_tensor)
        dt = time.time() - t0
        _add_timing(timings, "uint8_tensor_to_bytes_seconds", dt)
        tensor_timing["uint8_tensor_to_bytes_seconds"] = dt

        t0 = time.time()
        chunks = _deserialize_chunks(chunk_bytes)
        dt = time.time() - t0
        _add_timing(timings, "pickle_deserialize_seconds", dt)
        tensor_timing["pickle_deserialize_seconds"] = dt

        t0 = time.time()
        strategy = _strategy_from_dict(entry["strategy"])
        dt = time.time() - t0
        _add_timing(timings, "strategy_from_dict_seconds", dt)
        tensor_timing["strategy_from_dict_seconds"] = dt

        base_arr = None
        if entry.get("requires_base"):
            t0 = time.time()
            base_arr = _base_array_for_entry(base_state, entry)
            dt = time.time() - t0
            _add_timing(timings, "base_array_seconds", dt)
            tensor_timing["base_array_seconds"] = dt
            if base_arr is None:
                raise ValueError("base checkpoint is required for tensor: %s" % entry["name"])

        t0 = time.time()
        rec, _ = tc.decompress_tensor(chunks, strategy, entry["n_elems"], base_arr)
        dec_time = time.time() - t0
        _add_timing(timings, "codec_decode_seconds", dec_time)
        tensor_timing["codec_decode_seconds"] = dec_time
        total_dec_time += dec_time

        t0 = time.time()
        restored_tensor = float32_array_to_tensor(rec, entry["dtype"], entry["shape"])
        dt = time.time() - t0
        _add_timing(timings, "float32_array_to_tensor_seconds", dt)
        tensor_timing["float32_array_to_tensor_seconds"] = dt

        t0 = time.time()
        state = set_by_typed_path(state, entry["path"], restored_tensor)
        dt = time.time() - t0
        _add_timing(timings, "set_by_typed_path_seconds", dt)
        tensor_timing["set_by_typed_path_seconds"] = dt

        restored_count += 1
        tensor_timing["original_bytes"] = entry["original_bytes"]
        tensor_timing["total_seconds"] = time.time() - tensor_total_t0
        timings["per_tensor"].append(tensor_timing)

        if verbose:
            dec_mbs = entry["original_bytes"] / dec_time / 1e6 if dec_time > 0 else float("inf")
            print("%-50s %-20s %8s %9.1f" % (
                entry["name"][:50], str(entry["shape"])[:20], strategy.path, dec_mbs))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    t0 = time.time()
    save_pt_checkpoint(state, output_path)
    _add_timing(timings, "save_restored_pt_seconds", time.time() - t0)
    timings["total_seconds"] = time.time() - total_t0

    if verbose:
        print("=" * 95)
        print("Tensors restored: %d, Total decode time: %.1fs" % (restored_count, total_dec_time))
        print("Saved: %s" % output_path)

    return {
        "output_path": output_path,
        "restored_count": restored_count,
        "total_decode_time": total_dec_time,
        "timings": timings,
    }


def _is_current_metadata(metadata_path):
    try:
        with open(metadata_path, "r") as fh:
            metadata = json.load(fh)
        return metadata.get("format") == "ALC_COMPRESSED_PT_V1"
    except (OSError, ValueError):
        return False


def discover_compressed_pt_checkpoints(input_dir):
    checkpoints = []
    for root, dirs, files in os.walk(input_dir):
        for fname in sorted(files):
            if not (fname.endswith(".pt") or fname.endswith(".pth")):
                continue
            path = os.path.join(root, fname)
            metadata_path = path + ".json"
            if os.path.exists(metadata_path) and _is_current_metadata(metadata_path):
                checkpoints.append((path, metadata_path))
    return checkpoints


def _original_filename_from_metadata(input_path, metadata_path):
    with open(metadata_path, "r") as fh:
        metadata = json.load(fh)
    source = metadata.get("source_checkpoint")
    if source:
        return os.path.basename(source)
    filename = os.path.basename(input_path)
    if filename.startswith("compressed_"):
        return filename[len("compressed_"):]
    return filename


def find_matching_base(base_dir, original_filename):
    if base_dir is None:
        return None
    candidate = os.path.join(base_dir, original_filename)
    if os.path.exists(candidate):
        return candidate
    for root, dirs, files in os.walk(base_dir):
        if original_filename in files:
            return os.path.join(root, original_filename)
    return None


def restore_pt_checkpoints_in_dir(input_dir, base_path=None, base_dir=None, output_dir="./restored_output", verbose=True):
    checkpoints = discover_compressed_pt_checkpoints(input_dir)
    if not checkpoints:
        raise ValueError("no compressed .pt/.pth files with ALC metadata found in %s" % input_dir)

    os.makedirs(output_dir, exist_ok=True)
    results = []
    for input_path, metadata_path in checkpoints:
        original_filename = _original_filename_from_metadata(input_path, metadata_path)
        file_base_path = base_path or find_matching_base(base_dir, original_filename)
        output_path = os.path.join(output_dir, "restored_" + original_filename)

        if verbose:
            print("\n>>> %s" % input_path)
            if file_base_path:
                print("  Base: %s" % file_base_path)

        result = restore_pt_checkpoint(
            input_path,
            metadata_path=metadata_path,
            base_path=file_base_path,
            output_path=output_path,
            verbose=verbose,
        )
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Decompress ALC compressed .pt checkpoints")
    parser.add_argument("--input", type=str,
                        help="Compressed .pt checkpoint file")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Metadata JSON path (default: <input>.json)")
    parser.add_argument("--input_dir", type=str,
                        help="Directory containing compressed .pt/.pth checkpoints")
    parser.add_argument("--base", type=str, default=None,
                        help="Base checkpoint file (for delta decompression)")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Base checkpoint directory (for directory decompression)")
    parser.add_argument("--output", type=str, default=None,
                        help="Restored .pt output path")
    parser.add_argument("--output_dir", type=str, default="./restored_output",
                        help="Output directory (default: ./restored_output)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-tensor output")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.input:
        output_path = args.output or _default_restored_path(args.input, args.output_dir)
        restore_pt_checkpoint(
            args.input,
            metadata_path=args.metadata,
            base_path=args.base,
            output_path=output_path,
            verbose=verbose,
        )
        return

    if args.input_dir:
        results = restore_pt_checkpoints_in_dir(
            args.input_dir,
            base_path=args.base,
            base_dir=args.base_dir,
            output_dir=args.output_dir,
            verbose=verbose,
        )
        if verbose:
            print("\nRestored files: %d" % len(results))
            print("Output directory: %s" % args.output_dir)
        return

    parser.error("Must specify --input or --input_dir")


if __name__ == "__main__":
    main()
