import shutil

import numpy as np
import pytest
import torch
from conftest import equal, states

import bf16
import checkpoint_io as _tree
from checkpoint_archive import Archive, anchor, compress


def test_complete_states_metadata_dedup_and_single_raw_head(tmp_path):
    original = states()
    archive = compress(
        original, tmp_path / "archive", chunk_elems=128, compression_workers=2
    )
    assert archive.manifest["group_size"] == len(original)
    assert (archive.path / "latest.pt").is_file()
    assert not (archive.path / "v000004.pack").exists()
    assert not (archive.path / "v000004.metadata.pkl").exists()
    for version in range(5):
        recovered = archive.restore(version)
        equal(original[version], recovered)
        assert recovered["tied"] is recovered["model"]["weight"]
        entries = {s["name"]: s for s in archive.version(version)["tensors"].values()}
        if version < 4:
            assert entries["model.weight"]["mode"] == "DerivedBF16"
            assert entries["model.weight"]["chunks"] == []
            assert entries["model.unmatched"]["mode"] == "Direct"
            assert entries["frozen"]["mode"] == "FrozenRef"
        else:
            assert all(s["mode"] == "Raw" for s in entries.values())
    expected = sum(
        len(_tree.raw_view(e["tensor"]))
        for state in original
        for e in _tree.extract(state)[1].values()
    )
    assert archive.stats()["original_tensor_bytes"] == expected
    assert archive.stats()["archive_bytes"] == sum(
        p.stat().st_size for p in archive.path.iterdir()
    )


def test_default_disables_groups_and_explicit_group_size_keeps_local_anchors(tmp_path):
    original = states(count=8, elements=129)
    default = compress(original, tmp_path / "default", chunk_elems=64)
    grouped = compress(original, tmp_path / "grouped", group_size=2, chunk_elems=64)
    assert default.manifest["group_size"] == 8
    assert grouped.manifest["group_size"] == 2
    assert all(
        anchor(version, default.manifest["group_size"], 8) == 7 for version in range(8)
    )
    assert not any(
        spec.get("selection", {}).get("forced_group_anchor")
        for version in default.manifest["versions"]
        for spec in version["tensors"].values()
    )
    assert any(
        spec.get("selection", {}).get("forced_group_anchor")
        for spec in grouped.version(1)["tensors"].values()
    )
    equal(original[0], default.restore(0))
    equal(original[0], grouped.restore(0))


@pytest.mark.parametrize("group", [2, 4, 6, 8, 20])
def test_group_anchors_reverse_chains_and_partial_groups(tmp_path, group):
    original = states(count=20, elements=17)
    archive = compress(
        original,
        tmp_path / "archive",
        group_size=group,
        chunk_elems=8,
        compression_workers=2,
    )
    for version in range(20):
        expected = anchor(version, group, 20)
        assert expected - version <= group - 1
        assert all(
            len(job["layers"]) - 1 <= expected - version
            for job in archive.jobs(version)
        )
        if version < 19 and version == expected:
            assert all(
                s["mode"] in ("Direct", "Bytes", "DerivedBF16", "FrozenRef")
                for s in archive.version(version)["tensors"].values()
            )
    if group == 6:
        assert anchor(18, group, 20) == 19
    if group == 20:
        assert all(anchor(v, group, 20) == 19 for v in range(20))
    equal(original[9], archive.restore(9))


def test_explicit_flat_master_span_and_truncation(tmp_path):
    x = torch.tensor([9.0, 1.006, 1.018, -1.006, -1.018, 7.0], dtype=torch.float32)
    bits = (x.numpy().view(np.uint32)[1:5] >> 16).astype(np.uint16)
    state = {
        "flat": x,
        "module": {
            "w": torch.from_numpy(bits.view(np.int16).copy())
            .view(torch.bfloat16)
            .reshape(2, 2)
        },
    }
    archive = compress(
        [state, state],
        tmp_path / "archive",
        chunk_elems=2,
        bf16_pairs={"module.w": {"source": "flat", "start": 1}},
    )
    entry = next(
        s for s in archive.version(0)["tensors"].values() if s["name"] == "module.w"
    )
    assert entry["derivation"]["rule"] == "truncate"
    equal(state, archive.restore(0))


def test_bf16_failed_derivation_retains_original_payload(tmp_path):
    original = states(count=2)
    original[0]["model"]["weight"][0] = 8.0
    archive = compress(original, tmp_path / "archive", chunk_elems=128)
    spec = next(
        s for s in archive.version(0)["tensors"].values() if s["name"] == "model.weight"
    )
    assert spec["mode"] == "Direct" and spec["chunks"]
    equal(original[0], archive.restore(0))


def test_shape_change_has_no_delta_reference(tmp_path):
    original = [{"w": torch.linspace(-1.0, 1.0, n)} for n in [17, 19, 23]]
    archive = compress(original, tmp_path / "archive", group_size=20, chunk_elems=8)
    for version in [0, 1]:
        assert (
            next(iter(archive.version(version)["tensors"].values()))["mode"] == "Direct"
        )
        equal(original[version], archive.restore(version))


def test_compressed_archive_needs_no_original_inputs(tmp_path):
    folder = tmp_path / "inputs"
    folder.mkdir()
    original = states(3)
    paths = []
    for i, state in enumerate(original):
        path = folder / f"{i}.pt"
        torch.save(state, path)
        paths.append(path)
    compress(paths, tmp_path / "archive", chunk_elems=128)
    shutil.rmtree(folder)
    equal(original[0], Archive(tmp_path / "archive").restore(0))


@pytest.mark.parametrize("which", ["pack", "metadata", "latest"])
def test_corruption_is_rejected(tmp_path, which):
    archive = compress(states(3), tmp_path / "archive", chunk_elems=128)
    path = (
        next(archive.path.glob("*.pack"))
        if which == "pack"
        else archive.path
        / ("v000000.metadata.pkl" if which == "metadata" else "latest.pt")
    )
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 1
    path.write_bytes(raw)
    with pytest.raises((ValueError, RuntimeError)):
        if which == "pack":
            for version in range(2):
                archive.restore(version)
        else:
            archive.restore(2 if which == "latest" else 0)


def test_output_is_not_overwritten_and_incomplete_is_not_complete(tmp_path):
    root = tmp_path / "archive"
    root.mkdir()
    with pytest.raises(FileExistsError):
        compress(states(2), root)
    with pytest.raises(FileNotFoundError):
        Archive(root)
    with pytest.raises(MemoryError):
        compress(states(2), tmp_path / "partial", compression_memory_bytes=1)
    assert not (tmp_path / "partial/manifest.json").exists()


def test_all_bf16_rules_and_special_values():
    bits = np.array(
        [
            0,
            0x80000000,
            0x3F808000,
            0x3F818000,
            0x7F800000,
            0xFF800000,
            0x7FC01234,
            0x7F800001,
        ],
        np.uint32,
    )
    values = bits.view(np.float32)
    expected = (
        torch.from_numpy(values.copy())
        .to(torch.bfloat16)
        .view(torch.int16)
        .numpy()
        .tobytes()
    )
    assert bf16.convert(values, "rne") == expected
    assert bf16.verified_rule(values, expected) == "rne"


def test_fp64_and_non_fp32_buffers_preserve_bits(tmp_path):
    original = []
    for index in range(3):
        original.append(
            {
                "fp64": torch.linspace(-0.2, 0.5, 17, dtype=torch.float64)
                + index * 1e-9,
                "fp16": torch.tensor(
                    [0.0, -0.0, float("inf"), float("nan")], dtype=torch.float16
                ),
                "complex": torch.tensor([1 + 2j, -3 + 4j]),
                "bool": torch.tensor([True, False]),
                "u8": torch.tensor([0, 255], dtype=torch.uint8),
            }
        )
    archive = compress(original, tmp_path / "archive", chunk_elems=8)
    for index, state in enumerate(original):
        equal(state, archive.restore(index))


def test_sparse_tensor_is_rejected_instead_of_omitted(tmp_path):
    state = {"sparse": torch.eye(3).to_sparse()}
    with pytest.raises(ValueError, match="dense"):
        compress([state], tmp_path / "archive")


def test_anchor_groups_match_the_paper_eight_checkpoint_diagram():
    assert [anchor(version, 4, 8) for version in range(8)] == [3, 3, 3, 3, 7, 7, 7, 7]
