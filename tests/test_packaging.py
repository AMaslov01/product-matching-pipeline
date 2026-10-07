import json
import zipfile
from pathlib import Path

import pytest
from backbone_fixtures import write_saved_backbone

from matchcup import inference
from matchcup.packaging import (
    RUNTIME_MODULES,
    _archive_member_hashes,
    build_public_text_ablation,
    build_submission,
    mark_research_archive,
    refresh_archive_runtime,
)
from matchcup.transductive_profile import H12_V1_FEATURE_NAMES, feature_schema_digest


def test_submission_excludes_python_caches(tmp_path: Path) -> None:
    cross_encoder = tmp_path / "cross_encoder"
    fusion = tmp_path / "fusion"
    cross_encoder.mkdir()
    fusion.mkdir()
    write_saved_backbone(cross_encoder)
    (fusion / "fusion.cbm").write_bytes(b"model")
    (fusion / "feature_schema.json").write_text(
        json.dumps({"feature_names": ["transductive_identifier_df"], "categories": []}),
        encoding="utf-8",
    )
    (fusion / "fusion_model.py").write_text("# slow fallback", encoding="utf-8")
    (fusion / "fusion_report.json").write_text("{}", encoding="utf-8")
    cache = fusion / "__pycache__"
    cache.mkdir()
    (cache / "model.pyc").write_bytes(b"cache")
    output = tmp_path / "submission.zip"

    build_submission(
        cross_encoder,
        fusion,
        output,
        image="example/image:1",
        allow_ungated_build=True,
    )

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "metadata.json" in names
        assert "run.py" in names
        assert "models/fusion/fusion.cbm" in names
        assert "models/fusion/feature_schema.json" in names
        assert "matchcup/score_transform.py" in names
        assert "matchcup/transductive.py" in names
        feature_schema = json.loads(archive.read("models/fusion/feature_schema.json"))
        assert "transductive_identifier_df" in feature_schema["feature_names"]
        assert "models/fusion/fusion_model.py" not in names
        assert "models/fusion/fusion_report.json" not in names
        assert "matchcup/fusion.py" not in names
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)


def test_research_archive_marker_is_explicit_and_preserves_the_runnable_payload(
    tmp_path: Path,
) -> None:
    """A user-authorized diagnostic ZIP is runnable but never submission-eligible.

    Public seam: a caller marks an already-built archive and consumers inspect
    only the archive itself.  It must not silently drop or alter model/runtime
    members while adding the immutable failed-gate record.
    """
    cross_encoder = tmp_path / "cross_encoder"
    fusion = tmp_path / "fusion"
    cross_encoder.mkdir()
    fusion.mkdir()
    write_saved_backbone(cross_encoder)
    (fusion / "fusion.cbm").write_bytes(b"model")
    (fusion / "feature_schema.json").write_text(
        json.dumps({"feature_names": ["transductive_identifier_df"], "categories": []}),
        encoding="utf-8",
    )
    archive = tmp_path / "research.zip"
    build_submission(cross_encoder, fusion, archive, allow_ungated_build=True)
    before = _archive_member_hashes(archive)

    record = {
        "purpose": "user_authorized_gate_failed_research_archive",
        "failed_gates": {"single": {"passed": False}},
    }
    marker = mark_research_archive(archive, record)

    assert marker["submission_allowed"] is False
    assert marker["unzip_test"] == "passed"
    after = _archive_member_hashes(archive)
    assert after.keys() == {*before, "research_gate_override.json"}
    for name, digest in before.items():
        if name != "metadata.json":
            assert after[name] == digest
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist().count("metadata.json") == 1
        assert bundle.namelist().count("research_gate_override.json") == 1
        metadata = json.loads(bundle.read("metadata.json"))
        assert metadata["matchcup"]["submission_allowed"] is False
        assert metadata["matchcup"]["research_gate_override"] is True
        assert json.loads(bundle.read("research_gate_override.json")) == record


def test_submission_packages_two_models_and_explicit_runtime_contract(tmp_path: Path) -> None:
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    fusion = tmp_path / "fusion"
    control.mkdir()
    candidate.mkdir()
    fusion.mkdir()
    write_saved_backbone(control)
    write_saved_backbone(candidate)
    (fusion / "fusion.cbm").write_bytes(b"model")
    (fusion / "feature_schema.json").write_text(
        json.dumps(
            {
                "feature_names": ["text_score_category_rank"],
                "categories": ["A"],
                "score_transform": {
                    "version": 2,
                    "kind": "batch_category_percentile",
                    "source_feature": "text_score",
                    "output_feature": "text_score_category_rank",
                    "grouping": ["category"],
                    "rank_method": "average",
                    "percentile": True,
                    "score_source": "mean_probability",
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "ensemble.zip"

    result = build_submission(
        control,
        fusion,
        output,
        secondary_cross_encoder_dir=candidate,
        model_weights=(0.5, 0.5),
        cuda_precision="fp16_amp",
        allow_ungated_build=True,
    )

    assert result["model_count"] == 2
    assert result["cuda_precision"] == "fp16_amp"
    with zipfile.ZipFile(output) as archive:
        assert "models/cross_encoder/config.json" in archive.namelist()
        assert "models/cross_encoder_secondary/config.json" in archive.namelist()
        run_source = archive.read("run.py").decode("utf-8")
        assert 'root / "models" / "cross_encoder_secondary"' in run_source
        assert "model_weights=(0.5, 0.5)" in run_source
        assert "cuda_precision='fp16_amp'" in run_source
        metadata = json.loads(archive.read("metadata.json"))
        assert metadata["matchcup"]["model_count"] == 2
        assert metadata["matchcup"]["model_batch_size"] == 1024
        assert metadata["matchcup"]["score_contract"]["score_source"] == "mean_probability"


def test_profiled_fusion_requires_matching_manifest_before_packaging(tmp_path: Path) -> None:
    cross_encoder = tmp_path / "cross_encoder"
    fusion = tmp_path / "fusion"
    cross_encoder.mkdir()
    fusion.mkdir()
    write_saved_backbone(cross_encoder)
    (fusion / "fusion.cbm").write_bytes(b"model")
    names = list(H12_V1_FEATURE_NAMES)
    schema = {
        "feature_names": names,
        "categories": [],
        "transductive_profile": "h12-v1",
        "feature_schema_sha256": feature_schema_digest(names),
    }
    (fusion / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        build_submission(cross_encoder, fusion, tmp_path / "missing.zip", allow_ungated_build=True)

    (fusion / "fusion_manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "transductive_profile": "h12-v1",
                "feature_schema_sha256": feature_schema_digest(names),
            }
        ),
        encoding="utf-8",
    )
    archive = tmp_path / "profiled.zip"
    build_submission(cross_encoder, fusion, archive, allow_ungated_build=True)

    with zipfile.ZipFile(archive) as bundle:
        assert json.loads(bundle.read("models/fusion/fusion_manifest.json"))[
            "transductive_profile"
        ] == "h12-v1"


def test_submission_rejects_nonpositive_runtime_batch_size(tmp_path: Path) -> None:
    cross_encoder = tmp_path / "cross_encoder"
    fusion = tmp_path / "fusion"
    cross_encoder.mkdir()
    fusion.mkdir()
    write_saved_backbone(cross_encoder)
    (fusion / "fusion.cbm").write_bytes(b"model")
    (fusion / "feature_schema.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="model_batch_size"):
        build_submission(
            cross_encoder,
            fusion,
            tmp_path / "invalid.zip",
            allow_ungated_build=True,
            model_batch_size=0,
        )


def test_submission_requires_a_passing_novel_item_gate(tmp_path: Path) -> None:
    cross_encoder = tmp_path / "cross_encoder"
    fusion = tmp_path / "fusion"
    cross_encoder.mkdir()
    fusion.mkdir()
    write_saved_backbone(cross_encoder)
    (fusion / "fusion.cbm").write_bytes(b"model")
    (fusion / "feature_schema.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="holdout report is required"):
        build_submission(cross_encoder, fusion, tmp_path / "submission.zip")

    report = tmp_path / "holdout.json"
    report.write_text(
        '{"submission_allowed": false, "selected_score_mode": "text"}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="blocks submission"):
        build_submission(cross_encoder, fusion, tmp_path / "submission.zip", holdout_report=report)


def test_normal_builder_cannot_bypass_selected_holdout_mode(tmp_path: Path) -> None:
    cross_encoder = tmp_path / "cross_encoder"
    fusion = tmp_path / "fusion"
    cross_encoder.mkdir()
    fusion.mkdir()
    write_saved_backbone(cross_encoder)
    (fusion / "fusion.cbm").write_bytes(b"model")
    (fusion / "feature_schema.json").write_text("{}", encoding="utf-8")
    report = tmp_path / "holdout.json"
    report.write_text(
        json.dumps({"submission_allowed": True, "selected_score_mode": "hybrid"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot package"):
        build_submission(
            cross_encoder,
            fusion,
            tmp_path / "text.zip",
            score_mode="text",
            holdout_report=report,
        )


def test_public_text_ablation_changes_only_run_py_and_is_fail_closed(tmp_path: Path) -> None:
    cross_encoder = tmp_path / "cross_encoder"
    fusion = tmp_path / "fusion"
    cross_encoder.mkdir()
    fusion.mkdir()
    write_saved_backbone(cross_encoder)
    (fusion / "fusion.cbm").write_bytes(b"model")
    (fusion / "feature_schema.json").write_text("{}", encoding="utf-8")
    base = tmp_path / "hybrid.zip"
    build_submission(cross_encoder, fusion, base, allow_ungated_build=True, score_mode="hybrid")
    report = tmp_path / "holdout.json"
    report.write_text(
        json.dumps({"submission_allowed": True, "selected_score_mode": "hybrid"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="explicit acknowledgement"):
        build_public_text_ablation(
            base, tmp_path / "blocked.zip", holdout_report=report, acknowledge_public_ablation=False
        )
    result = build_public_text_ablation(
        base, tmp_path / "text.zip", holdout_report=report, acknowledge_public_ablation=True
    )
    assert result["changed_members"] == ["run.py"]
    assert set(result["base_member_sha256"]) == set(result["candidate_member_sha256"])
    for name, digest in result["base_member_sha256"].items():
        if name != "run.py":
            assert result["candidate_member_sha256"][name] == digest
    with zipfile.ZipFile(tmp_path / "text.zip") as archive:
        assert "score_mode='text'" in archive.read("run.py").decode("utf-8")

    text_report = tmp_path / "text_holdout.json"
    text_report.write_text(
        json.dumps({"submission_allowed": True, "selected_score_mode": "text"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="selecting hybrid"):
        build_public_text_ablation(
            base,
            tmp_path / "wrong-gate.zip",
            holdout_report=text_report,
            acknowledge_public_ablation=True,
        )


def _stale_copy(base: Path, destination: Path, member: str, content: bytes) -> Path:
    """Rewrite one archive member, standing in for a ZIP built by older code."""
    with zipfile.ZipFile(base) as source, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as target:
        for info in source.infolist():
            payload = content if info.filename == member else source.read(info.filename)
            target.writestr(info, payload)
    return destination


def test_runtime_refresh_replaces_only_matchcup_modules_and_is_fail_closed(
    tmp_path: Path,
) -> None:
    cross_encoder = tmp_path / "cross_encoder"
    fusion = tmp_path / "fusion"
    cross_encoder.mkdir()
    fusion.mkdir()
    write_saved_backbone(cross_encoder)
    (fusion / "fusion.cbm").write_bytes(b"model")
    (fusion / "feature_schema.json").write_text("{}", encoding="utf-8")
    fresh = tmp_path / "fresh.zip"
    build_submission(cross_encoder, fusion, fresh, allow_ungated_build=True)

    stale = _stale_copy(fresh, tmp_path / "stale.zip", "matchcup/inference.py", b"# old code\n")

    with pytest.raises(ValueError, match="explicit acknowledgement"):
        refresh_archive_runtime(
            stale, tmp_path / "blocked.zip", acknowledge_runtime_refresh=False
        )

    result = refresh_archive_runtime(
        stale, tmp_path / "refreshed.zip", acknowledge_runtime_refresh=True
    )

    assert result["changed_members"] == ["matchcup/inference.py"]
    source_root = Path(inference.__file__).resolve().parent
    with zipfile.ZipFile(tmp_path / "refreshed.zip") as archive:
        assert archive.testzip() is None
        for name in RUNTIME_MODULES:
            assert archive.read(f"matchcup/{name}") == (source_root / name).read_bytes()

    # Every member the runtime does not own must survive byte-for-byte.
    stale_hashes = _archive_member_hashes(stale)
    refreshed_hashes = _archive_member_hashes(tmp_path / "refreshed.zip")
    assert set(stale_hashes) == set(refreshed_hashes)
    for name, digest in stale_hashes.items():
        if not name.startswith("matchcup/"):
            assert refreshed_hashes[name] == digest, name

    # Refreshing an already-current archive is a no-op, not an error.
    unchanged = refresh_archive_runtime(
        fresh, tmp_path / "noop.zip", acknowledge_runtime_refresh=True
    )
    assert unchanged["changed_members"] == []

    with pytest.raises(FileExistsError):
        refresh_archive_runtime(
            stale, tmp_path / "refreshed.zip", acknowledge_runtime_refresh=True
        )

    truncated = _stale_copy(fresh, tmp_path / "truncated.zip", "run.py", b"print('x')\n")
    with zipfile.ZipFile(truncated) as source, zipfile.ZipFile(
        tmp_path / "incomplete.zip", "w"
    ) as target:
        for info in source.infolist():
            if info.filename != "matchcup/inference.py":
                target.writestr(info, source.read(info.filename))
    with pytest.raises(ValueError, match="runtime module"):
        refresh_archive_runtime(
            tmp_path / "incomplete.zip",
            tmp_path / "never.zip",
            acknowledge_runtime_refresh=True,
        )


def _minimal_fusion(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "fusion.cbm").write_bytes(b"model")
    (path / "feature_schema.json").write_text(
        json.dumps({"feature_names": ["transductive_identifier_df"], "categories": []}),
        encoding="utf-8",
    )
    return path


def test_submission_refuses_a_backbone_that_lost_its_contract(tmp_path: Path) -> None:
    """A decoder head that lost its pad token packages fine and dies on the contest GPU.

    The archive is copied wholesale, so nothing between training and the hidden
    split would notice. Failing here costs three JSON reads; failing there costs
    the archive its Success status, and with it its eligibility as a final.
    """
    cross_encoder = write_saved_backbone(tmp_path / "cross_encoder")
    fusion = _minimal_fusion(tmp_path / "fusion")
    config = json.loads((cross_encoder / "config.json").read_text(encoding="utf-8"))
    del config["pad_token_id"]
    (cross_encoder / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="pad_token_id"):
        build_submission(
            cross_encoder,
            fusion,
            tmp_path / "submission.zip",
            image="example/image:1",
            allow_ungated_build=True,
        )


def test_submission_refuses_a_precision_the_backbone_does_not_serve(tmp_path: Path) -> None:
    """OOF and serving must rank identically, and precision is what decides that."""
    cross_encoder = write_saved_backbone(tmp_path / "cross_encoder")
    fusion = _minimal_fusion(tmp_path / "fusion")

    with pytest.raises(ValueError, match="would rank differently"):
        build_submission(
            cross_encoder,
            fusion,
            tmp_path / "submission.zip",
            image="example/image:1",
            cuda_precision="bf16_weights",
            allow_ungated_build=True,
        )


def test_submission_records_every_packaged_backbone_contract(tmp_path: Path) -> None:
    cross_encoder = write_saved_backbone(tmp_path / "cross_encoder")
    secondary = write_saved_backbone(tmp_path / "secondary")
    fusion = _minimal_fusion(tmp_path / "fusion")
    (fusion / "feature_schema.json").write_text(
        json.dumps(
            {
                "feature_names": ["transductive_identifier_df"],
                "categories": [],
                "score_contract": {"kind": "raw_probability", "score_source": "mean_probability"},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "submission.zip"

    build_submission(
        cross_encoder,
        fusion,
        output,
        secondary_cross_encoder_dir=secondary,
        image="example/image:1",
        allow_ungated_build=True,
    )

    with zipfile.ZipFile(output) as archive:
        contracts = json.loads(archive.read("metadata.json"))["matchcup"]["backbone_contracts"]
    assert [entry["model_type"] for entry in contracts] == ["modernbert", "modernbert"]
    assert {entry["cuda_precision"] for entry in contracts} == {"fp16_amp"}
    assert all(entry["padding_side"] == "right" for entry in contracts)
