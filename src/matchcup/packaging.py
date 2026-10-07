from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from matchcup.backbone_contract import verify_backbone_contract
from matchcup.inference import TWO_SCORE_SOURCE, _resolve_score_source
from matchcup.transductive_profile import H12_V1, validate_fusion_manifest, validate_fusion_schema

RUNTIME_MODULES = (
    "__init__.py",
    "normalize.py",
    "parser.py",
    "serialize.py",
    "features.py",
    "transductive.py",
    "transductive_profile.py",
    "inference.py",
    "score_transform.py",
)
FUSION_RUNTIME_FILES = ("fusion.cbm", "feature_schema.json")
DEFAULT_RUNTIME_IMAGE = "bigslime19/matchcup-runtime:catboost-1.2.10"


def _archive_member_hashes(path: str | Path) -> dict[str, str]:
    """Return content hashes, rejecting ambiguous or corrupt ZIP members."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        if corrupt := archive.testzip():
            raise ValueError(f"Archive has a corrupt member: {corrupt}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Archive has duplicate member names")
        return {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in names
        }


def mark_research_archive(
    archive_path: str | Path,
    gate_record: dict[str, object],
) -> dict[str, int | str | bool]:
    """Make an existing archive explicitly research-only without touching its payload.

    This is deliberately a post-packaging operation: the regular archive build
    and all feature/schema checks remain identical, while the final ZIP carries
    an immutable record explaining why it is *not* eligible for submission.
    Rewriting instead of appending avoids duplicate ZIP members, which would
    make a metadata override ambiguous to different readers.
    """
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if not isinstance(gate_record, dict) or not gate_record:
        raise ValueError("Research archive requires a non-empty failed-gate record")
    try:
        gate_bytes = (json.dumps(gate_record, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Research archive record must be JSON-serializable") from exc

    temporary = archive_path.with_name(f".{archive_path.name}.research-tmp")
    if temporary.exists():
        raise FileExistsError(f"Research archive rewrite staging already exists: {temporary}")
    try:
        with zipfile.ZipFile(archive_path) as source:
            if corrupt := source.testzip():
                raise ValueError(f"Archive has a corrupt member: {corrupt}")
            names = source.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Archive has duplicate member names")
            if "metadata.json" not in names:
                raise ValueError("Archive is missing metadata.json")
            try:
                metadata = json.loads(source.read("metadata.json"))
            except json.JSONDecodeError as exc:
                raise ValueError("Archive metadata.json is not valid JSON") from exc
            if not isinstance(metadata, dict) or not isinstance(metadata.get("matchcup"), dict):
                raise ValueError("Archive metadata has no matchcup contract")
            metadata["matchcup"]["submission_allowed"] = False
            metadata["matchcup"]["research_gate_override"] = True
            metadata_bytes = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode(
                "utf-8"
            )
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
            ) as destination:
                for member in source.infolist():
                    if member.filename in {"metadata.json", "research_gate_override.json"}:
                        continue
                    destination.writestr(member, source.read(member.filename))
                destination.writestr("metadata.json", metadata_bytes)
                destination.writestr("research_gate_override.json", gate_bytes)
        _archive_member_hashes(temporary)
        temporary.replace(archive_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    with archive_path.open("rb") as stream:
        archive_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "archive": str(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": archive_sha256,
        "unzip_test": "passed",
        "submission_allowed": False,
    }


def _run_py(
    score_mode: str,
    *,
    cuda_precision: str = "fp16_amp",
    has_secondary_model: bool = False,
    model_weights: tuple[float, ...] | None = None,
    score_composition: str = "mean_probability",
    constant_categories: tuple[str, ...] | None = None,
    model_batch_size: int = 1024,
) -> str:
    if score_composition == TWO_SCORE_SOURCE:
        # Two-score serving hands both columns to the fusion untouched, so an
        # archive that still declared a 0.5/0.5 weight would be recording a
        # number nothing applies.
        model_weights = None
    elif model_weights is None:
        model_weights = (0.5, 0.5) if has_secondary_model else (1.0,)
    secondary = (
        'root / "models" / "cross_encoder_secondary"' if has_secondary_model else "None"
    )
    return f'''from __future__ import annotations

import argparse
import json
from pathlib import Path

from matchcup.inference import run_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", "--items-path", "-i", required=True)
    parser.add_argument("--matches_path", "--matches-path", "-m", required=True)
    parser.add_argument("--output_path", "--output-path", "-o", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    model_batch_size = int(metadata["matchcup"]["model_batch_size"])
    run_inference(
        args.items_path,
        args.matches_path,
        args.output_path,
        root / "models" / "cross_encoder",
        root / "models" / "fusion",
        secondary_model_path={secondary},
        model_weights={model_weights!r},
        score_mode={score_mode!r},
        score_composition={score_composition!r},
        cuda_precision={cuda_precision!r},
        constant_categories={constant_categories!r},
        model_batch_size=model_batch_size,
    )


if __name__ == "__main__":
    main()
'''


def _holdout_score_mode(report_path: str | Path) -> str:
    report_path = Path(report_path)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read novel-item holdout report: {report_path}") from exc
    if report.get("submission_allowed") is not True:
        raise ValueError(
            "Novel-item holdout blocks submission; model adaptation is required before packaging"
        )
    score_mode = report.get("selected_score_mode")
    if score_mode not in {"hybrid", "text"}:
        raise ValueError("Novel-item holdout report has no valid selected_score_mode")
    return score_mode


def build_submission(
    cross_encoder_dir: str | Path,
    fusion_dir: str | Path,
    output_zip: str | Path,
    *,
    secondary_cross_encoder_dir: str | Path | None = None,
    model_weights: tuple[float, ...] | None = None,
    score_composition: str = "mean_probability",
    cuda_precision: str = "fp16_amp",
    image: str = DEFAULT_RUNTIME_IMAGE,
    score_mode: str | None = None,
    holdout_report: str | Path | None = None,
    allow_ungated_build: bool = False,
    constant_categories: tuple[str, ...] | None = None,
    model_batch_size: int = 1024,
) -> dict[str, int | str]:
    cross_encoder_dir = Path(cross_encoder_dir)
    secondary_cross_encoder = (
        Path(secondary_cross_encoder_dir) if secondary_cross_encoder_dir is not None else None
    )
    fusion_dir = Path(fusion_dir)
    output_zip = Path(output_zip)
    # The model directories are copied in wholesale, so whatever the training run
    # saved is what serves. A decoder backbone that lost its pad token, its
    # explicit padding side or its pair separator on the way here still packages
    # cleanly and then fails on the contest GPU, where the only signal is a failed
    # phase. Reading three JSON files closes that gap before anything is staged.
    packaged_contracts = [verify_backbone_contract(cross_encoder_dir)]
    if secondary_cross_encoder is not None:
        packaged_contracts.append(verify_backbone_contract(secondary_cross_encoder))
    for contract in packaged_contracts:
        if contract["cuda_precision"] != cuda_precision:
            raise ValueError(
                "Packaged backbone expects "
                f"{contract['cuda_precision']!r} but the archive declares {cuda_precision!r}; "
                "OOF and serving would rank differently"
            )
    if holdout_report is None and not allow_ungated_build:
        raise ValueError(
            "A successful novel-item holdout report is required; "
            "use allow_ungated_build only for local runtime smoke tests"
        )
    selected_mode = _holdout_score_mode(holdout_report) if holdout_report else None
    if score_mode is None:
        score_mode = selected_mode or "hybrid"
    if score_mode not in {"hybrid", "text"}:
        raise ValueError(f"Unsupported score mode: {score_mode!r}")
    if cuda_precision not in {"fp16_amp", "fp16_weights", "bf16_weights"}:
        raise ValueError(f"Unsupported CUDA precision: {cuda_precision!r}")
    if model_batch_size < 1:
        raise ValueError("model_batch_size must be positive")
    if selected_mode is not None and score_mode != selected_mode:
        raise ValueError(
            f"Holdout selected score mode {selected_mode!r}, cannot package {score_mode!r}"
        )
    missing_fusion_files = [
        name for name in FUSION_RUNTIME_FILES if not (fusion_dir / name).is_file()
    ]
    if missing_fusion_files:
        raise FileNotFoundError(f"Missing fusion runtime files: {missing_fusion_files}")
    model_count = 2 if secondary_cross_encoder is not None else 1
    # The runtime owns this gate, so a shape it would refuse cannot be packaged.
    expected_source = _resolve_score_source(
        score_composition, model_count, score_mode, model_weights
    )
    if score_composition != TWO_SCORE_SOURCE:
        if model_weights is None:
            model_weights = tuple(1.0 / model_count for _ in range(model_count))
        if len(model_weights) != model_count or abs(sum(model_weights) - 1.0) > 1e-12:
            raise ValueError("Model weights must match model count and sum to one")
    feature_schema = json.loads(
        (fusion_dir / "feature_schema.json").read_text(encoding="utf-8")
    )
    profile = (
        validate_fusion_schema(feature_schema)
        if "feature_names" in feature_schema
        else H12_V1
    )
    fusion_manifest_path = fusion_dir / "fusion_manifest.json"
    if feature_schema.get("transductive_profile") is not None:
        if not fusion_manifest_path.is_file():
            raise ValueError(
                "Profiled fusion requires a matching fusion_manifest.json before packaging"
            )
        try:
            fusion_manifest = json.loads(fusion_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Could not read fusion_manifest.json") from exc
        validate_fusion_manifest(fusion_manifest, feature_schema)
    score_contract = feature_schema.get("score_contract") or feature_schema.get(
        "score_transform"
    )
    if score_contract is not None and score_contract.get("score_source") != expected_source:
        raise ValueError(
            "Fusion score source does not match packaged models: "
            f"expected {expected_source!r}"
        )
    staging = output_zip.parent / f".{output_zip.stem}_staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "matchcup").mkdir(parents=True)
    (staging / "models").mkdir(parents=True)
    source_root = Path(__file__).resolve().parent
    for name in RUNTIME_MODULES:
        shutil.copy2(source_root / name, staging / "matchcup" / name)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
    shutil.copytree(cross_encoder_dir, staging / "models" / "cross_encoder", ignore=ignore)
    if secondary_cross_encoder is not None:
        shutil.copytree(
            secondary_cross_encoder,
            staging / "models" / "cross_encoder_secondary",
            ignore=ignore,
        )
    runtime_fusion_dir = staging / "models" / "fusion"
    runtime_fusion_dir.mkdir()
    for name in FUSION_RUNTIME_FILES:
        shutil.copy2(fusion_dir / name, runtime_fusion_dir / name)
    if fusion_manifest_path.is_file():
        shutil.copy2(fusion_manifest_path, runtime_fusion_dir / fusion_manifest_path.name)
    (staging / "run.py").write_text(
        _run_py(
            score_mode,
            cuda_precision=cuda_precision,
            has_secondary_model=secondary_cross_encoder is not None,
            model_weights=model_weights,
            score_composition=score_composition,
            constant_categories=constant_categories,
            model_batch_size=model_batch_size,
        ),
        encoding="utf-8",
    )
    (staging / "metadata.json").write_text(
        json.dumps(
            {
                "image": image,
                "entry_point": "python -u run.py",
                "matchcup": {
                    "model_count": model_count,
                    "model_weights": None if model_weights is None else list(model_weights),
                    "score_composition": score_composition,
                    "cuda_precision": cuda_precision,
                    "model_batch_size": model_batch_size,
                    "score_contract": score_contract,
                    "transductive_profile": profile.name,
                    "feature_schema_sha256": feature_schema.get("feature_schema_sha256"),
                    "constant_categories": list(constant_categories or ()),
                    "backbone_contracts": packaged_contracts,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in staging.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(staging))
    size = output_zip.stat().st_size
    shutil.rmtree(staging)
    if size > 5_000_000_000:
        raise ValueError(f"Submission archive is too large: {size} bytes")
    return {
        "archive": str(output_zip),
        "bytes": size,
        "score_mode": score_mode,
        "score_composition": score_composition,
        "model_count": model_count,
        "cuda_precision": cuda_precision,
        "constant_categories": len(constant_categories or ()),
    }


def build_public_text_ablation(
    base_archive: str | Path,
    output_zip: str | Path,
    *,
    holdout_report: str | Path,
    acknowledge_public_ablation: bool,
) -> dict[str, object]:
    """Make the single diagnostic text-only archive from an immutable hybrid ZIP.

    This deliberately does *not* call :func:`build_submission`: ordinary
    packaging remains bound to the selected holdout mode.  The narrow escape
    hatch is only for the pre-registered Public attribution experiment, and
    proves that every member except ``run.py`` has unchanged uncompressed
    content.
    """
    if not acknowledge_public_ablation:
        raise ValueError("public text ablation requires explicit acknowledgement")
    if _holdout_score_mode(holdout_report) != "hybrid":
        raise ValueError("public text ablation requires a holdout report selecting hybrid")

    base_archive = Path(base_archive)
    output_zip = Path(output_zip)
    if not base_archive.is_file():
        raise FileNotFoundError(base_archive)
    if output_zip.exists():
        raise FileExistsError(f"Refusing to overwrite public ablation archive: {output_zip}")

    base_hashes = _archive_member_hashes(base_archive)
    if "run.py" not in base_hashes:
        raise ValueError("Base archive has no run.py")
    with zipfile.ZipFile(base_archive) as source:
        base_run = source.read("run.py").decode("utf-8")
        if base_run != _run_py("hybrid"):
            raise ValueError("Base archive is not the expected hybrid run.py")
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as target:
            for info in source.infolist():
                content = (
                    _run_py("text").encode("utf-8")
                    if info.filename == "run.py"
                    else source.read(info.filename)
                )
                target.writestr(info, content)

    candidate_hashes = _archive_member_hashes(output_zip)
    changed = sorted(
        name
        for name in base_hashes
        if base_hashes.get(name) != candidate_hashes.get(name)
    )
    if set(base_hashes) != set(candidate_hashes) or changed != ["run.py"]:
        raise AssertionError(
            "Public ablation is invalid: only run.py may differ from the immutable base archive"
        )
    return {
        "purpose": "public_text_only_attribution_ablation",
        "archive": str(output_zip),
        "bytes": output_zip.stat().st_size,
        "base_archive": str(base_archive),
        "score_mode": "text",
        "changed_members": changed,
        "base_member_sha256": base_hashes,
        "candidate_member_sha256": candidate_hashes,
    }


def refresh_archive_runtime(
    base_archive: str | Path,
    output_zip: str | Path,
    *,
    acknowledge_runtime_refresh: bool,
) -> dict[str, object]:
    """Rebuild a gated archive against the current runtime modules.

    A packaged archive carries its own copy of ``matchcup/*.py``.  When a
    runtime fix lands after packaging, the models, fusion and score contract
    inside the ZIP are still the gated ones, but the Python that runs them is
    stale.  Re-running the whole gated pipeline to pick up a runtime-only fix
    would retrain a fusion that nothing asked to change.

    This replaces exactly the runtime modules and proves every other member
    survived byte-for-byte, so the scoring contract cannot drift through the
    back door.  It deliberately does *not* call :func:`build_submission`:
    ordinary packaging stays bound to its holdout gate.
    """
    if not acknowledge_runtime_refresh:
        raise ValueError("runtime refresh requires explicit acknowledgement")

    base_archive = Path(base_archive)
    output_zip = Path(output_zip)
    if not base_archive.is_file():
        raise FileNotFoundError(base_archive)
    if output_zip.exists():
        raise FileExistsError(f"Refusing to overwrite refreshed archive: {output_zip}")

    base_hashes = _archive_member_hashes(base_archive)
    expected = {f"matchcup/{name}" for name in RUNTIME_MODULES}
    if missing := sorted(expected - set(base_hashes)):
        raise ValueError(f"Base archive is missing a runtime module: {missing}")

    source_root = Path(__file__).resolve().parent
    replacements = {
        f"matchcup/{name}": (source_root / name).read_bytes() for name in RUNTIME_MODULES
    }

    with zipfile.ZipFile(base_archive) as source:
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as target:
            for info in source.infolist():
                content = replacements.get(info.filename)
                if content is None:
                    content = source.read(info.filename)
                target.writestr(info, content)

    candidate_hashes = _archive_member_hashes(output_zip)
    if set(base_hashes) != set(candidate_hashes):
        raise AssertionError("Runtime refresh added or dropped an archive member")
    changed = sorted(
        name for name in base_hashes if base_hashes[name] != candidate_hashes[name]
    )
    if unexpected := [name for name in changed if name not in expected]:
        raise AssertionError(
            f"Runtime refresh must not change non-runtime members: {unexpected}"
        )
    return {
        "purpose": "runtime_module_refresh",
        "archive": str(output_zip),
        "bytes": output_zip.stat().st_size,
        "base_archive": str(base_archive),
        "changed_members": changed,
        "base_member_sha256": base_hashes,
        "candidate_member_sha256": candidate_hashes,
    }


def _probe_run_py(constant: float) -> str:
    """Entry point for a leaderboard probe that never loads a model.

    Average precision over a constant score equals the positive rate, so the
    macro AP reported for this archive is the mean per-category prevalence of the
    hidden split. That single number tells us whether our Gold-versus-Public gap
    is an operating-point difference or genuine model degradation.
    """
    return f'''from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


CONSTANT = {constant!r}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", "--items-path", "-i", required=True)
    parser.add_argument("--matches_path", "--matches-path", "-m", required=True)
    parser.add_argument("--output_path", "--output-path", "-o", required=True)
    args = parser.parse_args()
    matches = pq.read_table(args.matches_path, columns=["id1", "id2"])
    first = matches.column("id1").to_pylist()
    second = matches.column("id2").to_pylist()
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        handle.write("id1,id2,predict\\n")
        for left, right in zip(first, second):
            handle.write(f"{{left}},{{right}},{{CONSTANT}}\\n")
    print(f"wrote {{len(first)}} constant predictions to {{output}}")


if __name__ == "__main__":
    main()
'''


def build_constant_probe_submission(
    output_zip: str | Path,
    *,
    constant: float = 0.5,
    image: str = DEFAULT_RUNTIME_IMAGE,
) -> dict[str, int | str | float]:
    """Package a model-free archive that predicts one constant for every pair.

    Intentionally bypasses the novel-holdout gate that guards real submissions:
    this archive carries no model, so there is nothing for that gate to assess.
    It measures the evaluation split, not our system.
    """
    if not 0.0 < constant < 1.0:
        raise ValueError("constant must lie in (0, 1) so the output stays a valid score")
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(
        {
            "image": image,
            "entry_point": "python -u run.py",
            "matchcup": {
                "kind": "constant_probe",
                "constant": constant,
                "model_count": 0,
                "purpose": "measure mean per-category prevalence of the hidden split",
            },
        },
        indent=2,
    )
    with zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.writestr("run.py", _probe_run_py(constant))
        archive.writestr("metadata.json", metadata)
    return {
        "archive": str(output_zip),
        "bytes": output_zip.stat().st_size,
        "kind": "constant_probe",
        "constant": constant,
    }
