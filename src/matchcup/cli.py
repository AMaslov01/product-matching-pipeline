from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from matchcup.config import Config, load_config


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _config(args: argparse.Namespace) -> Config:
    return load_config(args.config)


def _ids_from_pairs(path: str | Path) -> set[int]:
    table = pq.read_table(path, columns=["id1", "id2"])
    return {int(value) for column in ("id1", "id2") for value in table[column].to_numpy()}


def doctor(args: argparse.Namespace) -> None:
    cfg = _config(args)
    expected = {
        "items": cfg.paths.raw_dir / "items.parquet",
        "items_human": cfg.paths.raw_dir / "items_human.parquet",
        "matches": cfg.paths.raw_dir / "matches.parquet",
        "matches_llm": cfg.paths.raw_dir / "matches_llm.parquet",
    }
    files = {}
    for name, path in expected.items():
        files[name] = {
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
            "rows": pq.ParquetFile(path).metadata.num_rows if path.exists() else None,
        }
    cfg.paths.work_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(cfg.paths.work_dir)
    _print({"files": files, "work_dir": str(cfg.paths.work_dir), "free_bytes": usage.free})
    if not all(record["exists"] for record in files.values()):
        raise SystemExit(2)


def canonicalize(args: argparse.Namespace) -> None:
    from matchcup.parser import canonicalize_parquet

    cfg = _config(args)
    section = cfg.section("preprocess")
    if args.dataset == "human":
        source = cfg.paths.raw_dir / "items_human.parquet"
        destination = cfg.paths.canonical_dir / "items_human.parquet"
    else:
        source = cfg.paths.raw_dir / "items.parquet"
        destination = cfg.paths.canonical_dir / "items_silver.parquet"
    accepted_ids = _ids_from_pairs(args.ids_from) if args.ids_from else None
    _print(
        canonicalize_parquet(
            source,
            destination,
            batch_size=args.batch_size or section.get("batch_size", 4096),
            workers=args.workers if args.workers is not None else section.get("workers", 1),
            remaining_chars=section.get("remaining_attributes_chars", 1800),
            compression=section.get("parquet_compression", "zstd"),
            accepted_ids=accepted_ids,
        )
    )


def make_folds(args: argparse.Namespace) -> None:
    from matchcup.folds import make_component_folds, validate_component_folds

    cfg = _config(args)
    output = cfg.paths.pairs_dir / "matches_gold_folds.parquet"
    result = make_component_folds(
        cfg.paths.raw_dir / "matches.parquet",
        output,
        n_folds=cfg.get("folds.count", 5),
        seed=cfg.get("folds.seed", 20260812),
        items_path=cfg.paths.raw_dir / "items_human.parquet",
    )
    validate_component_folds(output)
    _print(result)


def sample_silver_command(args: argparse.Namespace) -> None:
    from matchcup.sampling import sample_silver

    cfg = _config(args)
    _print(
        sample_silver(
            cfg.paths.raw_dir / "matches_llm.parquet",
            cfg.paths.pairs_dir / "matches_silver_sample.parquet",
            max_rows=args.max_rows or cfg.get("silver.max_rows", 2_000_000),
            seed=cfg.get("silver.seed", 20260812),
        )
    )


def make_novel_holdout_command(args: argparse.Namespace) -> None:
    from matchcup.holdout import make_novel_item_holdout

    cfg = _config(args)
    _print(
        make_novel_item_holdout(
            args.items or cfg.paths.raw_dir / "items.parquet",
            args.matches_llm or cfg.paths.raw_dir / "matches_llm.parquet",
            args.gold or cfg.paths.raw_dir / "matches.parquet",
            args.silver or cfg.paths.pairs_dir / "matches_silver_sample.parquet",
            args.output or cfg.paths.work_dir / "holdout" / "novel_items.parquet",
            max_rows=args.max_rows,
            seed=args.seed,
            temp_dir=args.temp_dir or cfg.paths.work_dir / "tmp",
        )
    )


def make_public_proxy_command(args: argparse.Namespace) -> None:
    from matchcup.holdout import make_natural_rate_public_proxy

    cfg = _config(args)
    _print(
        make_natural_rate_public_proxy(
            args.items or cfg.paths.raw_dir / "items.parquet",
            args.matches_llm or cfg.paths.raw_dir / "matches_llm.parquet",
            args.gold or cfg.paths.raw_dir / "matches.parquet",
            args.silver or cfg.paths.pairs_dir / "matches_silver_sample.parquet",
            args.output or cfg.paths.work_dir / "research" / "public-proxy" / "pairs.parquet",
            max_rows_per_category=args.max_rows_per_category,
            seed=args.seed,
            temp_dir=args.temp_dir or cfg.paths.work_dir / "tmp" / "public-proxy",
            report_path=args.report,
        )
    )


def research_llm_diagnostics_command(args: argparse.Namespace) -> None:
    from matchcup.holdout import describe_new_item_llm_slices

    cfg = _config(args)
    _print(
        describe_new_item_llm_slices(
            args.items or cfg.paths.raw_dir / "items.parquet",
            args.matches_llm or cfg.paths.raw_dir / "matches_llm.parquet",
            args.gold or cfg.paths.raw_dir / "matches.parquet",
            args.silver or cfg.paths.pairs_dir / "matches_silver_sample.parquet",
            args.output or cfg.paths.work_dir / "research" / "new_item_llm_diagnostic.json",
            temp_dir=args.temp_dir or cfg.paths.work_dir / "tmp",
        )
    )


def research_proxy_shift_command(args: argparse.Namespace) -> None:
    from matchcup.proxy_shift import diagnose_proxy_domain_shift

    cfg = _config(args)
    _print(
        diagnose_proxy_domain_shift(
            args.human_items or cfg.paths.raw_dir / "items_human.parquet",
            args.human_matches or cfg.paths.raw_dir / "matches.parquet",
            args.llm_items or cfg.paths.raw_dir / "items.parquet",
            args.matches_llm or cfg.paths.raw_dir / "matches_llm.parquet",
            args.silver or cfg.paths.pairs_dir / "matches_silver_sample.parquet",
            args.output
            or cfg.paths.work_dir / "research" / "public-diagnosis-20260814" / "proxy_shift.json",
            max_pairs_per_source_category=args.max_pairs_per_source_category,
            min_pairs_per_source_category=args.min_pairs_per_source_category,
            seed=args.seed,
            temp_dir=args.temp_dir or cfg.paths.work_dir / "tmp" / "proxy_shift",
        )
    )


def prepare_novel_holdout_command(args: argparse.Namespace) -> None:
    from matchcup.pairs import prepare_pairs
    from matchcup.parser import canonicalize_parquet

    cfg = _config(args)
    section = cfg.section("preprocess")
    feature_cfg = cfg.section("features")
    matches = Path(args.holdout or cfg.paths.work_dir / "holdout" / "novel_items.parquet")
    canonical = Path(
        args.canonical_output or cfg.paths.canonical_dir / "novel_holdout_items.parquet"
    )
    output = Path(args.output or cfg.paths.pairs_dir / "novel_holdout.parquet")
    canonical_result = canonicalize_parquet(
        args.items or cfg.paths.raw_dir / "items.parquet",
        canonical,
        batch_size=section.get("batch_size", 4096),
        workers=args.workers if args.workers is not None else section.get("workers", 1),
        remaining_chars=section.get("remaining_attributes_chars", 1800),
        compression=section.get("parquet_compression", "zstd"),
        accepted_ids=_ids_from_pairs(matches),
    )
    pair_result = prepare_pairs(
        canonical,
        matches,
        output,
        chunk_size=args.chunk_size,
        ngram_size=feature_cfg.get("char_ngram_size", 3),
        measurement_tolerance=feature_cfg.get("measurement_tolerance", 0.015),
        compression=section.get("parquet_compression", "zstd"),
        temp_dir=cfg.paths.work_dir / "tmp",
    )
    _print({"canonicalize": canonical_result, "prepare_pairs": pair_result})


def evaluate_novel_holdout_command(args: argparse.Namespace) -> None:
    from matchcup.holdout import evaluate_novel_item_holdout

    cfg = _config(args)
    _print(
        evaluate_novel_item_holdout(
            args.pairs or cfg.paths.pairs_dir / "novel_holdout.parquet",
            args.scores or cfg.paths.work_dir / "holdout" / "novel_text_scores.parquet",
            args.fusion or cfg.paths.models_dir / "fusion",
            args.output or cfg.paths.work_dir / "holdout" / "novel_holdout_report.json",
            min_hybrid_gain=args.min_hybrid_gain,
            public_macro_ap=args.public_macro_ap,
            near_public_margin=args.near_public_margin,
        )
    )


def prepare_pairs_command(args: argparse.Namespace) -> None:
    from matchcup.pairs import prepare_pairs

    cfg = _config(args)
    feature_cfg = cfg.section("features")
    preprocess_cfg = cfg.section("preprocess")
    if args.dataset == "gold":
        items = cfg.paths.canonical_dir / "items_human.parquet"
        matches = cfg.paths.pairs_dir / "matches_gold_folds.parquet"
        output = cfg.paths.pairs_dir / "gold.parquet"
    else:
        items = cfg.paths.canonical_dir / "items_silver.parquet"
        matches = cfg.paths.pairs_dir / "matches_silver_sample.parquet"
        output = cfg.paths.pairs_dir / "silver.parquet"
    _print(
        prepare_pairs(
            items,
            matches,
            output,
            chunk_size=args.chunk_size,
            include_features=not args.text_only,
            ngram_size=feature_cfg.get("char_ngram_size", 3),
            measurement_tolerance=feature_cfg.get("measurement_tolerance", 0.015),
            compression=preprocess_cfg.get("parquet_compression", "zstd"),
            temp_dir=cfg.paths.work_dir / "tmp",
            transductive_items_path=args.transductive_items,
            transductive_profile=args.transductive_profile,
        )
    )


def _xe_config(cfg: Config, stage: str, model_name: str) -> Any:
    from matchcup.cross_encoder import CrossEncoderConfig

    section = cfg.section("cross_encoder")
    gold = stage != "silver"
    return CrossEncoderConfig(
        model_name=model_name,
        model_revision=section.get("model_revision"),
        max_length=section.get("max_length", 384),
        train_batch_size=section.get("train_batch_size", 16),
        train_categories_per_batch=section.get("train_categories_per_batch", 1),
        eval_batch_size=section.get("eval_batch_size", 64),
        gradient_accumulation_steps=section.get("gradient_accumulation_steps", 4),
        learning_rate=section.get("gold_learning_rate" if gold else "silver_learning_rate"),
        weight_decay=section.get("weight_decay", 0.01),
        warmup_ratio=section.get("warmup_ratio", 0.06),
        epochs=section.get("gold_epochs" if gold else "silver_epochs"),
        ranking_weight=section.get("ranking_weight", 0.15) if gold else 0.0,
        ranking_from_epoch=section.get("ranking_from_epoch", 1),
        positive_prevalence=section.get("positive_prevalence", 0.5),
        mixed_precision=section.get("mixed_precision", "fp16"),
        num_workers=section.get("num_workers", 4),
        seed=section.get("seed", 20260812),
    )


def train_cross_encoder_command(args: argparse.Namespace) -> None:
    from matchcup.cross_encoder import train_cross_encoder
    from matchcup.schema_robustness import load_attribute_permutation_spec

    cfg = _config(args)
    if args.stage == "silver":
        input_path = Path(args.input or cfg.paths.pairs_dir / "silver.parquet")
        output = Path(args.output or cfg.paths.models_dir / "silver")
        model_name = args.base_model or cfg.get("cross_encoder.model_name")
        fold = None
        hard = None
    elif args.stage == "fold":
        if args.fold is None:
            raise ValueError("--fold is required for stage=fold")
        input_path = Path(args.input or cfg.paths.pairs_dir / "gold.parquet")
        output = Path(args.output or cfg.paths.models_dir / f"gold_fold_{args.fold}")
        model_name = args.base_model or str(cfg.paths.models_dir / "silver")
        fold = args.fold
        hard = None
    else:
        input_path = Path(args.input or cfg.paths.pairs_dir / "gold.parquet")
        output = Path(args.output or cfg.paths.models_dir / "final")
        model_name = args.base_model or str(cfg.paths.models_dir / "silver")
        fold = None
        default_hard = cfg.paths.pairs_dir / "hard_negatives.parquet"
        hard = (
            Path(args.hard_negatives)
            if args.hard_negatives
            else (default_hard if default_hard.exists() else None)
        )
    config = _xe_config(cfg, args.stage, model_name)
    # A fold has a validation loader, so the generic completion path does not
    # save it.  Gold OOF scoring nevertheless needs the model after its single
    # scheduled epoch (the normal cloud recipe deliberately stops at its exact
    # step budget before the validation pass), so make that final save explicit.
    if args.stage == "fold":
        config = replace(config, save_final_checkpoint=True)
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        config = replace(config, epochs=args.epochs)
    positive_prevalence = getattr(args, "positive_prevalence", None)
    if positive_prevalence is not None:
        config = replace(config, positive_prevalence=positive_prevalence)
    ranking_from_epoch = getattr(args, "ranking_from_epoch", None)
    if ranking_from_epoch is not None:
        if ranking_from_epoch < 0:
            raise ValueError("--ranking-from-epoch must be non-negative")
        config = replace(config, ranking_from_epoch=ranking_from_epoch)
    if getattr(args, "no_save_model", False):
        config = replace(config, save_model_on_completion=False)
    if args.disable_hard_negatives:
        if args.stage != "final":
            raise ValueError("--disable-hard-negatives is allowed only with --stage final")
        hard = None
    if args.schema_robustness_spec:
        if args.stage != "final":
            raise ValueError(
                "schema-robustness transform is allowed only for the explicit final refit"
            )
        spec_path = Path(args.schema_robustness_spec).expanduser().resolve()
        spec = load_attribute_permutation_spec(spec_path)
        config = replace(
            config,
            train_text_transform=spec,
            allow_full_training_transform=True,
            train_text_transform_provenance={
                "spec": spec.normalized(),
                "spec_sha256": spec.fingerprint(),
                "source_yaml_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            },
        )
    checkpoint_dir = None
    checkpoint_sync = None
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        if args.checkpoint_gcs:
            from matchcup.checkpoint_gcs import make_sync, pull_latest

            # Recover any checkpoint a preempted earlier instance already uploaded,
            # then keep mirroring new ones so the next preemption is survivable too.
            pull_latest(args.checkpoint_gcs, checkpoint_dir)
            checkpoint_sync = make_sync(args.checkpoint_gcs)
    _print(
        train_cross_encoder(
            input_path,
            output,
            config,
            validation_fold=fold,
            hard_negative_path=hard,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval_seconds=args.checkpoint_interval_seconds,
            checkpoint_sync=checkpoint_sync,
            max_optimizer_steps=getattr(args, "max_optimizer_steps", None),
            batch_fixture_path=getattr(args, "batch_fixture", None),
            fixture_groups_per_batch=getattr(args, "fixture_groups_per_batch", 1),
        )
    )


def score_cross_encoder_command(args: argparse.Namespace) -> None:
    from matchcup.cross_encoder import score_cross_encoder

    cfg = _config(args)
    fold = args.fold
    model = Path(
        args.model
        or (
            cfg.paths.models_dir / f"gold_fold_{fold}"
            if fold is not None
            else cfg.paths.models_dir / "final"
        )
    )
    output = Path(args.output or (cfg.paths.work_dir / "oof" / f"fold_{fold}.parquet"))
    _print(
        score_cross_encoder(
            args.input or cfg.paths.pairs_dir / "gold.parquet",
            model,
            output,
            max_length=args.max_length or cfg.get("cross_encoder.max_length", 384),
            batch_size=args.batch_size or cfg.get("cross_encoder.eval_batch_size", 64),
            mixed_precision=cfg.get("cross_encoder.mixed_precision", "fp16"),
            fold=fold,
            num_workers=cfg.get("cross_encoder.num_workers", 4),
        )
    )


def mine_hard_negatives_command(args: argparse.Namespace) -> None:
    from matchcup.hard_negatives import mine_hard_negatives

    cfg = _config(args)
    _print(
        mine_hard_negatives(
            args.gold or cfg.paths.pairs_dir / "gold.parquet",
            args.oof or cfg.paths.work_dir / "oof",
            args.output or cfg.paths.pairs_dir / "hard_negatives.parquet",
            top_fraction=args.top_fraction,
        )
    )


def materialize_augmentation_command(args: argparse.Namespace) -> None:
    """Build one explicit, train-only Gold augmentation family.

    The command deliberately materializes exactly one family per output.  That
    makes an A+B synthetic table impossible to create accidentally and leaves
    the returned JSON suitable for the immutable run manifest.
    """
    from matchcup.augmentation import (
        materialize_attribute_dropout,
        materialize_hard_identifier_negatives,
    )

    cfg = _config(args)
    source = Path(args.input or cfg.paths.pairs_dir / "gold.parquet")
    output = Path(args.output)
    seed = args.seed if args.seed is not None else cfg.get("cross_encoder.seed", 20260812)
    compression = cfg.get("preprocess.parquet_compression", "zstd")
    if args.kind == "attribute-dropout":
        report = materialize_attribute_dropout(
            source,
            output,
            seed=seed,
            compression=compression,
        )
    else:
        catalogue = Path(args.catalogue_items or cfg.paths.canonical_dir / "items_human.parquet")
        report = materialize_hard_identifier_negatives(
            source,
            catalogue,
            output,
            seed=seed,
            min_identifier_idf=args.min_identifier_idf,
            min_lcp_fraction=args.min_lcp_fraction,
            max_rows=args.max_rows,
            compression=compression,
        )
    _print(report)


def research_run_variant_command(args: argparse.Namespace) -> None:
    from matchcup.research import run_fold_ablation

    cfg = _config(args)
    section = cfg.section("cross_encoder")
    _print(
        run_fold_ablation(
            args.gold or cfg.paths.pairs_dir / "gold.parquet",
            args.output,
            variant=args.variant,
            fold=args.fold,
            base_model=args.base_model or cfg.get("cross_encoder.model_name"),
            silver_model=args.silver_model,
            mining_model=args.mining_model,
            model_revision=section.get("model_revision"),
            max_length=section.get("max_length", 384),
            train_batch_size=section.get("train_batch_size", 16),
            eval_batch_size=section.get("eval_batch_size", 64),
            gradient_accumulation_steps=section.get("gradient_accumulation_steps", 4),
            learning_rate=section.get("gold_learning_rate"),
            weight_decay=section.get("weight_decay", 0.01),
            warmup_ratio=section.get("warmup_ratio", 0.06),
            ranking_weight=section.get("ranking_weight", 0.15),
            mixed_precision=section.get("mixed_precision", "fp16"),
            num_workers=section.get("num_workers", 4),
            seed=section.get("seed", 20260812),
            hard_negative_fraction=args.hard_negative_fraction,
            schema_robustness_spec=args.schema_robustness_spec,
        )
    )


def research_compare_command(args: argparse.Namespace) -> None:
    from matchcup.research import paired_bootstrap_difference

    cfg = _config(args)
    result = paired_bootstrap_difference(
        args.gold or cfg.paths.pairs_dir / "gold.parquet",
        args.control,
        args.candidate,
        fold=args.fold,
        samples=args.samples,
        seed=args.seed,
    )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print(result)


def research_assess_candidate_command(args: argparse.Namespace) -> None:
    from matchcup.research import assess_candidate_gate

    cfg = _config(args)
    result = assess_candidate_gate(
        args.gold or cfg.paths.pairs_dir / "gold.parquet",
        args.control,
        args.candidate,
        fold=args.fold,
        minimum_macro_gain=args.minimum_macro_gain,
        maximum_category_drop=args.maximum_category_drop,
        samples=args.samples,
        seed=args.seed,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print(result)


def research_diagnose_scores_command(args: argparse.Namespace) -> None:
    from matchcup.research import compare_score_distributions

    cfg = _config(args)
    result = compare_score_distributions(
        args.gold or cfg.paths.pairs_dir / "gold.parquet",
        args.oof,
        args.final,
        top_fraction=args.top_fraction,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print(result)


def research_diagnose_fusion_shift_command(args: argparse.Namespace) -> None:
    from matchcup.research import describe_fusion_score_shift

    cfg = _config(args)
    result = describe_fusion_score_shift(
        args.gold or cfg.paths.pairs_dir / "gold.parquet",
        args.oof,
        args.final,
        args.fusion,
        top_fraction=args.top_fraction,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print(result)


def research_graph_diagnostics_command(args: argparse.Namespace) -> None:
    from matchcup.research import graph_diagnostics

    cfg = _config(args)
    result = graph_diagnostics(args.gold or cfg.paths.pairs_dir / "gold.parquet", args.scores)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print(result)


def train_fusion_command(args: argparse.Namespace) -> None:
    from matchcup.fusion import train_fusion

    cfg = _config(args)
    section = cfg.section("fusion")
    _print(
        train_fusion(
            cfg.paths.pairs_dir / "gold.parquet",
            args.oof or cfg.paths.work_dir / "oof",
            args.output or cfg.paths.models_dir / "fusion",
            iterations=section.get("iterations", 1200),
            learning_rate=section.get("learning_rate", 0.04),
            depth_candidates=section.get("depth_candidates"),
            l2_candidates=section.get("l2_leaf_reg_candidates"),
            seed=section.get("random_seed", 20260812),
            score_source=args.score_source,
            score_contract=args.score_contract,
            transductive_profile=args.transductive_profile,
            expected_feature_schema_sha256=args.expected_feature_schema_sha256,
        )
    )


def build_submission_command(args: argparse.Namespace) -> None:
    from matchcup.packaging import DEFAULT_RUNTIME_IMAGE, build_submission

    cfg = _config(args)
    _print(
        build_submission(
            args.cross_encoder or cfg.paths.models_dir / "final",
            args.fusion or cfg.paths.models_dir / "fusion",
            args.output or cfg.paths.outputs_dir / "matchcup-submission.zip",
            image=args.image or DEFAULT_RUNTIME_IMAGE,
            score_mode=args.score_mode,
            holdout_report=args.holdout_report,
            allow_ungated_build=args.allow_ungated_build,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="matchcup")
    parser.add_argument("--config", default="configs/default.yaml")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("doctor")
    command.set_defaults(func=doctor)

    command = commands.add_parser("canonicalize")
    command.add_argument("--dataset", choices=["human", "full"], required=True)
    command.add_argument("--ids-from")
    command.add_argument("--batch-size", type=int)
    command.add_argument("--workers", type=int)
    command.set_defaults(func=canonicalize)

    command = commands.add_parser("make-folds")
    command.set_defaults(func=make_folds)

    command = commands.add_parser("sample-silver")
    command.add_argument("--max-rows", type=int)
    command.set_defaults(func=sample_silver_command)

    command = commands.add_parser("make-novel-holdout")
    command.add_argument("--items")
    command.add_argument("--matches-llm")
    command.add_argument("--gold")
    command.add_argument("--silver")
    command.add_argument("--output")
    command.add_argument("--max-rows", type=int, default=100_000)
    command.add_argument("--seed", type=int, default=20260812)
    command.add_argument("--temp-dir")
    command.set_defaults(func=make_novel_holdout_command)

    command = commands.add_parser("make-public-proxy")
    command.add_argument("--items")
    command.add_argument("--matches-llm")
    command.add_argument("--gold")
    command.add_argument("--silver")
    command.add_argument("--output")
    command.add_argument("--report")
    command.add_argument("--max-rows-per-category", type=int, default=20_000)
    command.add_argument("--seed", type=int, default=20260820)
    command.add_argument("--temp-dir")
    command.set_defaults(func=make_public_proxy_command)

    command = commands.add_parser("research-llm-diagnostics")
    command.add_argument("--items")
    command.add_argument("--matches-llm")
    command.add_argument("--gold")
    command.add_argument("--silver")
    command.add_argument("--output")
    command.add_argument("--temp-dir")
    command.set_defaults(func=research_llm_diagnostics_command)

    command = commands.add_parser("research-proxy-shift")
    command.add_argument("--human-items")
    command.add_argument("--human-matches")
    command.add_argument("--llm-items")
    command.add_argument("--matches-llm")
    command.add_argument("--silver")
    command.add_argument("--output")
    command.add_argument("--max-pairs-per-source-category", type=int, default=1000)
    command.add_argument("--min-pairs-per-source-category", type=int, default=100)
    command.add_argument("--seed", type=int, default=20260812)
    command.add_argument("--temp-dir")
    command.set_defaults(func=research_proxy_shift_command)

    command = commands.add_parser("prepare-novel-holdout")
    command.add_argument("--items")
    command.add_argument("--holdout")
    command.add_argument("--canonical-output")
    command.add_argument("--output")
    command.add_argument("--workers", type=int)
    command.add_argument("--chunk-size", type=int, default=8192)
    command.set_defaults(func=prepare_novel_holdout_command)

    command = commands.add_parser("evaluate-novel-holdout")
    command.add_argument("--pairs")
    command.add_argument("--scores")
    command.add_argument("--fusion")
    command.add_argument("--output")
    command.add_argument("--min-hybrid-gain", type=float, default=0.002)
    command.add_argument("--public-macro-ap", type=float, default=0.3664957802653864)
    command.add_argument("--near-public-margin", type=float, default=0.05)
    command.set_defaults(func=evaluate_novel_holdout_command)

    command = commands.add_parser("prepare-pairs")
    command.add_argument("--dataset", choices=["gold", "silver"], required=True)
    command.add_argument("--chunk-size", type=int, default=8192)
    command.add_argument(
        "--text-only",
        action="store_true",
        help="Emit only text/category columns, skipping the 67 CatBoost pair features. "
        "Valid for Silver, which is used solely for cross-encoder pretraining; the "
        "features exist for the fusion model, which trains on Gold OOF only.",
    )
    command.add_argument(
        "--transductive-items",
        help="Full raw or canonical items parquet used only for label-free item statistics",
    )
    command.add_argument(
        "--transductive-profile",
        choices=["h12-v1", "h12-v1-c1", "h12-v1-c2", "h12-v1-c1-c2"],
        default="h12-v1",
        help="Immutable item-statistics profile for --transductive-items",
    )
    command.set_defaults(func=prepare_pairs_command)

    command = commands.add_parser("train-cross-encoder")
    command.add_argument("--stage", choices=["silver", "fold", "final"], required=True)
    command.add_argument("--fold", type=int)
    command.add_argument("--input")
    command.add_argument("--output")
    command.add_argument("--base-model")
    command.add_argument("--hard-negatives")
    command.add_argument("--epochs", type=int)
    command.add_argument(
        "--positive-prevalence",
        type=float,
        help=(
            "Target positive share in deterministic 16-row category microgroups; "
            "record this explicit operating point in the run manifest"
        ),
    )
    command.add_argument(
        "--ranking-from-epoch",
        type=int,
        help=(
            "Epoch index from which the ranking term applies; 1 is the historical "
            "behaviour, a value above --epochs keeps the loss pure BCE throughout"
        ),
    )
    command.add_argument(
        "--max-optimizer-steps",
        type=int,
        help="Bound a train receipt by optimizer steps; intended for timing only",
    )
    command.add_argument(
        "--no-save-model",
        action="store_true",
        help="Keep a bounded timing receipt from exporting a candidate model",
    )
    command.add_argument(
        "--batch-fixture",
        help="Immutable source-row microbatch fixture for a controlled training receipt",
    )
    command.add_argument(
        "--fixture-groups-per-batch",
        type=int,
        default=1,
        help="Number of consecutive fixture microbatches concatenated into one batch",
    )
    command.add_argument("--disable-hard-negatives", action="store_true")
    command.add_argument(
        "--schema-robustness-spec",
        help="Train-only schema robustness YAML; allowed only with --stage final",
    )
    command.add_argument(
        "--checkpoint-dir",
        help="Local directory for resumable step-level checkpoints (enables Spot resume)",
    )
    command.add_argument(
        "--checkpoint-gcs",
        help="gs:// prefix to mirror checkpoints to and pull them from on restart",
    )
    command.add_argument(
        "--checkpoint-interval-seconds",
        type=float,
        default=1200.0,
        help="Minimum wall-clock seconds between checkpoints (default 1200 = 20 min)",
    )
    command.set_defaults(func=train_cross_encoder_command)

    command = commands.add_parser("score-cross-encoder")
    command.add_argument("--fold", type=int)
    command.add_argument("--input")
    command.add_argument("--model")
    command.add_argument("--output")
    command.add_argument(
        "--max-length",
        type=int,
        help="Override cross_encoder.max_length; the scored length is a serving "
        "contract, so state it explicitly rather than via a shipped config",
    )
    command.add_argument("--batch-size", type=int, help="Override eval batch size")
    command.set_defaults(func=score_cross_encoder_command)

    command = commands.add_parser("mine-hard-negatives")
    command.add_argument("--gold")
    command.add_argument("--oof")
    command.add_argument("--output")
    command.add_argument("--top-fraction", type=float, default=0.15)
    command.set_defaults(func=mine_hard_negatives_command)

    command = commands.add_parser("materialize-augmentation")
    command.add_argument("--kind", choices=("attribute-dropout", "hard-identifiers"), required=True)
    command.add_argument("--input", help="Folded, real-only Gold pair table")
    command.add_argument("--output", required=True)
    command.add_argument(
        "--catalogue-items",
        help="Canonical full catalogue; required only for --kind hard-identifiers",
    )
    command.add_argument("--seed", type=int, help="Immutable materialization seed")
    command.add_argument("--min-identifier-idf", type=float, default=0.70)
    command.add_argument("--min-lcp-fraction", type=float, default=0.50)
    command.add_argument("--max-rows", type=int, default=4_066)
    command.set_defaults(func=materialize_augmentation_command)

    command = commands.add_parser("research-run-variant")
    command.add_argument(
        "--variant", choices=("gold_only", "silver_control", "final_like"), required=True
    )
    command.add_argument("--fold", type=int, default=0)
    command.add_argument("--gold")
    command.add_argument("--output", required=True)
    command.add_argument("--base-model")
    command.add_argument("--silver-model")
    command.add_argument("--mining-model")
    command.add_argument("--hard-negative-fraction", type=float, default=0.15)
    command.add_argument(
        "--schema-robustness-spec",
        help="Versioned YAML transform spec; accepted only by the silver_control research variant",
    )
    command.set_defaults(func=research_run_variant_command)

    command = commands.add_parser("research-compare")
    command.add_argument("--gold")
    command.add_argument("--control", required=True)
    command.add_argument("--candidate", required=True)
    command.add_argument("--fold", type=int, default=0)
    command.add_argument("--samples", type=int, default=1000)
    command.add_argument("--seed", type=int, default=20260812)
    command.add_argument("--output")
    command.set_defaults(func=research_compare_command)

    command = commands.add_parser("research-assess-candidate")
    command.add_argument("--gold")
    command.add_argument("--control", required=True)
    command.add_argument("--candidate", required=True)
    command.add_argument("--fold", type=int, default=0)
    command.add_argument("--minimum-macro-gain", type=float, default=0.005)
    command.add_argument("--maximum-category-drop", type=float, default=0.02)
    command.add_argument("--samples", type=int, default=1000)
    command.add_argument("--seed", type=int, default=20260812)
    command.add_argument("--output", required=True)
    command.set_defaults(func=research_assess_candidate_command)

    command = commands.add_parser("research-diagnose-scores")
    command.add_argument("--gold")
    command.add_argument("--oof", required=True)
    command.add_argument("--final", required=True)
    command.add_argument("--top-fraction", type=float, default=0.01)
    command.add_argument("--output", required=True)
    command.set_defaults(func=research_diagnose_scores_command)

    command = commands.add_parser("research-diagnose-fusion-shift")
    command.add_argument("--gold")
    command.add_argument("--oof", required=True)
    command.add_argument("--final", required=True)
    command.add_argument("--fusion", required=True)
    command.add_argument("--top-fraction", type=float, default=0.01)
    command.add_argument("--output", required=True)
    command.set_defaults(func=research_diagnose_fusion_shift_command)

    command = commands.add_parser("research-graph-diagnostics")
    command.add_argument("--gold")
    command.add_argument("--scores", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=research_graph_diagnostics_command)

    command = commands.add_parser("train-fusion")
    command.add_argument("--oof")
    command.add_argument("--output")
    command.add_argument(
        "--score-source",
        choices=("single_probability", "mean_probability"),
        default="single_probability",
    )
    command.add_argument(
        "--score-contract",
        choices=("category_rank", "raw"),
        default="raw",
        help=(
            "raw keeps every score independent of the inference batch. category_rank "
            "ranks within the batch and must be asked for explicitly: it is what "
            "candidate A shipped, scoring 0.4227 against a raw control at 0.4401"
        ),
    )
    command.add_argument(
        "--transductive-profile",
        choices=["h12-v1", "h12-v1-c1", "h12-v1-c2", "h12-v1-c1-c2"],
        help="Fail-closed profile required for newly trained C1/C2 fusions",
    )
    command.add_argument(
        "--expected-feature-schema-sha256",
        help="Predeclared full numeric schema digest; mismatches fail before CatBoost training",
    )
    command.set_defaults(func=train_fusion_command)

    command = commands.add_parser("build-submission")
    command.add_argument("--cross-encoder")
    command.add_argument("--fusion")
    command.add_argument("--output")
    command.add_argument("--image")
    command.add_argument("--score-mode", choices=("hybrid", "text"))
    command.add_argument("--holdout-report")
    command.add_argument("--allow-ungated-build", action="store_true")
    command.set_defaults(func=build_submission_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
