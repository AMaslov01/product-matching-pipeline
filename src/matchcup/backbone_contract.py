"""What an encoder and a decoder cross-encoder backbone disagree about at save time.

``inference.py`` has one scoring path and no model-name branches: it loads a
directory, tokenises ``(text_a, text_b)`` with ``padding=True,
pad_to_multiple_of=8``, and reads ``logits``.  mmBERT satisfies that contract
straight from the hub -- it declares a pad token, its tokenizer pads right, and
its pair post-processor already puts an ``<eos>`` between the two sides.
Qwen3-0.6B satisfies none of the three, and only the first of them fails
loudly; the other two produce a complete submission with a worse score and
nothing in the log.

The reconciliation therefore has to happen once, in the place the checkpoint is
written, and the saved directory has to carry it -- everything below is about
the *artifact*, not a runtime flag.  An archive that leaves this repo either has
the contract baked into its ``config.json`` and ``tokenizer_config.json`` or it
scores wrong on the contest H100.

``verify_backbone_contract`` re-derives the identical record from a saved
directory using nothing but its JSON, so a gate can assert
``verify_backbone_contract(d) == report["backbone_contract"]`` without importing
torch or reading the weights.

Every claim above about Qwen was measured on revision
``c1899de289a04d12100db370d81485cdf75e47ca`` under transformers 5.15.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONTRACT_VERSION = 1

PAD_SIDE = "right"
PAD_SIDE_KEY = "padding_side"

# Part 4.  Not a knob this module turns: ``run.py`` picks the serving numeric
# path, and ``inference.py`` refuses to treat that choice as free because a
# fusion trained on OOF produced under one precision is invalid against scores
# of the same weights produced under another.  What the contract owes the
# operator is the record of which precision a given backbone's OOF was produced
# at.  mmBERT keeps fp32 weights under fp16 autocast, the path every published
# Public score came from.  Qwen3 ships bfloat16 weights and has activation
# magnitudes fp16 autocast can overflow, so it has only one honest answer.
#
# The lookup is deliberately closed: an unlisted backbone raises rather than
# inheriting a neighbour's precision, because guessing here is exactly the
# silent mismatch the module exists to prevent.  Adding one is a line of code
# and a measurement, not a default.
_CUDA_PRECISION = {
    "modernbert": "fp16_amp",
    # The local resume smoke fixture is a tiny BERT; it shares modernbert's
    # fp32-weights-under-autocast path, and listing it keeps that test running
    # against the same code the real backbones take.
    "bert": "fp16_amp",
    "qwen3": "bf16_weights",
}

# Two probe sides that share no substring, so whatever sits between them in the
# encoding is unambiguously the separator and not a shared subword.
_PROBE_A = "name_a: alpha"
_PROBE_B = "name_b: beta"


def apply_backbone_contract(model: Any, tokenizer: Any) -> dict[str, Any]:
    """Make ``model`` and ``tokenizer`` serveable in place; return the contract.

    Idempotent by construction -- each part re-reads the live objects and skips
    work already done -- so it is safe to call at model construction *and*
    before every ``save_pretrained``.  Both are needed: the pad token and the
    pair separator have to be in force while the model trains, or the weights
    are fitted to an input shape serving will never produce.
    """
    config = model.config
    model_type = getattr(config, "model_type", None)
    if model_type not in _CUDA_PRECISION:
        raise ValueError(
            f"No backbone contract for model_type {model_type!r}; "
            f"known backbones: {sorted(_CUDA_PRECISION)}"
        )
    _pin_pad_token(config, tokenizer)
    _pin_padding_side(tokenizer)
    _install_pair_separator(tokenizer)
    separator = _separator_from_tokenizer(tokenizer)
    if separator is None:
        raise ValueError("Pair separator did not take effect on this tokenizer")
    return {
        "contract_version": CONTRACT_VERSION,
        "model_type": model_type,
        "pad_token_id": int(config.pad_token_id),
        "padding_side": PAD_SIDE,
        "pair_separator": separator,
        "cuda_precision": _CUDA_PRECISION[model_type],
    }


def verify_backbone_contract(model_dir: str | Path) -> dict[str, Any]:
    """Re-derive the contract from a saved directory, raising on any violation.

    Reads only the three JSON files, so it runs as a packaging gate without
    torch, transformers, or the weights -- the three costs a check on an archive
    should not have to pay.
    """
    model_dir = Path(model_dir)
    config = _read_json(model_dir / "config.json")
    tokenizer_config = _read_json(model_dir / "tokenizer_config.json")
    tokenizer_json = _read_json(model_dir / "tokenizer.json")

    model_type = config.get("model_type")
    if model_type not in _CUDA_PRECISION:
        raise ValueError(
            f"{model_dir} holds model_type {model_type!r}, which has no backbone contract"
        )
    pad_token_id = config.get("pad_token_id")
    if not isinstance(pad_token_id, int) or isinstance(pad_token_id, bool):
        raise ValueError(
            f"{model_dir}/config.json declares no integer pad_token_id; a decoder head "
            "pooling the last non-padding token cannot run a batch larger than one"
        )
    padding_side = tokenizer_config.get(PAD_SIDE_KEY)
    if padding_side != PAD_SIDE:
        # An absent key is a violation, not a pass.  It would leave the archive
        # taking whichever side the contest image's transformers happens to
        # default to, which is not the version that trained the model.
        raise ValueError(
            f"{model_dir}/tokenizer_config.json must pin {PAD_SIDE_KEY}={PAD_SIDE!r}, "
            f"found {padding_side!r}"
        )
    separator = _separator_from_template(_pair_template(tokenizer_json.get("post_processor")))
    if separator is None:
        raise ValueError(
            f"{model_dir}/tokenizer.json has no pair post-processor separating side A from "
            "side B; the last line of A would fuse into the first line of B"
        )
    return {
        "contract_version": CONTRACT_VERSION,
        "model_type": model_type,
        "pad_token_id": pad_token_id,
        "padding_side": padding_side,
        "pair_separator": separator,
        "cuda_precision": _CUDA_PRECISION[model_type],
    }


def _pin_pad_token(config: Any, tokenizer: Any) -> None:
    """Part 1: give the config a pad token id.

    ``Qwen3ForSequenceClassification`` pools the last non-padding token, and the
    shared implementation it inherits raises ``Cannot handle batch sizes > 1 if
    no padding token is defined.`` when ``config.pad_token_id`` is None.
    Qwen3-0.6B's published ``config.json`` defines none, so a 64-row batch dies
    on the first forward pass.  This is the only part of the contract that fails
    loudly, and it is here so the three silent ones travel with it.
    """
    if getattr(config, "pad_token_id", None) is not None:
        return
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        raise ValueError("Backbone tokenizer defines no pad token to pin onto the config")
    config.pad_token_id = int(pad_token_id)


def _pin_padding_side(tokenizer: Any) -> None:
    """Part 2: pad on the right, and make the saved directory say so.

    Serving calls ``tokenizer(a, b, padding=True, pad_to_multiple_of=8)`` and
    never passes ``padding_side``; it comes from ``tokenizer_config.json``.  For
    a decoder the side is not cosmetic: ``Qwen3Model.forward`` builds
    ``position_ids`` as ``arange(seq_len)`` whenever the caller omits them, and
    neither the training collate nor ``inference.py``'s ``encode`` passes them.
    Left padding therefore spends RoPE positions on pad tokens and shifts every
    real token by a per-row amount that depends on the rest of its batch.

    Assigning the attribute is not enough to survive the save.
    ``save_pretrained`` writes ``tokenizer_config.json`` from ``init_kwargs``,
    and Qwen's shipped config carries no ``padding_side`` entry, so a plain
    assignment is dropped on the way out (measured: the key is simply absent
    from the saved file).  mmBERT's shipped config already carries
    ``"padding_side": "right"``, which makes both writes here no-ops that leave
    its saved tokenizer byte-identical.
    """
    tokenizer.padding_side = PAD_SIDE
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if init_kwargs is None:
        raise ValueError("Tokenizer exposes no init_kwargs, so padding_side cannot be persisted")
    init_kwargs[PAD_SIDE_KEY] = PAD_SIDE


def _install_pair_separator(tokenizer: Any) -> None:
    """Part 3: stop the two sides of the pair from fusing into one string.

    Serving encodes ``tokenizer(list_a, list_b)``.  Qwen's tokenizer carries
    only a ``ByteLevel`` post-processor and no pair template at all, so the two
    sides are concatenated with nothing between them: ``serialize_pair`` hands
    over ``"...attributes_a: X"`` and
    ``"category_b: Y..."`` and the model reads ``"...Xcategory_b: Y..."``.
    ``serialize.py`` suffixes every field with ``_a``/``_b``, so the boundary is
    still recoverable a token or two later, but the last line of A and the first
    line of B share subword tokens -- and comparing those lines is the whole
    job.

    The separator is the tokenizer's own EOS rather than an invented special
    token: a new token means a new embedding row starting from noise, which is a
    worse trade than reusing one the backbone has already trained on.
    """
    if _separator_from_tokenizer(tokenizer) is not None:
        return
    from tokenizers import processors

    eos = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id
    if eos is None or eos_id is None:
        raise ValueError("Backbone tokenizer defines no EOS token to separate the pair with")
    backend = tokenizer.backend_tokenizer
    # Chained rather than replaced: Qwen's existing ByteLevel post-processor
    # owns byte-level offset handling, and swapping it out for a bare template
    # would quietly change single-sequence encodings too.
    backend.post_processor = processors.Sequence(
        [
            backend.post_processor,
            processors.TemplateProcessing(
                single="$A",
                pair=f"$A {eos} $B",
                special_tokens=[(eos, eos_id)],
            ),
        ]
    )


def _separator_from_tokenizer(tokenizer: Any) -> dict[str, Any] | None:
    """Tokens a live tokenizer actually emits between side A and side B.

    Probing beats reading the post-processor back: it is the same call shape
    serving makes, so it cannot disagree with what the model will see, and it
    doubles as the idempotence test for ``_install_pair_separator``.
    """
    encoded = tokenizer([_PROBE_A], [_PROBE_B])
    sides = encoded.sequence_ids(0)
    a_positions = [index for index, side in enumerate(sides) if side == 0]
    b_positions = [index for index, side in enumerate(sides) if side == 1]
    if not a_positions or not b_positions:
        raise ValueError("Tokenizer did not encode both sides of the probe pair")
    ids = list(encoded["input_ids"][0])[a_positions[-1] + 1 : b_positions[0]]
    if not ids:
        return None
    return {
        "tokens": list(tokenizer.convert_ids_to_tokens(ids)),
        "ids": [int(value) for value in ids],
    }


def _pair_template(post_processor: Any) -> dict[str, Any] | None:
    """Find the ``TemplateProcessing`` stage that decides the final pair layout."""
    if not isinstance(post_processor, dict):
        return None
    if post_processor.get("type") == "TemplateProcessing":
        return post_processor
    if post_processor.get("type") == "Sequence":
        # A Sequence applies its members in order, so the last template to run
        # is the one whose layout survives.
        for member in reversed(post_processor.get("processors") or []):
            found = _pair_template(member)
            if found is not None:
                return found
    return None


def _separator_from_template(template: dict[str, Any] | None) -> dict[str, Any] | None:
    """Read the same separator record out of a saved ``tokenizer.json`` template."""
    if template is None:
        return None
    pair = template.get("pair") or []
    a_positions = [
        index
        for index, entry in enumerate(pair)
        if isinstance(entry, dict) and entry.get("Sequence", {}).get("id") == "A"
    ]
    b_positions = [
        index
        for index, entry in enumerate(pair)
        if isinstance(entry, dict) and entry.get("Sequence", {}).get("id") == "B"
    ]
    if not a_positions or not b_positions:
        return None
    between = pair[a_positions[-1] + 1 : b_positions[0]]
    tokens = [
        entry["SpecialToken"]["id"]
        for entry in between
        if isinstance(entry, dict) and "SpecialToken" in entry
    ]
    if not tokens:
        return None
    special_tokens = template.get("special_tokens") or {}
    return {
        "tokens": tokens,
        "ids": [
            int(value)
            for token in tokens
            for value in (special_tokens.get(token) or {}).get("ids", [])
        ],
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Saved backbone is missing {path.name}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not read {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload
