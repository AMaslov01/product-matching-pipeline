"""The four things a Qwen checkpoint gets wrong, and the gate that catches them.

Three of the four fail silently -- a wrongly padded, wrongly separated archive
returns a complete submission with a worse score -- so each one is tested twice:
once that ``apply_backbone_contract`` fixes it, and once that
``verify_backbone_contract`` refuses a saved directory that still has it.  The
stubs stand in for a real tokenizer because the failures live in three lines of
JSON, not in the weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from matchcup.backbone_contract import (
    CONTRACT_VERSION,
    apply_backbone_contract,
    verify_backbone_contract,
)

QWEN = "Qwen/Qwen3-0.6B"


class _StubEncoding(dict):
    """The two things ``_separator_from_tokenizer`` reads off a BatchEncoding."""

    def __init__(self, input_ids: list[int], sides: list[int | None]) -> None:
        super().__init__(input_ids=[input_ids])
        self._sides = sides

    def sequence_ids(self, index: int) -> list[int | None]:
        assert index == 0
        return self._sides


class _StubTokenizer:
    """Exactly the tokenizer surface the contract touches, and nothing else.

    The pair encoding is derived from the live post-processor rather than from a
    flag, so installing a separator really does change what a later probe sees --
    which is what makes the idempotence test meaningful.
    """

    A_IDS = (11, 12)
    B_IDS = (21, 22)

    def __init__(
        self,
        *,
        pad_token_id: int | None = 7,
        eos_token: str | None = "<eos>",
        eos_token_id: int | None = 99,
        padding_side: str = "left",
        separated: bool = False,
    ) -> None:
        from tokenizers import processors

        self.pad_token_id = pad_token_id
        self.eos_token = eos_token
        self.eos_token_id = eos_token_id
        self.padding_side = padding_side
        self.init_kwargs: dict[str, object] = {}
        self.backend_tokenizer = SimpleNamespace(post_processor=processors.ByteLevel())
        self._shipped_post_processor = self.backend_tokenizer.post_processor
        self._shipped_separated = separated

    @property
    def separated(self) -> bool:
        return (
            self._shipped_separated
            or self.backend_tokenizer.post_processor is not self._shipped_post_processor
        )

    def __call__(self, texts_a: list[str], texts_b: list[str]) -> _StubEncoding:
        assert len(texts_a) == len(texts_b) == 1
        if self.separated:
            return _StubEncoding(
                [*self.A_IDS, self.eos_token_id, *self.B_IDS], [0, 0, None, 1, 1]
            )
        return _StubEncoding([*self.A_IDS, *self.B_IDS], [0, 0, 1, 1])

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.eos_token if value == self.eos_token_id else f"tok{value}" for value in ids]


def _qwen_pair() -> tuple[SimpleNamespace, _StubTokenizer]:
    """A decoder backbone as it comes off the hub: none of the four in place."""
    return SimpleNamespace(config=SimpleNamespace(model_type="qwen3", pad_token_id=None)), (
        _StubTokenizer()
    )


def _mmbert_pair() -> tuple[SimpleNamespace, _StubTokenizer]:
    """An encoder backbone as it comes off the hub: all four already in place."""
    return SimpleNamespace(config=SimpleNamespace(model_type="modernbert", pad_token_id=0)), (
        _StubTokenizer(eos_token="<eos>", eos_token_id=1, padding_side="right", separated=True)
    )


def test_pad_token_id_is_pinned_onto_the_config():
    model, tokenizer = _qwen_pair()
    record = apply_backbone_contract(model, tokenizer)
    assert model.config.pad_token_id == tokenizer.pad_token_id == 7
    assert record["pad_token_id"] == 7


def test_a_config_that_already_declares_a_pad_token_keeps_it():
    model, tokenizer = _qwen_pair()
    model.config.pad_token_id = 3
    apply_backbone_contract(model, tokenizer)
    assert model.config.pad_token_id == 3


def test_a_backbone_without_any_pad_token_is_refused():
    model, tokenizer = _qwen_pair()
    tokenizer.pad_token_id = None
    with pytest.raises(ValueError, match="no pad token"):
        apply_backbone_contract(model, tokenizer)


def test_padding_side_is_pinned_and_made_persistable():
    model, tokenizer = _qwen_pair()
    record = apply_backbone_contract(model, tokenizer)
    assert tokenizer.padding_side == "right"
    # The attribute alone is dropped by save_pretrained; init_kwargs is what
    # reaches tokenizer_config.json.
    assert tokenizer.init_kwargs["padding_side"] == "right"
    assert record["padding_side"] == "right"


def test_a_pair_separator_is_installed_when_the_two_sides_fuse():
    model, tokenizer = _qwen_pair()
    assert not tokenizer.separated
    record = apply_backbone_contract(model, tokenizer)
    assert tokenizer.separated
    assert record["pair_separator"] == {"tokens": ["<eos>"], "ids": [99]}


def test_an_existing_pair_separator_is_left_alone():
    model, tokenizer = _mmbert_pair()
    shipped = tokenizer.backend_tokenizer.post_processor
    record = apply_backbone_contract(model, tokenizer)
    assert tokenizer.backend_tokenizer.post_processor is shipped
    assert record["pair_separator"] == {"tokens": ["<eos>"], "ids": [1]}


def test_a_backbone_without_an_eos_token_cannot_be_separated():
    model, tokenizer = _qwen_pair()
    tokenizer.eos_token = None
    with pytest.raises(ValueError, match="no EOS token"):
        apply_backbone_contract(model, tokenizer)


def test_applying_twice_changes_nothing_the_second_time():
    model, tokenizer = _qwen_pair()
    first = apply_backbone_contract(model, tokenizer)
    installed = tokenizer.backend_tokenizer.post_processor
    second = apply_backbone_contract(model, tokenizer)
    assert second == first
    assert tokenizer.backend_tokenizer.post_processor is installed


@pytest.mark.parametrize(
    ("factory", "precision"),
    [(_qwen_pair, "bf16_weights"), (_mmbert_pair, "fp16_amp")],
)
def test_the_serving_precision_is_recorded_per_backbone(factory, precision):
    model, tokenizer = factory()
    assert apply_backbone_contract(model, tokenizer)["cuda_precision"] == precision


def test_the_encoder_backbone_is_an_observable_no_op_that_still_reports():
    model, tokenizer = _mmbert_pair()
    before = (
        model.config.pad_token_id,
        tokenizer.padding_side,
        tokenizer.backend_tokenizer.post_processor,
    )
    record = apply_backbone_contract(model, tokenizer)
    assert (
        model.config.pad_token_id,
        tokenizer.padding_side,
        tokenizer.backend_tokenizer.post_processor,
    ) == before
    # init_kwargs gains a key whose value the shipped tokenizer_config.json
    # already carries, so the saved file is unchanged.
    assert tokenizer.init_kwargs == {"padding_side": "right"}
    assert record == {
        "contract_version": CONTRACT_VERSION,
        "model_type": "modernbert",
        "pad_token_id": 0,
        "padding_side": "right",
        "pair_separator": {"tokens": ["<eos>"], "ids": [1]},
        "cuda_precision": "fp16_amp",
    }


def test_an_unlisted_backbone_is_refused_rather_than_guessed():
    model, tokenizer = _qwen_pair()
    model.config.model_type = "llama"
    with pytest.raises(ValueError, match="No backbone contract"):
        apply_backbone_contract(model, tokenizer)


def _save_backbone(
    root: Path,
    *,
    config: dict | None = None,
    tokenizer_config: dict | None = None,
    post_processor: dict | None = None,
) -> Path:
    """Write the three JSON files a saved Qwen directory verifies against."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "pad_token_id": 151643, **(config or {})}),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({"padding_side": "right", **(tokenizer_config or {})}), encoding="utf-8"
    )
    (root / "tokenizer.json").write_text(
        json.dumps({"post_processor": post_processor or _sequence_post_processor()}),
        encoding="utf-8",
    )
    return root


def _template_post_processor(separated: bool = True) -> dict:
    pair = [{"Sequence": {"id": "A", "type_id": 0}}]
    if separated:
        pair.append({"SpecialToken": {"id": "<|im_end|>", "type_id": 0}})
    pair.append({"Sequence": {"id": "B", "type_id": 0}})
    return {
        "type": "TemplateProcessing",
        "single": [{"Sequence": {"id": "A", "type_id": 0}}],
        "pair": pair,
        "special_tokens": {"<|im_end|>": {"id": "<|im_end|>", "ids": [151645]}},
    }


def _sequence_post_processor(separated: bool = True) -> dict:
    """The shape the contract actually writes: a template chained after ByteLevel."""
    return {
        "type": "Sequence",
        "processors": [
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False},
            _template_post_processor(separated),
        ],
    }


EXPECTED_RECORD = {
    "contract_version": CONTRACT_VERSION,
    "model_type": "qwen3",
    "pad_token_id": 151643,
    "padding_side": "right",
    "pair_separator": {"tokens": ["<|im_end|>"], "ids": [151645]},
    "cuda_precision": "bf16_weights",
}


def test_verify_reads_the_contract_back_out_of_a_saved_directory(tmp_path):
    assert verify_backbone_contract(_save_backbone(tmp_path / "qwen")) == EXPECTED_RECORD


def test_verify_finds_a_bare_template_as_well_as_a_chained_one(tmp_path):
    saved = _save_backbone(tmp_path / "bare", post_processor=_template_post_processor())
    assert verify_backbone_contract(saved) == EXPECTED_RECORD


def test_verify_rejects_a_config_without_a_pad_token_id(tmp_path):
    saved = _save_backbone(tmp_path / "qwen")
    assert verify_backbone_contract(saved) == EXPECTED_RECORD
    (saved / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
    with pytest.raises(ValueError, match="pad_token_id"):
        verify_backbone_contract(saved)


@pytest.mark.parametrize("padding_side", ["left", None])
def test_verify_rejects_anything_but_an_explicit_right_padding_side(tmp_path, padding_side):
    saved = _save_backbone(tmp_path / f"qwen-{padding_side}")
    assert verify_backbone_contract(saved) == EXPECTED_RECORD
    # None writes the key out entirely: an implied side is not a contract.
    payload = {} if padding_side is None else {"padding_side": padding_side}
    (saved / "tokenizer_config.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="padding_side"):
        verify_backbone_contract(saved)


@pytest.mark.parametrize(
    "post_processor",
    [
        pytest.param(_sequence_post_processor(separated=False), id="fused_pair_template"),
        pytest.param(
            {"type": "ByteLevel", "add_prefix_space": False}, id="no_template_at_all"
        ),
    ],
)
def test_verify_rejects_a_tokenizer_that_fuses_the_two_sides(tmp_path, post_processor):
    saved = _save_backbone(tmp_path / "qwen")
    assert verify_backbone_contract(saved) == EXPECTED_RECORD
    (saved / "tokenizer.json").write_text(
        json.dumps({"post_processor": post_processor}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="separating side A"):
        verify_backbone_contract(saved)


def test_verify_rejects_an_unlisted_backbone(tmp_path):
    saved = _save_backbone(tmp_path / "mystery", config={"model_type": "llama"})
    with pytest.raises(ValueError, match="no backbone contract"):
        verify_backbone_contract(saved)


def test_verify_rejects_a_directory_missing_a_file(tmp_path):
    saved = _save_backbone(tmp_path / "qwen")
    (saved / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="missing tokenizer.json"):
        verify_backbone_contract(saved)


def _cached_qwen_tokenizer():
    """The real Qwen tokenizer, or a skip -- never a download inside a test."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            QWEN, use_fast=True, local_files_only=True
        )
    except Exception as exc:  # noqa: BLE001 - any cache miss is the same skip
        pytest.skip(f"{QWEN} tokenizer is not in the local HF cache ({type(exc).__name__})")


def test_the_real_qwen_tokenizer_stops_fusing_the_pair(tmp_path):
    transformers = pytest.importorskip("transformers")
    tokenizer = _cached_qwen_tokenizer()
    config = transformers.AutoConfig.from_pretrained(QWEN, local_files_only=True)
    model = SimpleNamespace(config=config)

    side_a, side_b = "attributes_a: alpha", "category_b: beta"
    fused = tokenizer.decode(tokenizer([side_a], [side_b])["input_ids"][0])
    assert fused == side_a + side_b, "Gate 0 measured Qwen concatenating pairs with no separator"

    record = apply_backbone_contract(model, tokenizer)
    separated = tokenizer.decode(tokenizer([side_a], [side_b])["input_ids"][0])
    assert separated == f"{side_a}{tokenizer.eos_token}{side_b}"

    # The whole point of the contract is that it survives into the archive, and
    # padding_side in particular does not unless it reaches init_kwargs.
    saved = tmp_path / "qwen"
    config.save_pretrained(saved)
    tokenizer.save_pretrained(saved)
    assert verify_backbone_contract(saved) == record


def test_a_real_qwen3_head_cannot_score_a_batch_until_the_contract_is_applied(tmp_path):
    """The loud failure, on a real ``Qwen3ForSequenceClassification``.

    Weights are randomly initialised at a toy size rather than downloaded: the
    thing under test is the pooling head's pad-token requirement and the shape
    of what the tokenizer feeds it, neither of which depends on the values.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizer = _cached_qwen_tokenizer()

    config = transformers.Qwen3Config(
        vocab_size=tokenizer.vocab_size + 1000,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_labels=1,
        max_position_embeddings=512,
    )
    model = transformers.Qwen3ForSequenceClassification(config).eval()

    def encode():
        return tokenizer(
            ["attributes_a: alpha one two", "attributes_a: b"],
            ["category_b: beta", "category_b: gamma three four five"],
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=384,
            return_tensors="pt",
        )

    with pytest.raises(ValueError, match="Cannot handle batch sizes"):
        with torch.no_grad():
            model(**encode())

    record = apply_backbone_contract(model, tokenizer)
    encoded = encode()
    with torch.no_grad():
        assert model(**encoded).logits.shape == (2, 1)
    # Shorter row padded on the right, so no real token loses its RoPE position.
    assert int(encoded["input_ids"][0][-1]) == tokenizer.pad_token_id

    model.save_pretrained(tmp_path / "qwen")
    tokenizer.save_pretrained(tmp_path / "qwen")
    assert verify_backbone_contract(tmp_path / "qwen") == record
