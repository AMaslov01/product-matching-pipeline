"""A saved-backbone directory realistic enough for the packaging gate.

`build_submission` now verifies the backbone contract before staging, so a test
fixture that writes `config.json` as `{}` no longer stands in for a model
directory. Writing the three files a real `save_pretrained` produces keeps those
tests exercising packaging mechanics rather than working around the gate.
"""

from __future__ import annotations

import json
from pathlib import Path

# mmBERT ships all three already, which is why its contract is a no-op; a
# fixture that mirrors it is therefore the honest default.
_PAD_TOKEN_ID = 0
_EOS = "[SEP]"
_EOS_ID = 3


def write_saved_backbone(path: str | Path, *, model_type: str = "modernbert") -> Path:
    """Write the config and tokenizer files `verify_backbone_contract` reads."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps({"model_type": model_type, "pad_token_id": _PAD_TOKEN_ID}),
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps({"padding_side": "right"}), encoding="utf-8"
    )
    (path / "tokenizer.json").write_text(
        json.dumps(
            {
                "post_processor": {
                    "type": "TemplateProcessing",
                    "single": [{"Sequence": {"id": "A", "type_id": 0}}],
                    "pair": [
                        {"Sequence": {"id": "A", "type_id": 0}},
                        {"SpecialToken": {"id": _EOS, "type_id": 0}},
                        {"Sequence": {"id": "B", "type_id": 0}},
                    ],
                    "special_tokens": {_EOS: {"id": _EOS, "ids": [_EOS_ID], "tokens": [_EOS]}},
                }
            }
        ),
        encoding="utf-8",
    )
    return path
