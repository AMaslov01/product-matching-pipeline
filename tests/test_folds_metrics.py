from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from matchcup.folds import make_component_folds, validate_component_folds
from matchcup.metrics import macro_average_precision


def test_component_folds_do_not_leak_items(tmp_path: Path) -> None:
    source = tmp_path / "pairs.parquet"
    output = tmp_path / "folds.parquet"
    pq.write_table(
        pa.table(
            {
                "id1": [1, 2, 10, 20, 30, 40, 50, 60, 70, 80],
                "id2": [2, 3, 11, 21, 31, 41, 51, 61, 71, 81],
                "target": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            }
        ),
        source,
    )
    stats = make_component_folds(source, output, n_folds=2, seed=7)
    validate_component_folds(output)
    assert stats["rows"] == 10
    result = pq.read_table(output).to_pandas()
    assert result.loc[0, "fold"] == result.loc[1, "fold"]
    assert result.row_id.tolist() == list(range(10))


def test_macro_average_precision_weights_categories_equally() -> None:
    score, per_category = macro_average_precision(
        [1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], ["A", "A", "B", "B"]
    )
    assert score == 1.0
    assert per_category == {"A": 1.0, "B": 1.0}
