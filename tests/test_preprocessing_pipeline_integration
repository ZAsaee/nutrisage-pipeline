# tests/test_preprocessing_pipeline_integration.py
import pandas as pd
import sys
from src.preprocessing.pipeline import main as pipeline_main
from src.config import settings


def test_pipeline_full(tmp_path):
    # Use a small synthetic dataset matching schema
    data = {col: [1.0, 2.0] for col in settings.feature_columns}
    data[settings.label_column] = ['a', 'b']
    df = pd.DataFrame(data)
    input_file = tmp_path / 'sample.parquet'
    output_file = tmp_path / 'processed.parquet'
    df.to_parquet(input_file)

    # Temporarily override sys.argv and call the argparse-based main()
    orig_argv = sys.argv
    sys.argv = ['pipeline', '--input',
                str(input_file), '--output', str(output_file)]
    try:
        pipeline_main()
    finally:
        sys.argv = orig_argv

    # Validate output
    out_df = pd.read_parquet(output_file)
    # Core feature and label columns must be present
    core_cols = settings.feature_columns + [settings.label_column]
    for c in core_cols:
        assert c in out_df.columns
    # Ensure engineered feature is included
    assert 'fat_carb_ratio' in out_df.columns
    # Row count remains unchanged
    assert len(out_df) == len(df)
