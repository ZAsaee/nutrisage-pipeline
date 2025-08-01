# src/feature_importance.py
import argparse
import joblib
import pandas as pd
from src.config import settings


def main():
    parser = argparse.ArgumentParser(
        description="Compute and save feature importances.")
    parser.add_argument("--model-input", default=settings.model_output_path,
                        help="Path to trained XGBoost model joblib file")
    parser.add_argument("--data-input", default=settings.preprocessed_data_file,
                        help="Path to preprocessed dataset for feature names")
    parser.add_argument("--output", default=settings.feature_importance_output,
                        help="Path to save feature importances (CSV)")
    parser.add_argument("--top-n", type=int, default=None,
                        help="If set, only save the top N features by importance")
    args = parser.parse_args()

    model = joblib.load(args.model_input)
    df = pd.read_parquet(args.data_input)
    feature_names = df.drop(settings.label_column, axis=1).columns

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_importances_'):
        importances = model.best_estimator_.feature_importances_
    else:
        raise ValueError("Model does not support feature_importances_")

    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False)

    if args.top_n:
        fi_df = fi_df.head(args.top_n)

    fi_df.to_csv(args.output, index=False)
    print(fi_df)


if __name__ == "__main__":
    main()
