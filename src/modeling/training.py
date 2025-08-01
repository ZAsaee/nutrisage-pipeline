# src/training.py
import argparse
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from skopt import BayesSearchCV
from src.config import settings


def train_model(X, y, **kwargs):
    model = XGBClassifier(use_label_encoder=False,
                          eval_metric='mlogloss', **kwargs)
    model.fit(X, y)
    return model


def bayesian_search(X, y):
    param_space = {
        'n_estimators': (50, 300),
        'max_depth': (3, 12),
        'learning_rate': (0.01, 0.3, 'log-uniform'),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0)
    }
    base = XGBClassifier(use_label_encoder=False,
                         eval_metric='mlogloss', random_state=settings.random_state)
    opt = BayesSearchCV(
        estimator=base,
        search_spaces=param_space,
        n_iter=settings.bayes_search_iterations,
        cv=3,
        random_state=settings.random_state,
        n_jobs=-1
    )
    opt.fit(X, y)
    print("Best parameters found:", opt.best_params_)
    return opt.best_estimator_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a classification model with optional Bayesian search.")
    parser.add_argument("--input", default=settings.preprocessed_data_file,
                        help="Path to preprocessed parquet dataset")
    parser.add_argument("--model-output", default=settings.model_output_path,
                        help="Path to save the trained model")
    parser.add_argument("--bayes-search", action="store_true",
                        help="Run Bayesian hyperparameter search before training")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    X = df.drop(settings.label_column, axis=1)
    y = df[settings.label_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.test_size, random_state=settings.random_state
    )

    if args.bayes_search:
        model = bayesian_search(X_train, y_train)
    else:
        model = train_model(
            X_train, y_train,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=settings.random_state
        )

    preds = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))
    joblib.dump(model, args.model_output_path)
