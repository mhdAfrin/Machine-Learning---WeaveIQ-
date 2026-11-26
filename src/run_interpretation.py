"""Interpret the frozen model on saved development folds, without reading test data.

From the WeaveIQ folder: python -m src.run_interpretation
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits

from src.modeling import build_selected_pipeline


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_interpretation(project, repeats=5, output_dir=None,
                       allow_version_mismatch=False):
    project = Path(project).resolve()
    if repeats < 2:
        raise ValueError("Use at least two shuffle repeats.")
    configuration_path = project / "reports/selected_model_config.json"
    development_path = project / "data/processed/development.csv"
    settings_path = project / "reports/tuning/run_settings.json"
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    expected_version = settings["sklearn_version"]
    if sklearn.__version__ != expected_version and not allow_version_mismatch:
        raise RuntimeError(
            f"Saved experiment used scikit-learn {expected_version}; current version "
            f"is {sklearn.__version__}. Run from the original project's .venv. "
            "For a separately labelled portability experiment only, use "
            "--allow-version-mismatch --output-dir reports/portability_check."
        )
    if sklearn.__version__ != expected_version and allow_version_mismatch:
        if output_dir is None or Path(output_dir).resolve() == (project / "reports").resolve():
            raise ValueError("Use a separate --output-dir for a version-mismatch run.")

    development = pd.read_csv(development_path, dtype={"ID": "string"}, low_memory=False)
    inputs = configuration["input_columns"]
    target = configuration["target"]
    assert development["ID"].notna().all(), "Missing order IDs."
    assert development["source_row"].is_unique, "Repeated source row references."
    assert set(development["cv_fold"].unique()) == set(range(5)), "Expected five saved folds."
    assert development.groupby("ID")["cv_fold"].nunique().eq(1).all(), "An ID crosses folds."
    assert np.isfinite(development[target]).all(), "Invalid target values."
    assert development[target].ge(0).all(), "Negative targets."

    destination = Path(output_dir).resolve() if output_dir else project / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    figure_dir = destination / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    rows, fold_scores = [], []

    for fold in range(5):
        training = development["cv_fold"].ne(fold)
        validation = development["cv_fold"].eq(fold)
        assert set(development.loc[training, "ID"]).isdisjoint(
            development.loc[validation, "ID"]
        ), "Order leakage within a fold."
        estimator = build_selected_pipeline(configuration)
        # The original tuning workers limited native libraries to one thread.
        with threadpool_limits(limits=1):
            estimator.fit(development.loc[training, inputs], development.loc[training, target])
            validation_mae = mean_absolute_error(
                development.loc[validation, target],
                estimator.predict(development.loc[validation, inputs]),
            )
            importance = permutation_importance(
                estimator, development.loc[validation, inputs],
                development.loc[validation, target],
                scoring="neg_mean_absolute_error", n_repeats=repeats,
                random_state=42, n_jobs=1,
            )
        fold_scores.append({"Fold": fold, "Validation_MAE": float(validation_mae)})
        for position, feature in enumerate(inputs):
            rows.append({
                "Fold": fold, "Feature": feature,
                "MAE_increase": float(importance.importances_mean[position]),
                "Shuffle_std": float(importance.importances_std[position]),
            })
        print(f"Feature importance: fold {fold + 1}/5 complete; "
              f"validation MAE={validation_mae:.6f}", flush=True)

    by_fold = pd.DataFrame(rows)
    summary = (by_fold.groupby("Feature")
               .agg(MAE_increase=("MAE_increase", "mean"),
                    Fold_std=("MAE_increase", "std"))
               .sort_values("MAE_increase", ascending=False).reset_index())
    scores = pd.DataFrame(fold_scores)
    # Compare with stored scores for the same frozen settings, not with test results.
    feature_label = configuration["feature_version"].lower()
    previous_trials = pd.read_csv(
        project / f"reports/tuning/{feature_label}_histogram_gradient_boosting_trials.csv"
    )
    previous_best = previous_trials.loc[previous_trials["mean_test_MAE"].idxmax()]
    expected_scores = np.array([-previous_best[f"split{i}_test_MAE"] for i in range(5)])
    scores["Stored_validation_MAE"] = expected_scores
    scores["Absolute_difference"] = np.abs(scores.Validation_MAE - expected_scores)
    reproduced = bool(np.allclose(scores.Validation_MAE, expected_scores, rtol=0, atol=1e-6))

    by_fold.to_csv(destination / "feature_importance_by_fold.csv", index=False)
    summary.to_csv(destination / "feature_importance_summary.csv", index=False)
    scores.to_csv(destination / "feature_importance_validation_check.csv", index=False)
    plot_data = summary.sort_values("MAE_increase")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_data.Feature, plot_data.MAE_increase,
            xerr=plot_data.Fold_std, capsize=3, color="#2563a6")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Average validation MAE increase after shuffling")
    ax.set_title("Frozen configuration: development-fold input importance")
    fig.tight_layout()
    fig.savefig(figure_dir / "feature_importance.png", dpi=160)
    plt.close(fig)

    metadata = {
        "model": configuration["model"], "feature_version": configuration["feature_version"],
        "parameters": configuration["parameters"], "random_seed": 42,
        "folds": 5, "shuffle_repeats": repeats, "native_threads": 1,
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__, "numpy_version": np.__version__,
        "joblib_version": joblib.__version__, "original_sklearn_version": expected_version,
        "version_matches_original": sklearn.__version__ == expected_version,
        "stored_validation_scores_reproduced_to_1e_6": reproduced,
        "development_sha256": _digest(development_path),
        "configuration_sha256": _digest(configuration_path),
        "test_files_read": False, "saved_model_overwritten": False,
        "runtime_seconds": perf_counter() - start,
        "interpretation": (
            "Exploratory reliance of a weak predictive model. Folds were previously used "
            "for tuning. Correlated inputs and implausible shuffled combinations can distort "
            "individual rankings. Raw-input importance includes downstream engineered features. "
            "Fold standard deviations are not confidence intervals; importance is not causal."
        ),
    }
    (destination / "feature_importance_run.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print("Stored validation scores reproduced:", reproduced)
    if not reproduced:
        print("REVIEW NOTE: These refit validation scores differ from the stored "
              "tuning scores. Inspect feature_importance_validation_check.csv "
              "and the recorded package versions before claiming exact reproduction.")
    print("Reports saved to:", destination)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-version-mismatch", action="store_true")
    arguments = parser.parse_args()
    run_interpretation(arguments.project, arguments.repeats,
                       arguments.output_dir, arguments.allow_version_mismatch)


if __name__ == "__main__":
    main()
