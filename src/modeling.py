"""Rebuild the frozen selected pipeline without loading a version-specific pickle."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.feature_preparation import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERIC_FEATURES,
    make_features,
)


def build_preprocessor(engineered=False, scale_numeric=False):
    """Use the same preprocessing definitions as notebook 04."""
    numeric_columns = NUMERIC_FEATURES + (ENGINEERED_FEATURES if engineered else [])
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric_columns),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def build_selected_pipeline(configuration):
    """Return an unfitted copy of the already-selected configuration; no search."""
    if configuration["model"] != "Histogram Gradient Boosting":
        raise ValueError("This continuation expects the frozen boosting configuration.")
    if configuration["feature_version"] not in {"Original", "Engineered"}:
        raise ValueError("Unknown feature version in the selection record.")
    if configuration["input_columns"] != NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        raise ValueError("The saved input schema differs from feature_preparation.py.")
    engineered = configuration["feature_version"] == "Engineered"
    pipeline = Pipeline([
        ("features", FunctionTransformer(
            make_features, kw_args={"engineered": engineered}, validate=False,
        )),
        ("preprocessing", build_preprocessor(engineered=engineered)),
        ("model", HistGradientBoostingRegressor(
            random_state=42, early_stopping=False,
        )),
    ])
    pipeline.set_params(**configuration["parameters"])
    if pipeline.get_params()["model__early_stopping"] is not False:
        raise ValueError("Automatic ungrouped early stopping must stay disabled.")
    return pipeline
