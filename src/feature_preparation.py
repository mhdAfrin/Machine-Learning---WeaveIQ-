import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "Req_Finish_Fabrics",
    "Fabric_Allowance",
    "Shrink_allow",
    "Req_grey_fabric",
    "Req_beam_length(yds)",
    "epi",
    "ppi",
]

CATEGORICAL_FEATURES = ["Construction"]

ENGINEERED_FEATURES = [
    "epi_ppi_product",
    "epi_ppi_ratio",
    "planned_grey_extra",
]


def make_features(data, engineered=False):
    # Select only explicitly permitted prediction inputs
    features = data[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ].copy()

    # Convert numeric inputs; invalid entries become missing
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(
            features[column], errors="coerce"
        )

    features[NUMERIC_FEATURES] = features[
        NUMERIC_FEATURES
    ].replace([np.inf, -np.inf], np.nan)

    # Standardize construction text without changing its meaning
    construction = (
        features["Construction"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace("", pd.NA)
    )

    features["Construction"] = (
        construction.astype(object)
        .where(construction.notna(), np.nan)
    )

    if engineered:
        features["epi_ppi_product"] = (
            features["epi"] * features["ppi"]
        )

        features["epi_ppi_ratio"] = (
            features["epi"]
            / features["ppi"].where(features["ppi"] > 0)
        )

        features["planned_grey_extra"] = (
            features["Req_grey_fabric"]
            - features["Req_Finish_Fabrics"]
        )

    return features.replace([np.inf, -np.inf], np.nan)