"""
Minimal viable product for the CSIRO Image2Biomass prediction
competition.

This script provides a complete pipeline for loading the training and
test data, engineering simple features from the images and meta data,
training a multi‑output regression model and generating a submission
file.  The code is written with clarity and teaching in mind – each
function contains docstrings and explanatory comments so that readers
who are new to machine learning can follow along.

The implementation relies on the following Python packages:

* `pandas` and `numpy` for data manipulation.
* `Pillow` (PIL) for reading JPEG/PNG images.
* `scikit‑learn` for preprocessing utilities, wrapping multi‑output
  regression and computing evaluation metrics.
* `xgboost` for the base regression estimator.  It is installed by
  default in the Kaggle environment and does not require a GPU.

If you run this script inside a Kaggle notebook, ensure that the
working directory contains the competition data in the following
structure:

```
data/
  ├── train.csv
  ├── test.csv
  ├── train/
  │     ├── ... (image files)
  └── test/
        ├── ... (image files)
```

Example usage:
    python main.py --data-dir data

This will train the model on the training set, evaluate it on an
internal validation set, and write `submission.csv` into the current
working directory.  Adjust the `--val-size` argument to change the
proportion of data held out for validation.

Author: Julio Camargo
Date: 2025-12-07
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def extract_image_features(img_path: Path) -> np.ndarray:
    """Extract simple numeric features from an image.

    The Image2Biomass competition provides overhead photos of pasture
    plots.  While deep learning methods would normally be used to
    interpret such images, this baseline extracts low‑dimensional
    summary statistics that still capture colour and texture information.

    The features include:
      * Mean and standard deviation of the R, G and B channels.
      * Normalised colour histograms with 16 bins per channel.

    Parameters
    ----------
    img_path : Path
        Filesystem path to the JPEG/PNG image.

    Returns
    -------
    np.ndarray
        A one‑dimensional array of feature values.
    """
    try:
        with Image.open(img_path) as img:
            # Convert to RGB if necessary (grayscale or RGBA images)
            img = img.convert("RGB")
            arr = np.asarray(img, dtype=np.float32)
    except Exception as e:
        logging.error(f"Failed to read image {img_path}: {e}")
        # Return a vector of zeros if the image cannot be read
        return np.zeros(54, dtype=np.float32)

    # Compute per‑channel means and standard deviations
    mean_per_channel = arr.mean(axis=(0, 1))
    std_per_channel = arr.std(axis=(0, 1))

    # Compute normalised histograms for each channel
    hist_features: List[float] = []
    for channel in range(3):
        hist, _ = np.histogram(
            arr[:, :, channel], bins=16, range=(0.0, 255.0), density=True
        )
        hist_features.extend(hist.tolist())

    # Concatenate all features into a single vector
    features = np.concatenate([mean_per_channel, std_per_channel, np.array(hist_features)])
    return features


def load_and_prepare_data(
    train_csv: Path, images_dir: Path, data_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load the training data and prepare feature and target matrices.

    This function performs several preprocessing steps:
      1. Reads the `train.csv` file and pivots it so that each unique
         image has one row with all five target variables as separate
         columns.
      2. Extracts simple numeric features from each image file.
      3. Merges image features with meta data columns (NDVI,
         Height, Sampling_Date, State, Species).
      4. Returns the resulting feature matrix, target matrix and the
         list of columns to which one‑hot encoding should be applied.

    Parameters
    ----------
    train_csv : Path
        Path to the `train.csv` file.
    images_dir : Path
        Directory containing the training images.  The `image_path`
        column in the CSV is resolved relative to this directory.
    data_dir : Path
        Path to the data directory.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, List[str]]
        A tuple containing the feature matrix `X` (as a DataFrame),
        target matrix `y` (as a DataFrame with one column per target),
        and the list of categorical columns that require encoding.
    """
    logging.info("Loading training CSV...")
    df = pd.read_csv(train_csv)
    logging.info(f"Loaded {len(df):,} rows from {train_csv.name}")

    # Pivot the DataFrame so each unique image has one row with all targets
    logging.info("Pivoting targets to multi‑output format...")
    pivot_targets = df.pivot_table(
        index="image_path",
        columns="target_name",
        values="target",
        aggfunc="first"
    )

    # Keep the first occurrence of meta data for each image_path
    meta_cols = [
        "Sampling_Date",
        "State",
        "Species",
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
    ]
    meta_data = (
        df[["image_path"] + meta_cols]
        .drop_duplicates(subset=["image_path"])
        .set_index("image_path")
    )

    # Combine meta data and pivoted targets
    full_df = pivot_targets.join(meta_data, how="inner")
    full_df.reset_index(inplace=True)

    # Extract image features for each image
    logging.info("Extracting image features...")
    image_feature_list = []
    for img_rel_path in full_df["image_path"]:
        img_path = data_dir / img_rel_path
        image_feature_list.append(extract_image_features(img_path))
    image_features = np.vstack(image_feature_list)
    image_feature_cols = [
        f"img_feat_{i}" for i in range(image_features.shape[1])
    ]
    image_features_df = pd.DataFrame(image_features, columns=image_feature_cols)

    # Convert Sampling_Date to datetime and then to an ordinal number of days
    full_df["Sampling_Date"] = pd.to_datetime(full_df["Sampling_Date"])
    min_date = full_df["Sampling_Date"].min()
    full_df["Sampling_Date_ordinal"] = (
        (full_df["Sampling_Date"] - min_date).dt.days.astype(float)
    )

    # Assemble feature matrix X
    X = pd.concat(
        [
            image_features_df,
            full_df[[
                "Pre_GSHH_NDVI",
                "Height_Ave_cm",
                "Sampling_Date_ordinal",
                "State",
                "Species",
            ]].reset_index(drop=True),
        ],
        axis=1,
    )

    # Target matrix y
    target_cols = pivot_targets.columns.tolist()
    y = full_df[target_cols].reset_index(drop=True)

    # Categorical columns for one‑hot encoding
    categorical_cols = ["State", "Species"]
    for col in categorical_cols:
        X[col] = X[col].fillna("missing")

    return X, y, categorical_cols


def build_preprocessor(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    """Construct a scikit‑learn ColumnTransformer for preprocessing.

    The transformer will standardise numeric features and one‑hot encode
    categorical features.  Any remaining columns (such as the image
    features) are passed through unchanged by default.

    Parameters
    ----------
    categorical_cols : List[str]
        List of column names in the feature DataFrame that should be
        one‑hot encoded.
    numeric_cols : List[str]
        List of column names that should be standardised.

    Returns
    -------
    ColumnTransformer
        A fitted transformer ready to be used in a pipeline.
    """
    transformers = []
    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols,
            )
        )
    if numeric_cols:
        transformers.append(
            (
                "num",
                StandardScaler(),
                numeric_cols,
            )
        )

    # The remainder='passthrough' ensures image features are not touched
    preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")
    return preprocessor


def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, weights: Dict[str, float], target_names: List[str]) -> float:
    """Compute the weighted coefficient of determination (R²).

    Kaggle evaluates submissions using a globally weighted R² metric.
    This function applies the same weighting scheme across the target
    variables.  The weights dictionary must provide a weight for each
    target column name.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values of shape (n_samples, n_targets).
    y_pred : np.ndarray
        Predicted target values of the same shape as `y_true`.
    weights : Dict[str, float]
        Mapping from target name to its weight in the overall R².
    target_names : List[str]
        List of target column names corresponding to the columns in
        `y_true` and `y_pred`.

    Returns
    -------
    float
        Weighted R² score.
    """
    r2_scores = []
    weight_values = []
    for idx, name in enumerate(target_names):
        w = weights.get(name, 1.0)
        r2 = r2_score(y_true[:, idx], y_pred[:, idx])
        r2_scores.append(r2)
        weight_values.append(w)
    r2_scores = np.array(r2_scores)
    weight_values = np.array(weight_values)
    weighted_r2 = np.sum(weight_values * r2_scores) / weight_values.sum()
    return weighted_r2


def train_model(
    X: pd.DataFrame,
    y: pd.DataFrame,
    categorical_cols: List[str],
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[MultiOutputRegressor, ColumnTransformer, float]:
    """Train a multi‑output regression model and return the fitted objects.

    This function splits the data into a training and validation set,
    constructs the preprocessing pipeline and the estimator, fits the
    model and evaluates it on the validation set using the weighted
    R² metric.  It returns the trained model, the fitted
    preprocessor and the validation score.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix including image and meta features.
    y : pd.DataFrame
        Target matrix with one column per biomass component.
    categorical_cols : List[str]
        Columns of `X` that should be one‑hot encoded.  The
        remaining numeric columns will be standardised.
    val_size : float, default=0.2
        Fraction of the data to hold out for validation.
    random_state : int, default=42
        Random seed for reproducible splits.

    Returns
    -------
    Tuple[MultiOutputRegressor, ColumnTransformer, float]
        The trained model, the preprocessor and the weighted R² score
        on the validation split.
    """
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state
    )

    # Identify numeric columns (all image features + numeric meta features)
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    # Build the preprocessing transformer
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)

    # Fit the preprocessor on the training data
    logging.info("Fitting preprocessing transformer...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    # Define the base regressor.  Hyperparameters can be tuned later.
    base_estimator = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=0,
    )
    model = MultiOutputRegressor(base_estimator)

    logging.info("Training the model...")
    model.fit(X_train_processed, y_train.values)

    # Predict on validation set
    y_val_pred = model.predict(X_val_processed)

    # Compute weighted R².  Use the approximate weights from the competition.
    target_names = y.columns.tolist()
    default_weights = {
        "Dry_Green_g": 0.1,
        "Dry_Dead_g": 0.1,
        "Dry_Clover_g": 0.1,
        "GDM_g": 0.2,
        "Dry_Total_g": 0.5,
    }
    val_score = weighted_r2_score(
        y_val.values, y_val_pred, default_weights, target_names
    )
    logging.info(f"Validation weighted R²: {val_score:.4f}")

    return model, preprocessor, val_score


def predict_test(
    model: MultiOutputRegressor,
    preprocessor: ColumnTransformer,
    test_csv: Path,
    images_dir: Path,
    categorical_cols: List[str],
    data_dir: Path,
) -> pd.DataFrame:
    """Generate predictions for the test set.

    Parameters
    ----------
    model : MultiOutputRegressor
        Trained multi‑output regression model.
    preprocessor : ColumnTransformer
        Fitted preprocessing transformer.
    test_csv : Path
        Path to `test.csv`.
    images_dir : Path
        Directory containing the test images.
    categorical_cols : List[str]
        Categorical columns requiring encoding.
    data_dir : Path
        Path to the data directory.
    Returns
    -------
    pd.DataFrame
        A DataFrame with columns `sample_id` and `target` ready to be
        saved as `submission.csv`.
    """
    logging.info("Loading test CSV...")
    df_test = pd.read_csv(test_csv)
    logging.info(f"Loaded {len(df_test):,} rows from {test_csv.name}")

    # Determine the unique images in the test set
    unique_imgs = df_test["image_path"].unique()

    # Extract features for each unique image
    logging.info("Extracting image features for test set...")
    img_feat_dict = {}
    for img_rel_path in unique_imgs:
        img_path = data_dir / img_rel_path
        img_feat_dict[img_rel_path] = extract_image_features(img_path)

    # Build a DataFrame with one row per unique image
    meta_cols = [
        "Sampling_Date",
        "State",
        "Species",
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
    ]
    # For test meta data, we need to supply dummy or default values for
    # variables not present in test.csv.  According to the competition
    # description, test.csv contains only `sample_id`, `image_path` and
    # `target_name`.  We therefore populate the meta columns with
    # missing values (NaN) so that the column names exist.  Models
    # trained without meta data will ignore these columns.  If you
    # decide to include meta features in your model, you should load
    # them from any supplemental sources or drop them here.
    test_features = []
    for img_rel_path in unique_imgs:
        row = {
            "image_path": img_rel_path,
        }
        # Add empty meta fields
        for c in meta_cols:
            row[c] = np.nan
        test_features.append(row)
    df_test_feats = pd.DataFrame(test_features)
    df_test_feats.set_index("image_path", inplace=True)

    # Add image features
    image_feature_cols = [
        f"img_feat_{i}" for i in range(len(next(iter(img_feat_dict.values()))) )
    ]
    img_feat_df = pd.DataFrame.from_dict(img_feat_dict, orient="index")
    img_feat_df.columns = image_feature_cols

    # Combine meta and image features
    X_test = pd.concat([img_feat_df, df_test_feats], axis=1)

    # Convert Sampling_Date to ordinal days; fill missing with 0
    if "Sampling_Date" in X_test.columns:
        try:
            X_test["Sampling_Date"] = pd.to_datetime(X_test["Sampling_Date"])
            min_train_date = 0.0  # zero baseline
            X_test["Sampling_Date_ordinal"] = (
                X_test["Sampling_Date"].dt.dayofyear.fillna(0).astype(float)
            )
        except Exception:
            X_test["Sampling_Date_ordinal"] = 0.0

    # Ensure all expected columns are present
    for col in categorical_cols:
        if col not in X_test.columns:
            X_test[col] = "missing"
        X_test[col] = X_test[col].fillna("missing")
    if "Pre_GSHH_NDVI" not in X_test.columns:
        X_test["Pre_GSHH_NDVI"] = 0.0
    if "Height_Ave_cm" not in X_test.columns:
        X_test["Height_Ave_cm"] = 0.0

    # Select the same column order as training
    # Note: the preprocessor will handle missing columns gracefully
    X_test_processed = preprocessor.transform(X_test)

    # Predict all targets for each unique image
    preds = model.predict(X_test_processed)

    # Create a mapping from image to predicted values
    pred_df = pd.DataFrame(
        preds,
        index=unique_imgs,
        columns=model.estimators_[0].feature_names_in_ if False else None
    )
    # Note: we cannot easily retrieve target names from the model, so we
    # reconstruct them using the order of columns in `y` from training.

    # When training, the target order is fixed as y.columns; we store
    # this information inside the model for use here by injecting an
    # attribute.  However, if it is not present, we fall back to the
    # canonical order used in the competition.
    target_names = getattr(model, "target_names", [
        "Dry_Green_g",
        "Dry_Dead_g",
        "Dry_Clover_g",
        "GDM_g",
        "Dry_Total_g",
    ])
    pred_df.columns = target_names

    # Build the final submission DataFrame
    submission_rows = []
    for idx, row in df_test.iterrows():
        img_rel_path = row["image_path"]
        target_name = row["target_name"]
        sample_id = row["sample_id"]
        prediction = pred_df.loc[img_rel_path, target_name]
        submission_rows.append({"sample_id": sample_id, "target": prediction})
    submission_df = pd.DataFrame(submission_rows)
    return submission_df


def main():
    parser = argparse.ArgumentParser(description="CSIRO Image2Biomass MVP")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/kaggle/input/csiro-biomass",
        help="Directory containing train.csv, test.csv and image folders",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Fraction of training data reserved for validation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/kaggle/working/",
        help="Directory to save output files (e.g., submission.csv, trained models)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    train_images_dir = data_dir / "train"
    test_images_dir = data_dir / "test"

    # Load and prepare training data
    X, y, categorical_cols = load_and_prepare_data(train_csv, train_images_dir, data_dir)

    # Train the model
    model, preprocessor, val_score = train_model(
        X, y, categorical_cols, val_size=args.val_size
    )
    # Attach target names to the model for later use
    model.target_names = y.columns.tolist()

    # Predict on test set
    submission_df = predict_test(
        model, preprocessor, test_csv, test_images_dir, categorical_cols, data_dir
    )

    # Save submission file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    logging.info(f"Submission file written to {submission_path.resolve()}")


if __name__ == "__main__":
    main()