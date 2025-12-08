# Image2Biomass Prediction – MVP Baseline

This repository contains a **minimum‑viable product (MVP)** pipeline for the
CSIRO Image2Biomass prediction challenge on Kaggle.  The goal of the
competition is to predict various components of dry pasture biomass from
overhead images, along with a handful of simple field measurements.

This baseline is intended for participants who have experience with
statistics and exploratory data analysis but are new to building and
deploying machine‑learning models.  The code is extensively documented
and explains what each part of the pipeline is doing.

## Features

* **Data loading and reshaping** – reads the provided `train.csv` and
  restructures the data so that each unique image has one row with all
  five target variables (`Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`,
  `GDM_g`, `Dry_Total_g`).
* **Simple image feature extraction** – computes basic statistics and
  colour histograms from each image using the Pillow and NumPy
  libraries.  These lightweight features act as surrogates for a
  convolutional neural network in environments where deep learning
  frameworks are unavailable.
* **Incorporation of meta data** – numerical variables such as NDVI and
  average height are used directly, while categorical variables
  (`State` and `Species`) are one‑hot encoded.  Dates are converted to
  numeric offsets from the earliest sampling date.
* **Baseline model** – uses a gradient boosted decision tree (via
  XGBoost) wrapped in scikit‑learn’s `MultiOutputRegressor` to handle
  the five output variables simultaneously.  This model works out of
  the box on the CPU and does not require a GPU.
* **Weighted R² evaluation** – implements the weighted coefficient of
  determination used by Kaggle to score submissions.  Even if you
  choose to optimise for a different loss function during training, the
  weighted R² metric gives you feedback aligned with the competition
  leaderboard.
* **Training and inference scripts** – helper functions handle data
  splitting, model training, evaluation and prediction on the test
  set.  Predicted values are written out in the required format for
  submission (`submission.csv`).

## How to use this baseline

1.  Clone the repository or copy the contents into a Kaggle Notebook.
2.  Place the competition data in the expected directory structure:
    ```
    data/
      ├── train.csv
      ├── test.csv
      ├── train/
      │     ├── ... (image files)
      └── test/
            ├── ... (image files)
    ```
3.  Run `python mvp.py` from the project root.  The script will
    automatically load the data, train the model using an 80/20
    validation split, evaluate the model, and produce a
    `submission.csv` file in the project directory.

You are encouraged to explore the code and modify it to experiment with
alternative feature extraction techniques, different machine‑learning
models, cross‑validation strategies, or the inclusion of additional
data.  As you iterate, ensure that the notebook you submit on Kaggle
respects the competition requirements: submissions must be made through
Kaggle notebooks, have a run time of 9 hours or less, and must not
access the internet during scoring【205478836664948†screenshot】.

## A note on deep learning

This MVP deliberately avoids using deep learning libraries because
PyTorch and TensorFlow are not available in this environment.  Once you
move into the Kaggle environment you can consider replacing the
hand‑crafted image features with those extracted by a pre‑trained
convolutional neural network (e.g. EfficientNet or ResNet).  To do so
you would load the pre‑trained model using PyTorch or TensorFlow,
remove the final classification layer, and treat the activations of the
penultimate layer as a feature vector.  These deep features can then be
fed into your regression model in the same way that the simple
histogram statistics are used here.
