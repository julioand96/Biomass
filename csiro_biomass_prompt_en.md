# Mini‑prompt: summarised context of the CSIRO – Image2Biomass project

This brief summary contains the essential project information to feed
as initial context to a language model.  It focuses on the objective,
data, competition restrictions and baseline strategy.

## 📌 Key points

* **Objective:** to predict dry biomass (green, dead, clover, GDM and
  total) of pastures from top‑down images and sampling metadata.  This
  helps farmers and scientists manage grazing more effectively.
* **Dataset:** `train.csv` contains, for each image, the date,
  state, species, NDVI, height and five rows with different biomass
  components and their value【260648875389173†screenshot】.
  `test.csv` includes `sample_id`, `image_path` and
  `target_name`【22011555562683†screenshot】; predictions must be
  returned in a `submission.csv`.
* **Metric:** performance is measured using a weighted R² that gives
  more weight to total biomass and green matter; the weights are
  defined in the Kaggle rules.
* **Competition restrictions:** submissions must be Kaggle notebooks
  without internet access and executed in ≤ 9 h【205478836664948†screenshot】.
  Pre‑trained models and external data are allowed as long as they are
  freely accessible【205478836664948†screenshot】.
* **Current baseline:** an MVP has been implemented that extracts
  simple image features (RGB means and histograms) and combines NDVI,
  height, state, species and transformed date.  A
  `MultiOutputRegressor` with XGBoost is trained to predict the five
  outputs simultaneously, evaluating with weighted R².  This
  approach does not require deep‑learning libraries and serves as a
  reproducible starting point.

## 🧠 How to use this context

* When formulating questions or requesting code, remember that the
  solution must comply with the competition rules (notebook‑only,
  < 9 h run time).  Do not propose solutions that require internet
  during scoring.
* If information is missing (e.g. library versions), explain the
  reasonable assumptions you adopt.
* To develop improvements, suggest exploring pre‑trained image
  embeddings (EfficientNet, ViT), multi‑task models and robust
  losses.