# CSIRO Image2Biomass Project – Complete Context

This document summarises all the relevant information that has been
discussed about the **CSIRO – Image2Biomass Prediction** competition on
Kaggle.  It is designed as a reference resource for large language
models (LLMs) and people working on the project so that the context
and objectives remain clear at all times.  The sections are organised
hierarchically to make it easy to read and extract specific
segments.

## 🧭 Executive summary

* **Purpose of the competition:** to predict various components of dry
  biomass in pastures (green vegetation, dead vegetation, clover,
  “green dry matter” and total biomass) from top‑down images of
  pasture plots (70 cm × 30 cm) and a handful of field measurements.
  The ultimate goal is to provide a fast and scalable alternative to
  manual measurements, enabling farmers and scientists to better plan
  grazing and monitor soil health.
* **Nature of the competition:** this is a *research code competition*
  on Kaggle sponsored by CSIRO, MLA and Google.  Participants must
  publish notebooks with the code that reproduces their predictions.
  Performance is measured using a weighted coefficient of
  determination (R²) in which total biomass and green dry matter are
  given greater weight.
* **Data available:** the folder `train/` contains the training
  images; `train.csv` provides, for each combination of image and
  biomass component, the `sample_id`, `image_path`, sampling date,
  Australian state, dominant pasture species, average NDVI
  (`Pre_GSHH_NDVI`), average height (`Height_Ave_cm`), `target_name`
  and the biomass value (`target`)【260648875389173†screenshot】.
  `test.csv` contains only `sample_id`, `image_path` and
  `target_name`【22011555562683†screenshot】.
* **Key restrictions:** submissions **must** be made through Kaggle
  notebooks, without internet access and with a run‑time of less than
  9 hours【205478836664948†screenshot】.  External data and pre‑trained
  models are allowed as long as they are publicly available.
* **Current project strategy:** an MVP has been implemented that
  extracts simple features from the images (RGB means and standard
  deviations and colour histograms), combines metadata (NDVI, height,
  state, species and transformed date) and trains a multi‑output
  regressor based on XGBoost.  This model provides a robust and
  reproducible baseline on the CPU.

---

## 📖 Competition description

### General objective

The **CSIRO – Image2Biomass Prediction** competition challenges
participants to develop models capable of accurately estimating pasture
biomass from overhead images and field measurements.  The ability to
estimate biomass in real time helps farmers determine whether a
paddock has enough forage, when to let it rest and how to optimise
meat and milk production.

### Sponsors and organisation

The contest is organised by the Australian national science agency
(CSIRO) with support from Meat & Livestock Australia (MLA) and Google.
It forms part of Kaggle’s **research code competitions**: participants
not only submit predictions but also the code that makes them
reproducible.  The best solutions can be adopted in next‑generation
precision agriculture tools.

### Important rules

* **Submissions via Notebooks:** according to the competition
  requirements, solutions must be submitted as notebooks running on
  Kaggle’s infrastructure.  Notebooks cannot connect to the internet,
  must run in less than 9 hours and may use either CPUs or GPUs【205478836664948†screenshot】.
* **External data and models:** it is permitted to use pre‑trained
  weights and external datasets as long as they are publicly
  available【205478836664948†screenshot】.
* **Submission limits:** teams may make up to five submissions per day
  and must select their two best notebooks for final evaluation.
* **Fairness and reproducibility:** Kaggle may review the code to
  verify that results are reproducible and that time limits are
  respected.

---

## 🗂️ Dataset

### File structure

* **`train.csv`** – contains roughly 1 162 unique images.  For each
  image–biomass pair it includes the columns `sample_id`,
  `image_path`, `Sampling_Date`, `State`, `Species`,
  `Pre_GSHH_NDVI`, `Height_Ave_cm`, `target_name` and `target`
  (biomass value)【260648875389173†screenshot】.  Each image appears
  five times with different `target_name` and `target` values (one
  entry for each biomass component), allowing a vector of five
  outputs per image to be reconstructed.
* **`test.csv`** – includes `sample_id`, `image_path` and
  `target_name` for each image–objective pair【22011555562683†screenshot】.
  It does not contain metadata, so when preparing features for the
  model default or estimated values must be assigned to NDVI and
  height.
* **`sample_submission.csv`** – example submission format.  The
  `sample_id` column uniquely identifies each row (image + target),
  and `target` must contain the predictions.
* **Image folders** – `train/` and `test/` hold the photos in
  JPEG/PNG formats.  The paths indicated in `image_path` are
  relative to these folders.

### Available variables

1. **`image_path`** – relative path to the image in the
   corresponding folder.  Each photo represents a 70 cm × 30 cm
   rectangle of the pasture plot.  Framing and lighting vary with
   field conditions.
2. **`target_name`** – name of the biomass component.  It may be:
   * `Dry_Green_g` – dry weight of green vegetation (excluding
     clover)
   * `Dry_Dead_g` – dry biomass of dead material (litter)
   * `Dry_Clover_g` – dry biomass of clover
   * `GDM_g` – “green dry matter”, the sum of green vegetation and
     clover
   * `Dry_Total_g` – total dry biomass (green + dead + clover)
3. **`target`** – numerical value of the corresponding objective.  In
   `train.csv` it serves as the label; in `test.csv` it is the value
   to be predicted.
4. **`Sampling_Date`** – the date on which the sample was collected.
5. **`State`** – the Australian state where the sample was taken (e.g.
   Victoria, New South Wales).
6. **`Species`** – dominant pasture species (e.g. ryegrass, clover).
7. **`Pre_GSHH_NDVI`** – average NDVI measured over the plot.
8. **`Height_Ave_cm`** – mean height of the vegetation in centimetres.

### Data quality considerations

Some Kaggle discussions indicate that certain biomass annotations
contain errors or noise (for example, images without clover labelled
with high `Dry_Clover_g`).  It is also noted that average height has
little correlation with dead biomass; ratios such as `Dead/Total` and
`Dead/GDM` correlate better.  Therefore, it is important to inspect
the distributions and, when necessary, apply transformations (such as
working with fractions or logarithms) or implement robust losses
during training.

---

## 📏 Evaluation metric

Kaggle scores solutions using a **weighted coefficient of determination (R²)**.
For each biomass component the R² between the true values and
predictions is calculated and then averaged with greater weight for
total biomass and for green dry matter.  The exact weight of each
variable is specified in the Kaggle rules (for example,
`Dry_Total_g` receives around 50 % of the total weight).  This
approach forces models to prioritise good fits to total biomass, while
allowing higher error in `Dry_Clover_g` to have less impact on the
score.

In practice, when training a baseline model you can use a standard
loss (MSE or Huber), but it is advisable to measure performance on
validation using the weighted R² to align improvements with the
competition metric.

---

## 🛠️ Strategy and baseline

### Feature extraction

**Images.**  For an MVP without deep‑learning libraries, simple
numeric features are extracted:

1. **Basic statistics** – mean and standard deviation of the red,
   green and blue channels.  These capture global brightness and
   colour.
2. **Colour histograms** – normalised histograms with 16 bins for
   each channel.  They compactly represent the distribution of
   colours.

This vectorisation produces a 54‑dimensional vector per image.  In
environments with PyTorch/TensorFlow these vectors can be replaced by
embeddings extracted from pre‑trained networks (e.g. EfficientNet or
ResNet), which considerably improve predictive power.

**Metadata.**  `Pre_GSHH_NDVI` and `Height_Ave_cm` are used directly.
The date is transformed into a numerical value (days since the
earliest date), and the categorical variables `State` and `Species`
are converted to “one‑hot” vectors.  All numeric variables are
standardised so they have comparable variance.

### Baseline model

To handle the five outputs simultaneously a
**`MultiOutputRegressor`** from scikit‑learn is used with an
`XGBRegressor` as the base estimator.  XGBoost is robust to mixed
data structures and does not require a GPU, which makes it easy to
run in constrained environments.

Training steps:

1. **Pivot the data** so that each image becomes a row with five
   objective columns.
2. **Extract features** from the images and metadata.
3. **Split** into training and validation sets (e.g. 80 %/20 %).
4. **Preprocess** with a `ColumnTransformer` (one‑hot for
   categoricals and normalisation for numerics).  Image features are
   passed through unscaled.
5. **Train the model** (tune number of trees, depth and learning
   rate).  Evaluate with MSE but also compute weighted R² on
   validation to guide tuning.
6. **Infer on the test set** by generating predictions for each
   image.  Since `test.csv` lacks metadata, use default or estimated
   values; match predictions to each `sample_id` and save
   `submission.csv`.

### Potential improvements

* **Image embeddings via CNNs/ViTs:** use embeddings from
  pre‑trained models (EfficientNet, ViT, DINO) to capture complex
  spatial patterns.  These embeddings can be combined with metadata
  and fed into a simpler regressor (XGBoost, Ridge, NN).  Requires
  running PyTorch or TensorFlow on Kaggle.
* **Multi‑task or two‑phase models:** predict total biomass first and
  then the dead biomass fraction, as suggested by the community.
  This may improve the stability of `Dry_Dead_g` predictions.
* **Robust losses:** use Huber or Tukey losses to reduce the impact
  of outliers and label errors.
* **Stratified cross‑validation:** split images into folds that
  respect the distribution of states/seasons and total biomass,
  avoiding a fold containing all samples from a particular state.

---

## 🤖 Rules for the LLM

When interacting with a language model based on this project, follow
these guidelines to ensure coherence and relevance:

1. **Focus on the objective:** remember that the goal is to estimate
   pasture biomass; avoid deviating into unrelated topics.
2. **Respect Kaggle restrictions:** do not propose solutions that
   require internet during scoring or exceed 9 hours of execution.
   Remember that submissions must be notebooks【205478836664948†screenshot】.
3. **Be reproducible:** any code or method suggested must run with
   the tools available in the Kaggle environment.
4. **State assumptions:** if information is missing (e.g. library
   versions), state the reasonable assumptions you make.
5. **Output format:** when asking for code, always provide clear
   pseudocode and then explain what each part does; whenever
   possible, include references or comments to guide the reader.

---

## 🧪 Backlog / Next steps

* **Deep exploratory analysis:** explore distributions of the targets
  and metadata, detect outliers and labelling errors.  Use plots and
  descriptive statistics to guide new transformations.
* **Tests with image embeddings:** implement a pipeline in Kaggle
  that loads a pre‑trained model (EfficientNet, ViT or DINO),
  obtains a feature vector per image and trains a simple regressor.
  Compare its performance with the baseline.
* **Hyperparameter tuning for XGBoost:** explore more trees,
  regularisation, sampling methods and Huber loss to improve fit.
* **Ensembles and averaging:** combine several models (trees,
  neural networks, linear regressors) to obtain more robust
  predictions.
* **Two‑step prediction:** study the community suggestion to
  predict total biomass and then derive dead biomass as a fraction;
  assess whether it improves the weighted R².