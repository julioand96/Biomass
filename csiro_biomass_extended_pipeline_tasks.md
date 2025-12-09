# 🔁 Extended Pipeline Tasks for CSIRO Biomass (For LLM)

This markdown provides extended instructions to update the pipeline and notebook so it satisfies core requirements in data science model development and evaluation for the CSIRO Image2Biomass competition.

---

## 📊 1. PCA or t-SNE Visualizations for Pattern Discovery

**Goal**: Reduce metadata dimensionality and visualize clusters or relationships using:
- PCA (Principal Component Analysis)
- t-SNE (t-distributed Stochastic Neighbor Embedding)

**Steps**:
- Extract only metadata columns from `train_df`, excluding target columns.
- Standardize numeric metadata.
- Apply PCA and t-SNE.
- Plot using `matplotlib` or `seaborn`, coloring points by species or one biomass component (e.g., GDM_g).
- Add explanation markdown on observed clusters or separability.

---

## ⚙️ 3. Wrap Preprocessing and Modeling in Sklearn Pipeline

**Goal**: Make preprocessing reproducible and portable across training and inference.

**Steps**:
- Use `ColumnTransformer` with `StandardScaler` for numerical and `OneHotEncoder` for categorical columns.
- Combine with model in a full `Pipeline`.
- Use `Pipeline.fit(X_train, y_train)` and `.predict()` directly.

---

## 🔎 4. Perform Hyperparameter Search

**Goal**: Find best model parameters using:
- `RandomizedSearchCV` or `GridSearchCV`
- Target metric: weighted average R² or average R²

**Steps**:
- Use the sklearn `Pipeline` as the estimator.
- Perform cross-validation using `GroupKFold` (grouped by image ID).
- Search key parameters: `n_estimators`, `max_depth`, `learning_rate`.
- Plot R² per fold.

---

## 📈 5. Evaluate and Report Results

**Goal**: Use reliable validation strategies and visualization.

**Steps**:
- Use `cross_val_score` or `cross_validate` with `scoring='r2'`
- Store per-fold scores and compute average ± std.
- Include confusion matrices or ROC **only** if treating this as classification (not typical here).
- Markdown summary with model choice justification and validation findings.

