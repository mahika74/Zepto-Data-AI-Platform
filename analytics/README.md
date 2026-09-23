# Module 2 - Analytics & Machine Learning

## Overview

This module implements an end-to-end analytics and machine learning workflow using the classic Titanic dataset.

The workflow covers:

**Data Understanding → Data Cleaning → Exploratory Data Analysis → Visualization → Preprocessing → Classification → Class Imbalance → Hyperparameter Tuning → Regression → Model Evaluation → Model Persistence**

---

## Dataset

The classic Titanic dataset is loaded using Seaborn:

```python
import seaborn as sns

df = sns.load_dataset("titanic")
```

The raw dataset is immediately saved as:

`titanic.csv`

This file serves as an offline fallback so that the analysis can be reproduced without repeatedly downloading the dataset.

---

## Exploratory Data Analysis

The EDA workflow includes:

- Dataset structure and summary statistics
- Dataset shape and data types
- Missing-value percentage analysis
- Missing-value treatment
- Age and Fare distribution analysis
- Box plots and IQR-based outlier analysis
- Fare mean, median, and mode
- Skewness analysis
- Survival analysis using Boolean masking
- Survival rates by sex
- Survival rates by passenger class
- Survival analysis using sex and passenger class together
- Correlation matrix and heatmap
- Multiple survival-focused visualizations
- Exploratory z-score standardization of Age and Fare

The exploratory standardization is performed for analysis and does not feed the modeling pipeline.

---

## Data Cleaning and Missing Values

Missing values are analyzed based on their percentage in each column.

The treatment strategy follows the required thresholds:

- Less than 5% missing: rows are dropped
- 5% to 30% missing: values are imputed
- Very high missingness: the feature is explicitly evaluated and either removed or handled appropriately

The exact missing-value percentages and treatment decisions are documented in the EDA notebook.

---

## Survival and Correlation Analysis

Boolean masking is used to investigate survival patterns.

The analysis includes survival rates by:

- Sex
- Passenger class
- Sex and passenger class together

A correlation matrix is created using:

- `survived`
- `pclass`
- `age`
- `sibsp`
- `parch`
- `fare`

A correlation heatmap is also generated.

The strongest absolute off-diagonal correlations are identified and interpreted in the EDA notebook.

---

## Data Visualization

Multiple visualizations are created to develop a coherent understanding of Titanic survival patterns.

The visualizations analyze relationships involving:

- Survival
- Sex
- Passenger class
- Age
- Fare

Each visualization is accompanied by an interpretation of the observed pattern.

---

## Exploratory Standardization

Z-score standardization is performed on:

- Age
- Fare

The mean and standard deviation are shown before and after standardization.

This standardization is exploratory only and is not used as input to the machine learning models.

---

# Machine Learning

## Train/Test Split

The cleaned dataset is divided into training and testing sets using a stratified train/test split.

Stratification is used to preserve the distribution of the target variable, `survived`, across the training and testing datasets.

---

## Preprocessing

Preprocessing is fitted only on the training data to prevent data leakage.

The preprocessing pipeline handles:

- Missing numerical values
- Missing categorical values
- Numerical feature scaling using `StandardScaler`
- Categorical feature encoding

`Pipeline` and `ColumnTransformer` are used to combine preprocessing and machine learning steps.

---

## Classification Models

Three classification algorithms are trained using the same train/test split:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The Decision Tree is visualized using `plot_tree()` with feature names and class names.

---

## Classification Evaluation

The classification models are evaluated using:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- ROC-AUC

The evaluation results are presented together for comparison.

---

## Class Imbalance Analysis

The target class distribution is examined.

Different approaches are compared:

1. Baseline classifier
2. `class_weight="balanced"`
3. SMOTE applied only to the training data

Precision, Recall, and F1 Score are compared to understand the effect of class-balancing techniques.

---

## Random Forest Hyperparameter Tuning

`GridSearchCV` is used to tune the Random Forest classifier.

The following parameters are explored:

- `n_estimators`
- `max_depth`
- `max_features`

The Random Forest estimator uses Out-of-Bag evaluation:

```python
RandomForestClassifier(
    oob_score=True,
    ...
)
```

The best hyperparameters and OOB score are reported in the modeling notebook.

---

# Regression

## Fare Regression

A regression model is developed to predict passenger fare using the available passenger features.

The regression model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R²
- Adjusted R²

A residual plot is generated to examine prediction errors and assess potential heteroscedasticity.

---

# Model Comparison

## Classification Comparison

The classification models are compared using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The comparison evaluates the models across multiple performance measures.

## Regression Evaluation

The regression model is evaluated separately using:

- MAE
- RMSE
- R²
- Adjusted R²

The final notebook provides a written comparison based on the observed metric values.

---

# Model Persistence

The complete fitted preprocessing and classification pipeline is saved using `joblib`.

The saved model is:

`best_pipeline.joblib`

The saved pipeline is reloaded and used to generate predictions from raw input data.

This demonstrates that the complete preprocessing and prediction workflow can be reused after training.

---

# Project Structure

```text
analytics/
├── 01_eda.ipynb
├── 02_modeling.ipynb
├── titanic.csv
├── best_pipeline.joblib
└── README.md
```

### `01_eda.ipynb`

Contains the exploratory analytics workflow:

- Dataset understanding
- Missing-value analysis
- Missing-value treatment
- Distribution analysis
- Outlier analysis
- Survival analysis
- Correlation analysis
- Data visualization
- Exploratory standardization

### `02_modeling.ipynb`

Contains the machine learning workflow:

- Stratified train/test split
- Training-only preprocessing
- Logistic Regression
- Decision Tree
- Random Forest
- Classification evaluation
- Class imbalance analysis
- SMOTE
- Random Forest GridSearchCV
- OOB score
- Fare regression
- Regression evaluation
- Residual analysis
- Model comparison
- Model persistence
- Model reload and prediction

### `titanic.csv`

Raw Titanic dataset saved as an offline fallback after the initial dataset load.

### `best_pipeline.joblib`

Serialized fitted machine learning pipeline containing the preprocessing and classification workflow.

---

# How to Run

## Step 1 - Exploratory Data Analysis

Open:

`01_eda.ipynb`

Run the notebook from top to bottom.

This performs dataset inspection, cleaning, exploratory analysis, visualizations, correlation analysis, and exploratory standardization.

## Step 2 - Machine Learning

Open:

`02_modeling.ipynb`

Run the notebook from top to bottom.

This performs:

- Train/test splitting
- Preprocessing
- Classification
- Model evaluation
- Class imbalance analysis
- Hyperparameter tuning
- Regression
- Final model comparison
- Model persistence

---

# Key Design Decisions

- The raw Titanic dataset is saved as `titanic.csv` for offline reproducibility.
- A stratified train/test split is used to preserve target-class proportions.
- Preprocessing is fitted only on training data to prevent data leakage.
- The same train/test split is used for the classification models.
- SMOTE is applied only to the training data.
- `Pipeline` and `ColumnTransformer` are used to organize preprocessing and modeling.
- Random Forest hyperparameters are optimized using `GridSearchCV`.
- Out-of-Bag evaluation is enabled for the Random Forest.
- The complete fitted pipeline is saved using `joblib`.
- The saved pipeline is reloaded and tested with raw input data.

---

# Reproducibility

The analytics workflow is organized into two notebooks:

- `01_eda.ipynb`
- `02_modeling.ipynb`

The raw dataset and trained pipeline are stored locally in the `analytics` directory so that the analysis and prediction workflow can be reproduced without relying on the original interactive session.
