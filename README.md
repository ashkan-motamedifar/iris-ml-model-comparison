# Iris ML Model Comparison

## Overview

This repository presents a comparative study of classical machine learning classifiers on the **Iris dataset**.
The objective is to evaluate and benchmark multiple models using a rigorous pipeline with **hyperparameter tuning (GridSearchCV)** and **cross-validation (StratifiedKFold)**.

The project follows an **academic approach**: preprocessing, model selection, parameter optimization, and evaluation on an independent test set.

---

## Dataset

* **Source**: Iris dataset (`sklearn.datasets.load_iris`)
* **Samples**: 150 instances
* **Features**: 4 numerical features (sepal length/width, petal length/width)
* **Classes**: 3 species of Iris (*Setosa, Versicolor, Virginica*)

---

## Methods

### 1. Preprocessing

* Data split into **70% train / 30% test** (stratified).
* Standardization applied to all pipelines (trees do not require scaling but kept for uniformity).

### 2. Models Compared

* Logistic Regression
* Support Vector Machine (SVC)
* k-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest

### 3. Hyperparameter Tuning

* Performed using **GridSearchCV** with **5-fold Stratified K-Fold** cross-validation.
* Optimization target: **macro-averaged F1 score**.

### 4. Evaluation Metrics

* Accuracy
* Precision (macro)
* Recall (macro)
* F1-Score (macro)

---

## Results

Each model is trained and tuned on the training set, then evaluated on the test set.
Results are ranked by **Test F1-score**.

Example output (abridged):

| Model | CV F1  | Test ACC | Test Precision | Test Recall | Test F1 | Best Params                        |
| ----- | ------ | -------- | -------------- | ----------- | ------- | ---------------------------------- |
| SVM   | 0.9733 | 0.9778   | 0.978          | 0.978       | 0.978   | {kernel: rbf, C: 10, gamma: scale} |
| RF    | 0.9667 | 0.9556   | 0.956          | 0.956       | 0.956   | {...}                              |
| …     | …      | …        | …              | …           | …       | …                                  |

---

## Reproducibility

### Requirements

* Python 3.8+
* `scikit-learn`
* `pandas`

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook iris_model-comparison_grid-search.ipynb
```

---

## Conclusion

This study shows that classical ML algorithms, when properly tuned, can achieve high performance on the Iris dataset.

* **SVM and Random Forest** typically achieve the best balance of precision, recall, and F1.
* The pipeline demonstrates how systematic **cross-validation and hyperparameter optimization** are essential for fair model comparison.
