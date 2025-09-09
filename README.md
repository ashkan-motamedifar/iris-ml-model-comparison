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

| Model | CV_F1 | Test_ACC | Test_Precision | Test_Recall | Test_F1 | Best_Params |
|-------|-------|----------|----------------|-------------|---------|-------------|
| DT    | 0.9522 | 0.9556 | 0.9608 | 0.9556 | 0.9554 | {'clf__criterion': 'gini', 'clf__max_depth': None, 'clf__min_samples_split': 5} |
| LogReg | 0.9809 | 0.9111 | 0.9155 | 0.9111 | 0.9107 | {'clf__C': 1, 'clf__penalty': 'l2', 'clf__solver': 'lbfgs'} |
| SVM   | 0.9714 | 0.9111 | 0.9155 | 0.9111 | 0.9107 | {'clf__C': 0.1, 'clf__gamma': 'scale', 'clf__kernel': 'linear'} |
| KNN   | 0.9713 | 0.9111 | 0.9298 | 0.9111 | 0.9095 | {'clf__n_neighbors': 5, 'clf__p': 1, 'clf__weights': 'uniform'} |
| RF    | 0.9522 | 0.8889 | 0.8981 | 0.8889 | 0.8878 | {'clf__max_depth': None, 'clf__min_samples_split': 2, 'clf__n_estimators': 100} |


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
