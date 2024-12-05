
# TreeBasedFeatureImportanceAnalyzer: A SHAP-Based Feature Importance Framework

This project evolved from a naive implementation of SHAP-based feature importance to a robust and flexible solution capable of handling **regression**, **ranking**, and **classification** problems.

## Background

**SHAP (SHapley Additive exPlanations)** provides a unified framework for explaining model predictions. While its application is straightforward for regression, handling **log-odds** and **probabilities** in classification and understanding its application in **ranking tasks** can be challenging.

This document highlights:
1. Mistakes in the naive implementation.
2. Enhancements made to handle various use cases effectively.

---

## Mistakes in the Naive Approach

### 1. **Assuming Log-Odds Are Intuitive**
- **Mistake**: The naive approach directly aggregated SHAP values in **log-odds space** using a mean absolute strategy:
  \[
  \text{Mean Absolute SHAP Value} = \frac{1}{n} \sum_{i=1}^{n} |\text{SHAP Value}[i, f, c]|
  \]
- **Why It’s a Problem**:
  - Log-odds are non-linear and unintuitive for most end users.
  - A small change in log-odds can result in a large change in probabilities near the extremes (\(P \to 0\) or \(P \to 1\)).
  - Averaging in log-odds space does not directly translate to how features affect probabilities.

---

### 2. **Ignoring Ranking-Specific Scenarios**
- **Mistake**: The naive implementation treated ranking problems as regression without emphasizing the relative nature of ranking scores.
- **Why It’s a Problem**:
  - In ranking, model outputs are not probabilities but scores used to order items.
  - Direct interpretation of SHAP values without considering ranking-specific insights (e.g., relative importance) can be misleading.

---

### 3. **Verification Overlooked Prediction Spaces**
- **Mistake**: The naive approach attempted to directly compare SHAP sums in log-odds space to probabilities, leading to incorrect assertions.
- **Why It’s a Problem**:
  - SHAP sums in log-odds space must first be transformed into probabilities using the sigmoid function:
    \[
    P = \frac{1}{1 + e^{-\text{log-odds}}}
    \]

### 4. **Treating Ranking problems as Classification ones**
- **Mistake**: The naive approach attempted to directly calculate mean without acounting for the shape of the result. 
- **Why It’s a Problem**:
  - 3D array result for classificaiton problems, in order to account for each class 3d-dimensions
  - 2D array result for Regression or ranking.
  - Mean wasn't calculated correctly.

---

## Enhancements in `TreeBasedFeatureImportanceAnalyzer`

### **1. Handling Log-Odds and Probabilities**
- SHAP values are computed in log-odds space for classification by default.
- For probabilities:
  - Convert reconstructed log-odds sums to probabilities using:
    \[
    P = \frac{1}{1 + e^{-\left(\text{Base Value}[c] + \sum \text{SHAP Value}[f, c]\right)}}
    \]
- Added the ability to verify predictions in both log-odds and probability spaces.

---

### **2. Support for Ranking**
- Treated ranking tasks as distinct from regression or classification.
- Adapted SHAP value interpretation for ranking scores:
  - Scores are relative and not bound to [0, 1] like probabilities.
  - Aggregated SHAP values directly in raw score space:
    \[
    \text{Mean Absolute SHAP Value} = \frac{1}{n} \sum_{i=1}^{n} |\text{SHAP Value}[i, f]|
    \]

---

### **3. Flexible Aggregation Strategies**
- Supports both `mean_abs` and `std` as aggregation strategies:
  - **`mean_abs`**: Highlights globally important features by averaging the absolute magnitude of SHAP values.
  - **`std`**: Identifies volatile features with high variability across samples.

---

## Mathematical Foundations

### **Log-Odds Space**
- Log-odds \( z \) for class \( c \):
  \[
  z = \text{Base Value}[c] + \sum_{f} \text{SHAP Value}[i, f, c]
  \]
- Converts to probabilities using the sigmoid function:
  \[
  P_c = \frac{1}{1 + e^{-z}}
  \]

### **Probability Space**
- Contribution of a feature \( f \) to the probability of class \( c \):
  \[
  \Delta P_{i, f, c} = \left| \frac{1}{1 + e^{-\left(\text{Base Value}[c] + \text{SHAP Value}[i, f, c]\right)}} - \frac{1}{1 + e^{-\text{Base Value}[c]}} \right|
  \]
- Mean absolute SHAP values in probability space:
  \[
  \text{Mean Absolute SHAP Value (Probability)} = \frac{1}{n} \sum_{i=1}^{n} \Delta P_{i, f, c}
  \]

---

## Features of the Enhanced Class

### **Key Methods**

1. **`get_feature_importance`**:
   - Computes feature importance using SHAP values.
   - Supports regression, ranking, and classification.

2. **`verify_predictions`**:
   - Verifies SHAP values align with model predictions.
   - Converts log-odds sums to probabilities for classification.

3. **`export_to_csv`**:
   - Exports mean absolute SHAP values and standard deviations to a CSV file for easy analysis.

---

## Example Usage

### **For Regression**
```python
analyzer = TreeBasedFeatureImportanceAnalyzer(model, X)
print(analyzer.get_feature_importance(aggregation_strategy="mean_abs"))
analyzer.verify_predictions()
```

### **For Ranking**
```python
ranker = lgb.LGBMRanker()
analyzer = TreeBasedFeatureImportanceAnalyzer(ranker, X)
print(analyzer.get_feature_importance(aggregation_strategy="mean_abs"))
analyzer.verify_predictions()
```

### **For Classification**
```python
analyzer = TreeBasedFeatureImportanceAnalyzer(classifier, X)
print(analyzer.get_feature_importance(class_index=1, aggregation_strategy="mean_abs"))
analyzer.verify_predictions(class_index=1)
```

### **Export to CSV**
```python
analyzer.export_to_csv("feature_importance.csv", feature_names=["Feature1", "Feature2", ..., "FeatureN"], class_index=1)
```

---
