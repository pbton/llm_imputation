
# 🧠 Missing Data Imputation & Regression Analysis

## 📌 Overview

This project explores various imputation strategies for handling missing data in a structured dataset. The goal is to evaluate how each method affects the performance of a downstream regression model predicting a target variable (`rating`).

The analysis covers:
- Generation of synthetic missingness at 4 levels (5%, 10%, 15%, 20%)
- Imputation using multiple strategies
- Regression model training and performance comparison
- Evaluation using MAE, MSE, and RMSE metrics

---

## 📊 Dataset

The original dataset (`df`) includes features like:
- `title`
- `genres`
- `relevance`
- `tag`
- `timestamp`
- `rating` *(target variable)*

Missing data is introduced artificially to simulate real-world scenarios.

---

## 🛠️ Imputation Methods

For each missingness level, the following imputation methods are applied:

1. **Case-wise deletion** – removes rows with missing values  
2. **Zero imputation** – replaces missing values with 0  
3. **Mean imputation** – replaces missing values with the column mean  
4. **K-Nearest Neighbors (KNN)** – imputes values based on neighboring records  
5. **Multivariate Linear Regression** – predicts missing values using linear models  
6. **Large Language Model (LLM) / GPT-based Imputation** – fills missing values using contextual inferences from GPT-4o via OpenAI API  

---

## 📈 Evaluation Metrics

To assess regression performance on the imputed datasets, we calculate:

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**

Each imputed dataset is used to train a `LinearRegression` model, and performance metrics are computed on a held-out test set.

---

## 🔄 Workflow

1. Drop `rating` column and create baseline DataFrame  
2. Generate datasets with 5%, 10%, 15%, and 20% missing values  
3. Apply each imputation method to all datasets  
4. Fit regression model and evaluate  
5. Visualize missingness and output results in tables  

---

## 📦 Requirements

- Python 3.8+  
- Libraries:
  - `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`  
  - `openai` (for GPT-based imputation)  
  - `fancyimpute` (for KNN and regression imputation)  

Install dependencies:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib openai fancyimpute
```

---

## 🔐 LLM Imputation (Optional)

To enable GPT-based imputation:

1. Get your OpenAI API key: [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)  
2. Set it in your environment or code:

```python
openai.api_key = "sk-..."  # never commit this to version control!
```

The GPT model used is `gpt-4o`.

---

## 📁 Outputs

- 📈 Heatmaps of missingness for each level  
- 📋 Tables comparing regression performance across imputation methods and missing percentages  
- ✅ Insights into which imputation methods yield better model performance under various levels of missing data  

---

## 🚀 Future Work

- Compare with deep learning-based imputers  
- Use classification as the downstream task  
- Add support for real-world datasets with natural missingness  

---

## 🧠 Author Notes

This project is intended for research, education, and experimentation. GPT-based imputation is computationally and monetarily expensive—use with care on large datasets.
