# Machine Learning Assignment 2: Hotel Booking Demand Classification

**Student Name / BITS ID:** 2026ct05075  
**Course:** M.Tech (AIML / DSE) - Machine Learning  
**Submission Deadline:** 27-Nov-2026

---

## 1. Problem Statement
The objective of this project is to predict hotel booking cancellations using historical data from the **Hotel Booking Demand** dataset. Predicting cancellations allows hotels to optimize room occupancy, manage pricing strategies, and minimize lost revenue due to empty rooms.

## 2. Dataset Description
- **Dataset Source:** Hotel Booking Demand Dataset (Kaggle)
- **Number of Instances:** 119,390
- **Number of Original Features:** 32 (including categorical, numerical, and date variables)
- **Target Variable:** `is_canceled` (Binary: `1` if the booking was canceled, `0` otherwise)
- **Class Distribution:**
  - **Class 0 (Not Canceled):** 75,166 (62.95%)
  - **Class 1 (Canceled):** 44,224 (37.05%)
  - *Observation:* There is a moderate class imbalance present in the dataset.

### Feature Preprocessing Steps:
- **Target Leakage / Irrelevant Columns Dropped:** `reservation_status` (directly mirrors the target), `reservation_status_date` (temporal leak), and `company` (over 94% missing values).
- **Missing Value Imputation:** Numerical features (such as `children` and `agent`) were imputed using the **median**. Categorical features (such as `country`) were imputed with the constant value `'Unknown'` to avoid dropping critical samples.
- **Categorical Encoding:** One-Hot Encoding was applied to non-numerical categorical features (e.g. `hotel`, `meal`, `market_segment`, etc.), resulting in 252 features post-encoding.
- **Feature Scaling:** StandardScaler was applied to all numerical columns.
- **Data Splitting:** 80:20 stratified train-test split (using a fixed random seed of `42`) to maintain the class distribution ratio in both sets.

## 3. GitHub Repository Link
- **GitHub Repository:** https://github.com/2026ct05075/2026CT05075-ML-Assignment-2

## 4. Models Used & Comparative Performance
We trained and evaluated 5 ML classification models using the same preprocessed datasets:
1. **Logistic Regression**
2. **Decision Tree**
3. **K-Nearest Neighbors (KNN)** (Trained on a 15k subset for speed optimization)
4. **Gaussian Naive Bayes**
5. **Random Forest Ensemble**

### Comparative Performance Table
| Model | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8197 | 0.8963 | 0.8132 | 0.6663 | 0.7324 | 0.6052 |
| **Decision Tree** | 0.8589 | 0.8523 | 0.8052 | 0.8167 | 0.8109 | 0.6984 |
| **K-Nearest Neighbors** | 0.7944 | 0.8562 | 0.7268 | 0.7131 | 0.7199 | 0.5576 |
| **Naive Bayes** | 0.4628 | 0.5715 | 0.4064 | 0.9780 | 0.5742 | 0.2136 |
| **Random Forest** | **0.8938** | **0.9581** | **0.8940** | **0.8093** | **0.8495** | **0.7700** |

## 5. Key Observations
- **Random Forest** achieved the best overall performance, leading across almost all metrics with an **Accuracy of 89.38%**, an **AUC of 95.81%**, and an **MCC of 0.7700**.
- **Gaussian Naive Bayes** displayed a near-perfect **Recall of 97.80%**, but suffered from very poor **Precision (40.64%)** and overall **Accuracy (46.28%)**, indicating a high rate of false positives.
- **Decision Tree** performed moderately well (**Accuracy: 85.89%**), showing balanced precision and recall, but is more prone to overfitting compared to Random Forest.
- **Logistic Regression** and **KNN** provided decent, stable baselines but were outperformed by the ensemble architecture of Random Forest.

--- 
*Note: Model comparison visualization is stored as `model_comparison_chart.png` in the repository.*
