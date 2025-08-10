# 🌾 Oats Crop Classification on Imbalanced Dataset

This project applies various classification models in R to detect whether **Oats** (class = 1) or **Other** (class = 0) are being grown on farmland using optical and radar-based remote sensing data. The dataset is derived from a modified version of the Winnipeg Crop Mapping Dataset, hosted by the UCI Machine Learning Archive.

## 📁 Dataset

- **Source**: [UCI Machine Learning Archive](https://archive.ics.uci.edu/dataset/525/crop+mapping+using+fused+optical+radar+data+set)
- **Features**: 20 randomly selected variables (`A01`–`A30`), plus a binary target `Class`
- **Observations**: 5000
- **Class Distribution**:
  - Oats: 14.2%
  - Other: 85.8%

## 🧰 Technologies Used

- **Language**: R
- **IDE**: RStudio
- **Packages**: `dplyr`, `tree`, `e1071`, `ROCR`, `randomForest`, `adabag`, `rpart`, `smotefamily`, `neuralnet`, `xgboost`

## ⚙️ Project Workflow

1. **Exploratory Data Analysis**
   - Summary statistics, standard deviations, and correlation analysis
   - Identification of outliers and multicollinearity (e.g., `A16` `A19`, and `A20`)
2. **Preprocessing**
   - Log transformation of skewed features (`A26`)
   - Removal of highly correlated variables
   - SMOTE oversampling for class balance
   - Feature scaling for neural network models
3. **Model Training**
   - Trained and evaluated:
     - Decision Tree
     - Naive Bayes
     - Bagging
     - Boosting
     - Random Forest (RF)
     - RF with hyperparameter tuning on SMOTE-balanced data
     - Artificial Neural Network (ANN) on SMOTE-balanced data
     - XGBoost
4. **Evaluation Metrics**
   - Confusion Matrix
   - Accuracy, Precision, Recall, F1-Score
   - ROC Curve and AUC

## 📊 Model Performance Summary

| Model                 | Accuracy | Precision | Recall | F1-Score | AUC   |
|-----------------------|----------|-----------|--------|----------|-------|
| Decision Tree         | 86.6%    | 0.533     | 0.040  | 0.074    | 0.689 |
| Naive Bayes           | 80.2%    | 0.246     | 0.228  | 0.237    | 0.699 |
| Bagging               | 86.7%    | 0.667     | 0.030  | 0.057    | 0.771 |
| Boosting              | 86.6%    | 0.505     | 0.238  | 0.323    | 0.734 |
| Random Forest         | 87.1%    | 0.714     | 0.074  | 0.135    | 0.777 |
| Tuned RF + SMOTE      | 80.1%    | 0.420     | 0.532  | 0.470    | 0.720 |
| ANN + SMOTE           | 66.5%    | 0.247     | 0.723  | 0.368    | 0.728 |
| XGBoost               | 77.4%    | 0.321     | 0.609  | 0.421    | 0.775 |

## 🔑 Outcomes

- Class imbalance had a major impact on model performance, particularly on recall for the minority class.
- F1-score was a more reliable evaluation metric than accuracy, which was inflated by the dominance of the majority class.
- The attributes `A26`, `A02`, `A06`, `A07`, `A08`, and `A14` were consistently identified as the most important across the basic models (Decision Tree, Naive Bayes, Bagging, Boosting, Random Forest). These features were used in subsequent models to reduce overfitting and improve generalization.
- Tuned RF with SMOTE, ANN with SMOTE, and XGBoost achieved higher recall and F1-scores than other models, with a slight trade-off in accuracy and precision due to more aggressive prediction of the minority class (Oats).
- The tuned Random Forest model performed best overall, offering the most balanced trade-off between precision and recall, and thus achieving the highest F1-score after SMOTE balancing and hyperparameter tuning.

### 🚀 How to Run

1. Clone the repository or download the ZIP file from GitHub.
2. Open the project folder in RStudio.
3. Run the R script (`oats_classification.r`) inside the RStudio environment.

## 👤 Author

Developed by Juan Nathan.





