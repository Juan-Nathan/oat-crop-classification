# 🌾 Oat Crop Classification using Remote Sensing Data

This project applies various classification models in R to predict whether **oat crops** are being grown on farmland using optical and radar-based remote sensing data. The dataset is derived from a modified version of the Winnipeg Crop Mapping Dataset, hosted by the UCI Machine Learning Archive.

## Dataset

- `WinnData.csv`
- **Source**: [UCI Machine Learning Archive](https://archive.ics.uci.edu/dataset/525/crop+mapping+using+fused+optical+radar+data+set)
- **Personal Subset** (randomly sampled with seed = 33270961):
  - **Observations**: 5,000
  - **Features**: 20 variables from `A01` to `A30`
  - **Target**: Binary variable `Class`
  - **Class Distribution**:
    - Oats: 14%
    - Other: 86%

## Technologies Used

- **Language**: R
- **IDE**: RStudio
- **Packages**: `tree`, `e1071`, `ROCR`, `randomForest`, `adabag`, `rpart`, `smotefamily`, `neuralnet`, `xgboost`

## Project Workflow

### 1. Exploratory Data Analysis
- Summary statistics, standard deviations, and correlation analysis
- Detection of outliers and assessment of multicollinearity

### 2. Preprocessing
- Log transformation of `A26` to handle outliers
- Removal of highly correlated variables (`A16` and `A19`)
- Synthetic Minority Oversampling Technique (SMOTE) for class balance
- Feature scaling to ensure stable convergence (when applicable)

### 3. Model Training
- Decision Tree
- Naive Bayes
- Bagged Trees
- AdaBoost
- Random Forest (RF)
- Hyperparameter-tuned RF on SMOTE-balanced data
- Artificial Neural Network (ANN) on SMOTE-balanced data
- XGBoost

### 4. Evaluation
- Accuracy, Precision, Recall, F1-Score, and AUC

## Model Performance Summary

| Model                 | Accuracy | Precision | Recall | F1-Score |  AUC  |
|-----------------------|----------|-----------|--------|----------|-------|
| Decision Tree         | 87%      | 53%       | 4%     | 7%       | 0.69  |
| Naive Bayes           | 80%      | 25%       | 23%    | 24%      | 0.70  |
| Bagged Trees          | 87%      | 67%       | 3%     | 6%       | 0.77  |
| AdaBoost              | 87%      | 51%       | 24%    | 32%      | 0.73  |
| RF                    | 87%      | 71%       | 7%     | 14%      | 0.78  |
| Tuned RF with SMOTE   | 80%      | 42%       | 53%    | 47%      | 0.72  |
| ANN with SMOTE        | 67%      | 25%       | 72%    | 37%      | 0.73  |
| XGBoost               | 77%      | 32%       | 61%    | 42%      | 0.78  |

## Results

- **Class imbalance** had a major negative impact on model performance, particularly on recall for the minority class.
- Accuracy was **not** a reliable evaluation metric, as it was inflated by the majority class.
- The attributes `A26`, `A02`, `A06`, `A07`, `A08`, and `A14` were consistently identified as the most significant across the "basic" models (Decision Tree, Naive Bayes, Bagged Trees, AdaBoost, Random Forest). These features were used in subsequent models to reduce overfitting and improve generalization.
- **Tuned RF with SMOTE**, **ANN with SMOTE**, and **XGBoost** achieved much higher recall and F1-scores than other models, with a slight trade-off in accuracy and precision due to more aggressive prediction of the minority class.
- **Tuned RF with SMOTE** provided the **best overall balance** with the highest accuracy and F1-score among the three. However, **ANN with SMOTE** was **better for maximizing the identification of oat crops**, achieving the highest recall while providing acceptable class discrimination with a fair AUC.

## How to Run

1. Clone the repository or download the ZIP file from GitHub.
2. Open the project folder in RStudio.
3. Run the R script (`oat_crops_classification.r`) inside the RStudio environment.

## Author

Developed by Juan Nathan.




































