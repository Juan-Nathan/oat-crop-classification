# 🌾 Classification of Oat Crops using Remote Sensing Data

This project applies various classification models in R to detect whether **oats** are being grown on farmland using optical and radar-based remote sensing data. The dataset is derived from a modified version of the Winnipeg Crop Mapping Dataset, hosted by the UCI Machine Learning Archive.

## Dataset

- `WinnData.csv`
- **Source**: [UCI Machine Learning Archive](https://archive.ics.uci.edu/dataset/525/crop+mapping+using+fused+optical+radar+data+set)
- **Features**: 20 randomly selected variables (`A01`–`A30`), plus a binary target `Class`
- **Observations**: 5000
- **Class Distribution**:
  - Oats: 14.2%
  - Other: 85.8%

## Technologies Used

- **Language**: R
- **IDE**: RStudio
- **Packages**: `tree`, `e1071`, `ROCR`, `randomForest`, `adabag`, `rpart`, `smotefamily`, `neuralnet`, `xgboost`

## Project Workflow

1. **Exploratory Data Analysis**
   - Summary statistics, standard deviations, and correlation analysis
   - Identification of outliers and multicollinearity (e.g., `A16`, `A19`, and `A20`)
2. **Preprocessing**
   - Log transformation of skewed features (`A26`)
   - Removal of highly correlated variables (`A16` and `A19`)
   - Synthetic Minority Oversampling Technique (SMOTE) for class balance
   - Feature scaling for neural network models
3. **Model Training**
   - Trained and evaluated:
     - Decision Tree
     - Naive Bayes
     - Bagged Trees
     - AdaBoost
     - Random Forest (RF)
     - Hyperparameter-tuned RF on SMOTE-balanced data
     - Artificial Neural Network (ANN) on SMOTE-balanced data
     - XGBoost
4. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-Score
   - ROC Curve and AUC

## Model Performance Summary

| Model                 | Accuracy | Precision | Recall | F1-Score |  AUC  |
|-----------------------|----------|-----------|--------|----------|-------|
| Decision Tree         | 87%      | 53%       | 4%     | 7%       | 0.69  |
| Naive Bayes           | 80%      | 25%       | 23%    | 24%      | 0.70  |
| Bagged Trees          | 87%      | 67%       | 3%     | 6%       | 0.77  |
| AdaBoost              | 87%      | 51%       | 24%    | 32%      | 0.73  |
| Random Forest         | 87%      | 71%       | 7%     | 14%      | 0.78  |
| Tuned RF on SMOTE     | 80%      | 42%       | 53%    | 47%      | 0.72  |
| ANN on SMOTE          | 67%      | 25%       | 72%    | 37%      | 0.73  |
| XGBoost               | 77%      | 32%       | 61%    | 42%      | 0.78  |

## Outcomes

- Class imbalance had a major impact on model performance, particularly on recall for the minority class.
- F1-score was a more reliable evaluation metric than accuracy, which was inflated by the dominance of the majority class.
- The attributes `A26`, `A02`, `A06`, `A07`, `A08`, and `A14` were consistently identified as the most significant across the "basic" models (Decision Tree, Naive Bayes, Bagged Trees, AdaBoost, Random Forest). These features were used in subsequent models to reduce overfitting and improve generalization.
- Tuned RF on SMOTE, ANN on SMOTE, and XGBoost achieved much higher recall and F1-scores than other models, with a slight trade-off in accuracy and precision due to more aggressive prediction of the minority class (Oats).
- Tuned RF on SMOTE provides the best overall balance with the highest accuracy and F1-score among the three, while ANN on SMOTE is better for maximizing the detection of Oats, as it achieves the highest recall and a decent AUC that indicates acceptable discrimination between classes.

## How to Run

1. Clone the repository or download the ZIP file from GitHub.
2. Open the project folder in RStudio.
3. Run the R script (`oats_classification.r`) inside the RStudio environment.

## Author

Developed by Juan Nathan.


