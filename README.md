# Boosting Cardiac Risk Prediction

A machine learning framework for heart disease prediction that combines stacked gradient boosting with Bayesian hyperparameter tuning.

**Mohammad Nadeem, Parupalli Bhargav Kumar**
School of Computer Science Engineering and Technology, Bennett University
Supervised by Dr. Shailza Kanwar

*Undergraduate research project carried out in 2025 at Bennett University under the supervision of Dr. Shailza Kanwar. The code was re-implemented and published in 2026.*

## Method

1. **Data:** UCI Heart Disease dataset (Cleveland data), 303 patients, 13 clinical attributes, binary target (164 without and 139 with heart disease)
2. **Pre-processing:** stratified 80:20 train-test split, outlier removal on the training set (z-score and IQR), imputation, one-hot encoding, standard scaling
3. **Feature selection:** Pearson correlation check and SelectKBest with the ANOVA F-statistic
4. **Class balancing:** Borderline-SMOTE, applied to training data only
5. **Base learners:** XGBoost, LightGBM, CatBoost, HistGradientBoosting
6. **Tuning:** Bayesian optimization with Optuna (40 trials per model, 5-fold stratified CV ROC-AUC)
7. **Ensemble:** stacking with a logistic regression meta-learner
8. **Evaluation:** accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix and ROC curves, compared with logistic regression and random forest baselines

All data-dependent steps are inside a single `imbalanced-learn` pipeline, so they are fitted on training folds only and the test set is used only for the final evaluation.

## Results

Test set of 61 patients; cross-validated ROC-AUC on the training set (mean ± SD).

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.852 | 0.806 | 0.893 | 0.847 | 0.947 | 0.857 ± 0.065 |
| Random forest (baseline) | 0.902 | 0.844 | 0.964 | 0.900 | 0.935 | 0.888 ± 0.063 |
| XGBoost | 0.869 | 0.833 | 0.893 | 0.862 | 0.946 | 0.884 ± 0.049 |
| LightGBM | 0.869 | 0.833 | 0.893 | 0.862 | 0.955 | 0.894 ± 0.052 |
| CatBoost | 0.836 | 0.781 | 0.893 | 0.833 | 0.942 | 0.892 ± 0.058 |
| HistGradientBoosting | 0.885 | 0.862 | 0.893 | 0.877 | 0.951 | 0.888 ± 0.050 |
| Stacked ensemble | 0.885 | 0.862 | 0.893 | 0.877 | 0.948 | 0.890 ± 0.052 |

The stacked ensemble performed similarly to the best individual boosting models. On a dataset of this size the differences between models are within the expected variation, so stacking gave stable results rather than a clear improvement.

All figures and tables are in the [`results/`](results) folder.

## How to run

Open `Cardiac_Risk_Prediction_Stacked_Boosting.ipynb` in Google Colab and select **Runtime → Run all**. The notebook installs the required libraries, downloads the dataset from the UCI repository, and saves all figures, tables, tuned hyperparameters and the trained model in `results/`.

## Dataset

A. Janosi, W. Steinbrunn, M. Pfisterer and R. Detrano, "Heart Disease," UCI Machine Learning Repository, 1988, doi: 10.24432/C52P4X.
