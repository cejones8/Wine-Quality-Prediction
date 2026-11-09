# Interpretable Machine Learning for Wine Quality Prediction

Research project at James Madison University (Mar. - Aug. 2025) predicting wine quality scores from physicochemical properties, with a focus on explaining what drives the predictions.

## Dataset

UCI Wine Quality dataset: 6,497 samples, 11 physicochemical features (e.g., alcohol, volatile acidity, free sulfur dioxide).

## Approach

- Median imputation for missing values
- Correlation-based feature selection to remove multicollinearity
- Random Forest classifier (scikit-learn) for multi-class quality prediction
- SMOTE and RandomizedSearchCV to address severe class imbalance
- Permutation importance and SHAP for interpretability

## Results

- Macro-average F1 improved from 0.52 to 0.55, with better recall on rare quality classes
- Weighted F1 held steady at 0.82
- Top predictors: alcohol, volatile acidity, and free sulfur dioxide
- Alcohol's effect was conditional on low volatile acidity, consistent with oenological literature

## Files

- `WINE.ipynb`: full analysis and modeling
- `Wine_Quality.pdf`: project write-up
- `Wine_Quality.zip`: project write-up, full analysis, and images

## Run it yourself

1. Clone the repo
2. Install dependencies: `pip install pandas numpy scikit-learn imbalanced-learn shap matplotlib`
3. Open `WINE.ipynb` in Jupyter
