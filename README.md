# Customer Churn Prediction Dashboard

An advanced ML + BI project:
- Models: Logistic Regression & XGBoost
- Explainability: SHAP
- Dashboards: Power BI & Tableau
- Dataset: [Telco Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Folder structure
- `churn_pipeline.py` – main script
- `data/` – input dataset
- `models/` – saved models (.pkl)
- `outputs/` – ROC curves, SHAP plots, predictions_for_dashboard.csv

## Run
```bash
python churn_pipeline.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
