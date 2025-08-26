#!/usr/bin/env python3
"""
churn_pipeline.py
End-to-end training, evaluation, explainability and export for dashboards.
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report, roc_curve, auc
)
import shap
import warnings
warnings.filterwarnings("ignore")

# --------------------------
# Utility functions
# --------------------------
def load_telco(csv_path):
    df = pd.read_csv(csv_path)
    # common Telco cleaning:
    # Convert TotalCharges to numeric (some rows are blank)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan), errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    # Standardize churn label to 0/1
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
    return df

def make_feature_lists(df, ignore_cols):
    # numeric columns: pick common numeric features (tenure, MonthlyCharges, TotalCharges, SeniorCitizen)
    numeric = [c for c in ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges'] if c in df.columns and c not in ignore_cols]
    # categorical = all object/string columns except ignore and ID
    categorical = [c for c in df.columns if (df[c].dtype == 'object' or df[c].dtype.name == 'category') and c not in ignore_cols]
    # drop target if present
    return numeric, categorical

def build_preprocessor(numeric_features, categorical_features):
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ])
    return preprocessor

def get_transformed_feature_names(preprocessor, numeric_features, categorical_features):
    # after fitting preprocessor, extract feature names
    feature_names = []
    # numeric names
    feature_names.extend(numeric_features)
    # onehot names
    cat_transform = preprocessor.named_transformers_['cat']
    ohe = cat_transform.named_steps['onehot']
    ohe_names = list(ohe.get_feature_names_out(categorical_features))
    feature_names.extend(ohe_names)
    return feature_names

# --------------------------
# Main pipeline
# --------------------------
def main(args):
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    print("[1/9] Loading data...")
    df = load_telco(args.data)
    id_col = 'customerID' if 'customerID' in df.columns else None
    target = 'Churn'
    ignore_cols = [id_col, target] if id_col else [target]

    numeric_features, categorical_features = make_feature_lists(df, ignore_cols)
    print(" Numeric features:", numeric_features)
    print(" Categorical features:", categorical_features[:10], "..." if len(categorical_features)>10 else "")

    X = df.drop(columns=[target])
    if id_col:
        X_ids = X[id_col].copy()
        X = X.drop(columns=[id_col])
    else:
        X_ids = pd.Series(np.arange(len(X)), name='id')

    y = df[target].copy()

    # split for evaluation
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(X, y, X_ids, test_size=0.2, random_state=42, stratify=y)

    print("[2/9] Building preprocessors...")
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Pipelines with SMOTE before classifier (works after preprocessor)
    print("[3/9] Building pipelines...")
    pipe_logreg = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', LogisticRegression(max_iter=2000, solver='lbfgs'))
    ])
    pipe_xgb = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1))
    ])

    # Hyperparameter search (randomized for speed)
    print("[4/9] Hyperparameter tuning (RandomizedSearchCV)... this may take a while")
    param_dist_logreg = {
        'clf__C': [0.01, 0.1, 1, 10],
    }
    param_dist_xgb = {
        'clf__n_estimators': [100, 200, 400],
        'clf__max_depth': [3, 4, 6],
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__subsample': [0.6, 0.8, 1.0],
        'clf__colsample_bytree': [0.6, 0.8, 1.0]
    }

    rs_logreg = RandomizedSearchCV(pipe_logreg, param_dist_logreg, n_iter=4, scoring='roc_auc', cv=3, verbose=1, random_state=42)
    rs_xgb = RandomizedSearchCV(pipe_xgb, param_dist_xgb, n_iter=20, scoring='roc_auc', cv=3, verbose=1, random_state=42)

    rs_logreg.fit(X_train, y_train)
    print(" LogReg best score:", rs_logreg.best_score_, "params:", rs_logreg.best_params_)
    rs_xgb.fit(X_train, y_train)
    print(" XGB best score:", rs_xgb.best_score_, "params:", rs_xgb.best_params_)

    best_logreg = rs_logreg.best_estimator_
    best_xgb = rs_xgb.best_estimator_

    print("[5/9] Evaluate on test set...")
    # test predictions
    y_proba_logreg = best_logreg.predict_proba(X_test)[:,1]
    y_proba_xgb    = best_xgb.predict_proba(X_test)[:,1]
    # pick primary model for metrics (show both)
    for name, y_proba in [('LogisticRegression', y_proba_logreg), ('XGBoost', y_proba_xgb)]:
        y_pred = (y_proba >= 0.5).astype(int)
        print(f"\n--- {name} metrics ---")
        print(" ROC AUC:", roc_auc_score(y_test, y_proba))
        print(classification_report(y_test, y_pred, digits=4))

    # ROC curves
    fpr1, tpr1, _ = roc_curve(y_test, y_proba_logreg)
    fpr2, tpr2, _ = roc_curve(y_test, y_proba_xgb)
    auc1, auc2 = auc(fpr1, tpr1), auc(fpr2, tpr2)

    plt.figure(figsize=(8,6))
    plt.plot(fpr1, tpr1, label=f'LogReg (AUC = {auc1:.3f})')
    plt.plot(fpr2, tpr2, label=f'XGBoost (AUC = {auc2:.3f})')
    plt.plot([0,1],[0,1],'k--', alpha=0.3)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('outputs/roc_both.png', dpi=150)
    plt.close()
    print("[6/9] ROC saved -> outputs/roc_both.png")

    # ---------------------------
    # SHAP explainability for XGBoost
    # ---------------------------
    print("[7/9] Running SHAP for XGBoost (summary plot)")
    # Need to transform data with preprocessor to numeric matrix and get feature names
    # Extract preprocessor (fitted inside pipeline)
    fitted_preprocessor = best_xgb.named_steps['preprocessor']
    # Fit preprocessor on full training used in pipeline (automatic)
    X_train_trans = fitted_preprocessor.transform(X_train)
    X_test_trans = fitted_preprocessor.transform(X_test)

    # Build feature names
    feature_names = get_transformed_feature_names(fitted_preprocessor, numeric_features, categorical_features)
    # Create explainer with the raw xgb model
    xgb_model = best_xgb.named_steps['clf']
    # Use shap.Explainer (works with sklearn/xgboost objects)
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X_test_trans)  # this is a shap.Explanation object

    # Summary plot
    shap.summary_plot(shap_values, features=X_test_trans, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('outputs/shap_summary.png', dpi=150)
    plt.close()
    print("[7/9] SHAP summary saved -> outputs/shap_summary.png")

    # ---------------------------
    # Retrain best models on full dataset for export (deploy)
    # ---------------------------
    print("[8/9] Retraining best models on FULL dataset for deployment/export...")
    # Fit preprocessor inside the pipelines on full X & y
    best_logreg.fit(X, y)
    best_xgb.fit(X, y)

    # Save serialized pipelines
    joblib.dump(best_logreg, 'models/logreg_pipeline.pkl')
    joblib.dump(best_xgb, 'models/xgb_pipeline.pkl')
    print("[8/9] Models saved -> models/")

    # Produce predictions for dashboard (entire dataset)
    proba_logreg_full = best_logreg.predict_proba(X)[:,1]
    proba_xgb_full = best_xgb.predict_proba(X)[:,1]
    ensemble_prob = (proba_logreg_full + proba_xgb_full) / 2.0
    predicted_label = (ensemble_prob >= 0.5).astype(int)
    # Risk buckets: thresholds adjustable
    risk = np.where(ensemble_prob >= 0.6, 'High',
            np.where(ensemble_prob >= 0.4, 'Medium', 'Low'))

    out_df = pd.DataFrame({
        'customerID': X_ids.values,
        'churn_prob_logreg': proba_logreg_full,
        'churn_prob_xgb': proba_xgb_full,
        'churn_prob_ensemble': ensemble_prob,
        'predicted_churn': predicted_label,
        'risk_segment': risk
    })
    # Attach a few useful drill-down columns for dashboards (if present)
    add_cols = ['tenure','MonthlyCharges','TotalCharges','Contract','PaymentMethod']
    for c in add_cols:
        if c in X.columns:
            out_df[c] = X[c].values

    out_csv = 'outputs/predictions_for_dashboard.csv'
    out_df.to_csv(out_csv, index=False)
    print(f"[9/9] Dashboard CSV exported -> {out_csv}")

    print("All done. Useful files in ./models and ./outputs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/WA_Fn-UseC_-Telco-Customer-Churn.csv', help='Path to Telco churn CSV')
    args = parser.parse_args()
    main(args)
