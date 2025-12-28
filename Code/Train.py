import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def train_and_evaluate_model(data_path='anemia.csv'):
    """
    Loads data, trains an XGBoost classifier, evaluates it, generates plots, and saves the model.
    """
    print("--- Starting Anemia Prediction Model Pipeline ---")

    # Define directory to save plots
    PLOTS_DIR = 'static/plots'
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Plots will be saved in '{PLOTS_DIR}/' directory.")

    # 1. Load Data
    try:
        print(f"\n[1/8] Loading data from '{data_path}'...")
        data = pd.read_csv(data_path)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found.")
        return

    # 2. Data Preprocessing
    print("\n[2/8] Preprocessing data...")
    X = data.drop('Result', axis=1)
    y = data['Result']
    print("Features (X) and Target (y) have been separated.")

    # 3. Generate Data Exploration Plots
    print("\n[3/8] Generating data exploration plots...")

    # Plot 1: Target Variable Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title('Distribution of Anemia Cases (0 = No, 1 = Yes)')
    plt.xlabel('Result')
    plt.ylabel('Count')
    plt.savefig(os.path.join(PLOTS_DIR, 'target_distribution.png'))
    plt.close()

    # Plot 2: Feature Distributions
    X.hist(bins=15, figsize=(15, 10), layout=(2, 3))
    plt.suptitle('Distribution of Features')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_distributions.png'))
    plt.close()

    # Plot 3: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'))
    plt.close()
    print("Data exploration plots saved.")

    # 4. Split Data
    print("\n[4/8] Splitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split successfully.")

    # 5. Train XGBoost Model
    print("\n[5/8] Training the XGBoost classifier...")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    print("XGBoost model trained successfully.")

    # 6. Evaluate Model and Generate Performance Plots
    print("\n[6/8] Evaluating model and generating performance plots...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # --- Print Metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("\n--- Model Performance Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("---------------------------------")


    # Plot 4: Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Anemia', 'Anemia'], yticklabels=['No Anemia', 'Anemia'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'))
    plt.close()

    # Plot 5: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'))
    plt.close()
    print("Model performance plots saved.")

    # 7. Explainability with SHAP
    print("\n[7/8] Generating model explanations with SHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Plot 6: SHAP Summary Plot (Beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title('SHAP Summary Plot for Anemia Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'shap_summary_plot.png'))
    plt.close()
    
    # Plot 7: SHAP Feature Importance (Bar)
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'shap_feature_importance.png'))
    plt.close()
    print("SHAP explanation plots saved.")

    # 8. Save the Model
    print("\n[8/8] Saving the trained model...")
    model_filename = 'xgboost_anemia_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Model successfully saved as '{model_filename}'.")

    print("\n--- Pipeline Finished Successfully ---")

if __name__ == '__main__':
    train_and_evaluate_model('anemia.csv')

