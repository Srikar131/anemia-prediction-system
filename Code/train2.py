import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap
import matplotlib.pyplot as plt
import joblib
import warnings
import os
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

def generate_xai_report(data_path='anemia.csv'):
    print("--- Starting Explainable AI (XAI) Pipeline ---")

    PLOTS_DIR = 'static/xai_plots'
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"XAI Plots will be saved in '{PLOTS_DIR}/' directory.")

    # 1. Load and Prepare Data
    try:
        print(f"\n[1/5] Loading data...")
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: '{data_path}' not found.")
        return

    X = data.drop('Result', axis=1)
    y = data['Result']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Train Model
    print("\n[2/5] Training Model (XGBoost)...")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    print("Model trained.")

    # 3. GLOBAL EXPLAINABILITY (How the model works generally)
    print("\n[3/5] Generating Global Explanations...")

    # A. Permutation Importance (Model Agnostic)
    # Checks how much accuracy drops if we randomly shuffle a feature
    print("   -> Generating Permutation Importance Plot...")
    perm_importance = permutation_importance(model, X_test, y_test)
    sorted_idx = perm_importance.importances_mean.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.title("Which features break the model if removed?")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '01_global_permutation_importance.png'))
    plt.close()

    # B. Partial Dependence Plot (PDP)
    # Shows the marginal effect of the most important feature (likely Hemoglobin)
    print("   -> Generating Partial Dependence Plot...")
    top_feature = X.columns[sorted_idx][-1] # Get most important feature
    
    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(model, X_test, [top_feature], ax=ax)
    plt.title(f"How {top_feature} values affect Anemia Risk (PDP)")
    plt.savefig(os.path.join(PLOTS_DIR, '02_global_partial_dependence.png'))
    plt.close()

    # 4. SHAP EXPLAINABILITY (Deep Dive)
    print("\n[4/5] Generating SHAP Deep-Dive Explanations...")
    
    # Calculate SHAP values (Using the modern 'Explainer' API for richer plots)
    explainer = shap.Explainer(model, X_test)
    shap_values_exp = explainer(X_test) # Returns an Explanation object
    
    # C. SHAP Beeswarm (Summary)
    print("   -> Generating SHAP Beeswarm Plot...")
    plt.figure()
    shap.plots.beeswarm(shap_values_exp, show=False)
    plt.title('SHAP Global Summary (Beeswarm)', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '03_shap_beeswarm.png'), bbox_inches='tight')
    plt.close()

    # D. SHAP Dependence Plot
    # Shows interaction between the top feature and the one that interacts with it most
    print(f"   -> Generating SHAP Dependence Plot for {top_feature}...")
    plt.figure()
    shap.plots.scatter(shap_values_exp[:, top_feature], color=shap_values_exp, show=False)
    plt.title(f'Dependence Plot: {top_feature}', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '04_shap_dependence.png'), bbox_inches='tight')
    plt.close()

    # 5. LOCAL EXPLAINABILITY (Patient Specific)
    print("\n[5/5] Generating Local Explanations (Single Patient)...")
    
    # Find one positive case (Has Anemia) and one negative case (No Anemia) in the test set
    # We use the model predictions to ensure we explain what the *model* thinks
    y_pred = model.predict(X_test)
    
    try:
        # Get index of first predicted positive and negative
        pos_idx = np.where(y_pred == 1)[0][0]
        neg_idx = np.where(y_pred == 0)[0][0]
        
        # E. Waterfall Plot (Positive Case)
        print(f"   -> Generating Waterfall for Patient #{pos_idx} (Predicted: Anemia)...")
        plt.figure()
        shap.plots.waterfall(shap_values_exp[pos_idx], show=False)
        plt.title(f'Why did the model predict Anemia for Patient #{pos_idx}?', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, '05_local_waterfall_anemia.png'), bbox_inches='tight')
        plt.close()

        # F. Waterfall Plot (Negative Case)
        print(f"   -> Generating Waterfall for Patient #{neg_idx} (Predicted: No Anemia)...")
        plt.figure()
        shap.plots.waterfall(shap_values_exp[neg_idx], show=False)
        plt.title(f'Why did the model predict NO Anemia for Patient #{neg_idx}?', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, '06_local_waterfall_healthy.png'), bbox_inches='tight')
        plt.close()

    except IndexError:
        print("Could not find examples of both classes in test set to generate local plots.")

    print("\n--- XAI Pipeline Finished. Check 'static/xai_plots' for results. ---")

if __name__ == '__main__':
    generate_xai_report('anemia.csv')
