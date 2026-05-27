"""
============================================================
  LOAN APPROVAL PREDICTION SYSTEM
  Automates loan decisions using ML — Logistic Regression
  and Decision Tree classifiers with full evaluation.
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')
np.random.seed(42)

OUTPUT_DIR = "/home/claude/loan_approval_system/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ──────────────────────────────────────────────

def generate_loan_dataset(n=614):
    """Generate a realistic synthetic loan dataset."""
    rng = np.random.default_rng(42)

    gender        = rng.choice(['Male', 'Female'], n, p=[0.80, 0.20])
    married       = rng.choice(['Yes', 'No'], n, p=[0.65, 0.35])
    dependents    = rng.choice(['0', '1', '2', '3+'], n, p=[0.57, 0.17, 0.16, 0.10])
    education     = rng.choice(['Graduate', 'Not Graduate'], n, p=[0.78, 0.22])
    self_employed = rng.choice(['Yes', 'No'], n, p=[0.14, 0.86])

    applicant_income   = rng.integers(1500, 15000, n)
    coapplicant_income = np.where(married == 'Yes',
                                  rng.integers(0, 6000, n),
                                  rng.integers(0, 500, n))
    loan_amount       = (applicant_income * rng.uniform(0.3, 4.0, n) / 1000).astype(int) + 50
    loan_amount_term  = rng.choice([12, 36, 60, 84, 120, 180, 240, 300, 360, 480], n,
                                   p=[0.01,0.03,0.02,0.02,0.04,0.07,0.04,0.05,0.68,0.04])
    credit_history    = rng.choice([1, 0], n, p=[0.84, 0.16])
    property_area     = rng.choice(['Urban', 'Semiurban', 'Rural'], n, p=[0.38, 0.36, 0.26])

    # Approval logic (realistic rules)
    score = (
        credit_history * 40
        + (education == 'Graduate') * 10
        + np.clip((applicant_income + coapplicant_income) / 1000, 0, 20)
        + (property_area == 'Semiurban') * 5
        + (property_area == 'Urban') * 3
        - (self_employed == 'Yes') * 5
        + rng.normal(0, 8, n)
    )
    loan_status = np.where(score >= 42, 'Y', 'N')

    df = pd.DataFrame({
        'Loan_ID'            : [f'LP{str(i).zfill(6)}' for i in range(1, n+1)],
        'Gender'             : gender,
        'Married'            : married,
        'Dependents'         : dependents,
        'Education'          : education,
        'Self_Employed'      : self_employed,
        'ApplicantIncome'    : applicant_income,
        'CoapplicantIncome'  : coapplicant_income.astype(int),
        'LoanAmount'         : loan_amount,
        'Loan_Amount_Term'   : loan_amount_term,
        'Credit_History'     : credit_history.astype(float),
        'Property_Area'      : property_area,
        'Loan_Status'        : loan_status,
    })

    # Sprinkle realistic missings
    for col, rate in [('Gender',0.013),('Married',0.003),('Dependents',0.025),
                      ('Self_Employed',0.032),('LoanAmount',0.035),
                      ('Loan_Amount_Term',0.021),('Credit_History',0.083)]:
        mask = rng.random(n) < rate
        df.loc[mask, col] = np.nan

    return df


# ──────────────────────────────────────────────
# 2. PREPROCESSING
# ──────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    data = df.copy()
    data.drop(columns=['Loan_ID'], inplace=True)

    # Fill missing values
    for col in ['Gender','Married','Dependents','Self_Employed','Credit_History']:
        data[col].fillna(data[col].mode()[0], inplace=True)
    for col in ['LoanAmount','Loan_Amount_Term']:
        data[col].fillna(data[col].median(), inplace=True)

    # Feature engineering
    data['Total_Income']        = data['ApplicantIncome'] + data['CoapplicantIncome']
    data['Income_to_Loan']      = data['Total_Income'] / (data['LoanAmount'] + 1)
    data['Log_LoanAmount']      = np.log1p(data['LoanAmount'])
    data['Log_ApplicantIncome'] = np.log1p(data['ApplicantIncome'])

    # Encode categoricals
    le = LabelEncoder()
    cat_cols = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
    for col in cat_cols:
        data[col] = le.fit_transform(data[col].astype(str))

    data['Loan_Status'] = (data['Loan_Status'] == 'Y').astype(int)

    X = data.drop('Loan_Status', axis=1)
    y = data['Loan_Status']
    return X, y, data


# ──────────────────────────────────────────────
# 3. MODEL TRAINING
# ──────────────────────────────────────────────

def train_models(X_train, y_train):
    lr = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])

    dt = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model',   DecisionTreeClassifier(max_depth=5, min_samples_split=20,
                                           min_samples_leaf=10, random_state=42))
    ])

    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    return lr, dt


# ──────────────────────────────────────────────
# 4. EVALUATION
# ──────────────────────────────────────────────

def evaluate_model(name, model, X_test, y_test, X_train, y_train):
    y_pred     = model.predict(X_test)
    y_prob     = model.predict_proba(X_test)[:, 1]
    acc        = accuracy_score(y_test, y_pred)
    f1         = f1_score(y_test, y_pred)
    auc        = roc_auc_score(y_test, y_prob)
    cv_scores  = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(5), scoring='accuracy')

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  CV Mean  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Rejected','Approved'])}")

    return {
        'name': name, 'accuracy': acc, 'f1': f1, 'auc': auc,
        'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
        'y_pred': y_pred, 'y_prob': y_prob,
        'cm': confusion_matrix(y_test, y_pred)
    }


# ──────────────────────────────────────────────
# 5. VISUALISATIONS
# ──────────────────────────────────────────────

PALETTE = {
    'bg'      : '#0d1117',
    'panel'   : '#161b22',
    'accent1' : '#58a6ff',
    'accent2' : '#3fb950',
    'accent3' : '#f78166',
    'accent4' : '#d2a8ff',
    'text'    : '#e6edf3',
    'muted'   : '#8b949e',
}

def apply_dark_style(ax, title=''):
    ax.set_facecolor(PALETTE['panel'])
    ax.tick_params(colors=PALETTE['muted'])
    ax.xaxis.label.set_color(PALETTE['muted'])
    ax.yaxis.label.set_color(PALETTE['muted'])
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    if title:
        ax.set_title(title, color=PALETTE['text'], fontsize=12, fontweight='bold', pad=12)


def plot_eda(df: pd.DataFrame):
    fig = plt.figure(figsize=(18, 12), facecolor=PALETTE['bg'])
    fig.suptitle('LOAN DATASET — EXPLORATORY DATA ANALYSIS',
                 fontsize=18, fontweight='bold', color=PALETTE['text'], y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.92, bottom=0.08)

    # 1. Loan Status distribution
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df['Loan_Status'].value_counts()
    bars = ax1.bar(['Approved', 'Rejected'], counts.values,
                   color=[PALETTE['accent2'], PALETTE['accent3']], width=0.5,
                   edgecolor='none')
    for bar, v in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{v}\n({v/len(df)*100:.1f}%)',
                 ha='center', color=PALETTE['text'], fontsize=10, fontweight='bold')
    apply_dark_style(ax1, 'Loan Status Distribution')
    ax1.set_ylabel('Count')

    # 2. Credit History vs Approval
    ax2 = fig.add_subplot(gs[0, 1])
    ch_data = df.dropna(subset=['Credit_History'])
    approved   = ch_data[ch_data['Loan_Status'] == 'Y']['Credit_History'].value_counts()
    rejected   = ch_data[ch_data['Loan_Status'] == 'N']['Credit_History'].value_counts()
    x = np.array([0, 1])
    w = 0.35
    ax2.bar(x - w/2, [approved.get(0,0), approved.get(1,0)],
            width=w, label='Approved', color=PALETTE['accent2'], alpha=0.9)
    ax2.bar(x + w/2, [rejected.get(0,0), rejected.get(1,0)],
            width=w, label='Rejected', color=PALETTE['accent3'], alpha=0.9)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['No Credit History', 'Has Credit History'])
    ax2.legend(facecolor=PALETTE['panel'], labelcolor=PALETTE['text'], fontsize=9)
    apply_dark_style(ax2, 'Credit History vs Approval')

    # 3. Income distribution
    ax3 = fig.add_subplot(gs[0, 2])
    for status, color, label in [('Y', PALETTE['accent2'], 'Approved'),
                                   ('N', PALETTE['accent3'], 'Rejected')]:
        subset = df[df['Loan_Status'] == status]['ApplicantIncome'].dropna()
        ax3.hist(subset, bins=30, alpha=0.7, color=color, label=label, edgecolor='none')
    ax3.legend(facecolor=PALETTE['panel'], labelcolor=PALETTE['text'], fontsize=9)
    apply_dark_style(ax3, 'Applicant Income Distribution')
    ax3.set_xlabel('Income')
    ax3.set_ylabel('Frequency')

    # 4. Property Area
    ax4 = fig.add_subplot(gs[1, 0])
    area_ct = pd.crosstab(df['Property_Area'], df['Loan_Status'])
    areas = area_ct.index.tolist()
    approved_v = area_ct.get('Y', pd.Series([0]*len(areas))).values
    rejected_v = area_ct.get('N', pd.Series([0]*len(areas))).values
    xi = np.arange(len(areas))
    ax4.bar(xi - 0.2, approved_v, 0.4, color=PALETTE['accent2'], label='Approved')
    ax4.bar(xi + 0.2, rejected_v, 0.4, color=PALETTE['accent3'], label='Rejected')
    ax4.set_xticks(xi); ax4.set_xticklabels(areas)
    ax4.legend(facecolor=PALETTE['panel'], labelcolor=PALETTE['text'], fontsize=9)
    apply_dark_style(ax4, 'Property Area vs Loan Status')

    # 5. Loan Amount distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(df['LoanAmount'].dropna(), bins=35,
             color=PALETTE['accent1'], edgecolor='none', alpha=0.85)
    apply_dark_style(ax5, 'Loan Amount Distribution')
    ax5.set_xlabel('Loan Amount (K)')
    ax5.set_ylabel('Frequency')

    # 6. Education vs Approval
    ax6 = fig.add_subplot(gs[1, 2])
    edu_ct = pd.crosstab(df['Education'], df['Loan_Status'])
    edus = edu_ct.index.tolist()
    app_e = edu_ct.get('Y', pd.Series([0]*len(edus))).values
    rej_e = edu_ct.get('N', pd.Series([0]*len(edus))).values
    xi = np.arange(len(edus))
    ax6.bar(xi - 0.2, app_e, 0.4, color=PALETTE['accent2'], label='Approved')
    ax6.bar(xi + 0.2, rej_e, 0.4, color=PALETTE['accent3'], label='Rejected')
    ax6.set_xticks(xi); ax6.set_xticklabels(edus)
    ax6.legend(facecolor=PALETTE['panel'], labelcolor=PALETTE['text'], fontsize=9)
    apply_dark_style(ax6, 'Education vs Loan Status')

    path = f"{OUTPUT_DIR}/01_eda.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"[✓] EDA saved → {path}")


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE['bg'])
    fig.suptitle('CONFUSION MATRICES', fontsize=16, fontweight='bold',
                 color=PALETTE['text'], y=1.02)

    cmaps = ['Blues', 'Greens']
    for ax, res, cmap in zip(axes, results, cmaps):
        cm = res['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    xticklabels=['Rejected','Approved'],
                    yticklabels=['Rejected','Approved'],
                    linewidths=2, linecolor=PALETTE['bg'],
                    cbar_kws={'shrink': 0.8})
        ax.set_facecolor(PALETTE['panel'])
        ax.set_title(res['name'], color=PALETTE['text'], fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted', color=PALETTE['muted'])
        ax.set_ylabel('Actual', color=PALETTE['muted'])
        ax.tick_params(colors=PALETTE['muted'])

    fig.patch.set_facecolor(PALETTE['bg'])
    path = f"{OUTPUT_DIR}/02_confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"[✓] Confusion matrices saved → {path}")


def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=PALETTE['bg'])
    ax.set_facecolor(PALETTE['panel'])

    colors = [PALETTE['accent1'], PALETTE['accent2']]
    for res, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{res['name']}  (AUC = {res['auc']:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.06, color=color)

    ax.plot([0,1],[0,1], 'w--', lw=1, alpha=0.4, label='Random (AUC = 0.500)')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate', color=PALETTE['muted'])
    ax.set_ylabel('True Positive Rate', color=PALETTE['muted'])
    ax.set_title('ROC CURVES — MODEL COMPARISON', color=PALETTE['text'],
                 fontsize=14, fontweight='bold', pad=14)
    ax.legend(facecolor=PALETTE['panel'], labelcolor=PALETTE['text'], fontsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.tick_params(colors=PALETTE['muted'])
    ax.grid(alpha=0.15, color=PALETTE['muted'])

    path = f"{OUTPUT_DIR}/03_roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"[✓] ROC curves saved → {path}")


def plot_feature_importance(dt_model, feature_names):
    importances = dt_model.named_steps['model'].feature_importances_
    indices     = np.argsort(importances)[::-1]
    top_n       = min(12, len(feature_names))
    idx         = indices[:top_n]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PALETTE['bg'])
    ax.set_facecolor(PALETTE['panel'])

    colors = [PALETTE['accent1'] if i == 0 else PALETTE['accent4'] for i in range(top_n)]
    bars   = ax.barh(range(top_n), importances[idx][::-1], color=colors[::-1],
                     edgecolor='none', height=0.65)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx[::-1]], color=PALETTE['text'], fontsize=11)
    ax.set_xlabel('Importance Score', color=PALETTE['muted'])
    ax.set_title('DECISION TREE — FEATURE IMPORTANCE', color=PALETTE['text'],
                 fontsize=14, fontweight='bold', pad=14)
    ax.tick_params(colors=PALETTE['muted'])
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.grid(axis='x', alpha=0.15, color=PALETTE['muted'])

    path = f"{OUTPUT_DIR}/04_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"[✓] Feature importance saved → {path}")


def plot_metrics_comparison(results):
    metrics  = ['accuracy', 'f1', 'auc', 'cv_mean']
    labels   = ['Accuracy', 'F1 Score', 'ROC-AUC', 'CV Accuracy']
    models   = [r['name'] for r in results]
    values   = [[r[m] for m in metrics] for r in results]

    x   = np.arange(len(metrics))
    w   = 0.30
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=PALETTE['bg'])
    ax.set_facecolor(PALETTE['panel'])

    bar_colors = [PALETTE['accent1'], PALETTE['accent2']]
    for i, (model_vals, color, name) in enumerate(zip(values, bar_colors, models)):
        bars = ax.bar(x + (i - 0.5) * w, model_vals, w, label=name,
                      color=color, edgecolor='none', alpha=0.9)
        for bar, v in zip(bars, model_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', color=PALETTE['text'], fontsize=9.5)

    ax.set_xticks(x); ax.set_xticklabels(labels, color=PALETTE['text'], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', color=PALETTE['muted'])
    ax.set_title('MODEL PERFORMANCE COMPARISON', color=PALETTE['text'],
                 fontsize=14, fontweight='bold', pad=14)
    ax.legend(facecolor=PALETTE['panel'], labelcolor=PALETTE['text'], fontsize=11)
    ax.tick_params(colors=PALETTE['muted'])
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.grid(axis='y', alpha=0.15, color=PALETTE['muted'])

    path = f"{OUTPUT_DIR}/05_metrics_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"[✓] Metrics comparison saved → {path}")


# ──────────────────────────────────────────────
# 6. PREDICTION FUNCTION
# ──────────────────────────────────────────────

def predict_loan(model, applicant: dict, feature_names):
    """
    Predict loan approval for a single applicant.
    applicant dict keys must match training feature names after encoding.
    """
    df_in = pd.DataFrame([applicant])
    # Simple label encode for demo
    maps = {
        'Gender'       : {'Male': 1, 'Female': 0},
        'Married'      : {'Yes': 1, 'No': 0},
        'Dependents'   : {'0': 0, '1': 1, '2': 2, '3+': 3},
        'Education'    : {'Graduate': 0, 'Not Graduate': 1},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    }
    for col, m in maps.items():
        if col in df_in.columns:
            df_in[col] = df_in[col].map(m)

    df_in['Total_Income']        = df_in['ApplicantIncome'] + df_in['CoapplicantIncome']
    df_in['Income_to_Loan']      = df_in['Total_Income'] / (df_in['LoanAmount'] + 1)
    df_in['Log_LoanAmount']      = np.log1p(df_in['LoanAmount'])
    df_in['Log_ApplicantIncome'] = np.log1p(df_in['ApplicantIncome'])

    df_in = df_in[feature_names]
    prob   = model.predict_proba(df_in)[0][1]
    status = 'APPROVED ✅' if prob >= 0.5 else 'REJECTED ❌'
    return status, prob


# ──────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────

def main():
    print("\n" + "═"*55)
    print("   LOAN APPROVAL PREDICTION SYSTEM")
    print("   Python · Logistic Regression · Decision Tree")
    print("═"*55)

    # ── Data
    print("\n[1/6] Generating dataset...")
    df = generate_loan_dataset(614)
    print(f"      Shape: {df.shape}  |  Approved: {(df['Loan_Status']=='Y').sum()}  |  Rejected: {(df['Loan_Status']=='N').sum()}")
    print(f"      Missing values:\n{df.isnull().sum()[df.isnull().sum()>0].to_string()}")

    # ── EDA
    print("\n[2/6] Creating EDA visualisations...")
    plot_eda(df)

    # ── Preprocessing
    print("\n[3/6] Preprocessing...")
    X, y, processed = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"      Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    print(f"      Features: {list(X.columns)}")

    # ── Training
    print("\n[4/6] Training models...")
    lr_model, dt_model = train_models(X_train, y_train)
    print("      Logistic Regression — done")
    print("      Decision Tree       — done")

    # ── Evaluation
    print("\n[5/6] Evaluating models...")
    lr_res = evaluate_model("Logistic Regression", lr_model, X_test, y_test, X_train, y_train)
    dt_res = evaluate_model("Decision Tree",       dt_model, X_test, y_test, X_train, y_train)
    results = [lr_res, dt_res]

    # ── Plots
    print("\n[6/6] Generating visualisations...")
    plot_confusion_matrices(results)
    plot_roc_curves(results, y_test)
    plot_feature_importance(dt_model, list(X.columns))
    plot_metrics_comparison(results)

    # ── Decision Tree rules
    tree_rules = export_text(dt_model.named_steps['model'],
                             feature_names=list(X.columns), max_depth=3)
    rules_path = f"{OUTPUT_DIR}/decision_tree_rules.txt"
    with open(rules_path, 'w') as f:
        f.write("DECISION TREE RULES (top 3 levels)\n")
        f.write("="*60 + "\n\n")
        f.write(tree_rules)
    print(f"[✓] Tree rules saved → {rules_path}")

    # ── Sample prediction
    print("\n" + "─"*55)
    print("  SAMPLE PREDICTION")
    print("─"*55)
    sample = {
        'Gender': 'Male', 'Married': 'Yes', 'Dependents': '1',
        'Education': 'Graduate', 'Self_Employed': 'No',
        'ApplicantIncome': 5000, 'CoapplicantIncome': 2000,
        'LoanAmount': 150, 'Loan_Amount_Term': 360,
        'Credit_History': 1.0, 'Property_Area': 'Urban',
    }
    print(f"  Applicant: {sample}")
    for name, model in [("Logistic Regression", lr_model), ("Decision Tree", dt_model)]:
        status, prob = predict_loan(model, sample, list(X.columns))
        print(f"  {name:22s} → {status}  (confidence: {prob:.2%})")

    print("\n" + "═"*55)
    print("  All outputs saved to:", OUTPUT_DIR)
    print("═"*55 + "\n")

    return lr_model, dt_model, X, y, df, results, y_test


if __name__ == '__main__':
    main()
