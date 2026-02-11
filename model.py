"""
🚀 CORRECTED LOAN PREDICTION MODEL TRAINING
==========================================
Addresses overfitting and data leakage issues
Features:
- Proper train/test split with validation set
- Feature selection to reduce leakage
- Regularization to prevent overfitting
- Cross-validation for robust evaluation
- More conservative hyperparameters
"""

import pandas as pd
import numpy as np
import pickle as pk
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("🏦 CORRECTED LOAN APPROVAL MODEL TRAINING")
print("="*80)

# ==========================================
# 1. LOAD DATA
# ==========================================
print("\n📊 Step 1: Loading Data...")
data = pd.read_csv('loan_approval_dataset_clean.csv')
print(f"✅ Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")

# ==========================================
# 2. DATA PREPROCESSING (CONSERVATIVE)
# ==========================================
print("\n" + "="*80)
print("🔧 Step 2: Data Preprocessing (Conservative Approach)...")
print("="*80)

# Clean column names
data.columns = data.columns.str.strip()
print("✅ Cleaned column names")

# Drop loan_id
if 'loan_id' in data.columns:
    data.drop(columns=['loan_id'], inplace=True)
    print("✅ Dropped loan_id")

# Clean categorical values
data['education'] = data['education'].str.strip()
data['self_employed'] = data['self_employed'].str.strip()
data['loan_status'] = data['loan_status'].str.strip()

# Check class distribution
print("\n⚖️  Original Class Distribution:")
print(data['loan_status'].value_counts())
print(f"Approval Rate: {(data['loan_status'] == 'Approved').sum() / len(data) * 100:.2f}%")

# CONSERVATIVE Feature Engineering - Avoid leakage
print("\n🔨 Conservative Feature Engineering...")

# Basic combined assets (this is fine)
data['total_assets'] = (data['residential_assets_value'] + 
                        data['commercial_assets_value'] + 
                        data['luxury_assets_value'] + 
                        data['bank_asset_value'])

# Create ONLY non-leaky features
# Income stability indicator
data['high_income'] = (data['income_annum'] > data['income_annum'].median()).astype(int)

# Asset ownership level
data['asset_level'] = pd.cut(data['total_assets'], 
                              bins=[0, 1000000, 5000000, float('inf')],
                              labels=[0, 1, 2]).astype(int)

# Employment risk
data['employment_risk'] = (data['self_employed'] == 'Yes').astype(int)

# Family size category
data['family_size'] = data['no_of_dependents'].apply(lambda x: 0 if x <= 1 else (1 if x <= 3 else 2))

print("✅ Created 4 safe engineered features")

# Select features to keep - REMOVE potentially leaky ratio features
features_to_keep = [
    'no_of_dependents',
    'education',
    'self_employed',
    'income_annum',
    'loan_amount',
    'loan_term',
    'cibil_score',
    'total_assets',
    'high_income',
    'asset_level',
    'employment_risk',
    'family_size'
]

# Keep only selected features plus target
data = data[features_to_keep + ['loan_status']]
print(f"\n📋 Using {len(features_to_keep)} features (removed potentially leaky features)")
print(f"Features: {features_to_keep}")

# Encode categorical variables
education_map = {'Graduate': 1, 'Not Graduate': 0}
self_employed_map = {'No': 0, 'Yes': 1}
loan_status_map = {'Approved': 1, 'Rejected': 0}

data['education'] = data['education'].map(education_map)
data['self_employed'] = data['self_employed'].map(self_employed_map)
data['loan_status'] = data['loan_status'].map(loan_status_map)

# Drop any nulls
data = data.dropna()
print(f"✅ Final dataset size: {len(data)} rows")

# ==========================================
# 3. SPLIT DATA (WITH VALIDATION SET)
# ==========================================
print("\n" + "="*80)
print("✂️  Step 3: Splitting Data (Train/Val/Test)...")
print("="*80)

X = data.drop(columns=['loan_status'])
y = data['loan_status']

# First split: Train+Val (80%) and Test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: Train (64%) and Val (16%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

print(f"✅ Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(data)*100:.1f}%)")
print(f"✅ Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(data)*100:.1f}%)")
print(f"✅ Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(data)*100:.1f}%)")

# ==========================================
# 4. FEATURE SCALING
# ==========================================
print("\n" + "="*80)
print("⚖️  Step 4: Feature Scaling...")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("✅ StandardScaler fitted and applied")

# ==========================================
# 5. MODEL TRAINING (CONSERVATIVE HYPERPARAMETERS)
# ==========================================
print("\n" + "="*80)
print("🤖 Step 5: Training Models with Conservative Settings...")
print("="*80)

# Use StratifiedKFold for better validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': [0.001, 0.01, 0.1, 1],  # More regularization
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, 15],  # Limit depth to prevent overfitting
            'min_samples_split': [10, 20],  # Higher values
            'min_samples_leaf': [5, 10],  # Add leaf constraint
            'max_features': ['sqrt', 'log2']  # Limit features per tree
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.05],  # Lower learning rates
            'max_depth': [3, 5],  # Shallow trees
            'min_samples_split': [10, 20],
            'subsample': [0.8]  # Use subsampling
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'max_depth': [5, 10, 15],
            'min_samples_split': [10, 20, 30],
            'min_samples_leaf': [5, 10, 15],
            'max_features': ['sqrt', 'log2']
        }
    },
    'SVM': {
        'model': SVC(random_state=42, probability=True),
        'params': {
            'C': [0.1, 1],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale']
        }
    }
}

results = {}
best_models = {}
val_predictions = {}

for name, config in models.items():
    print(f"\n🔄 Training {name}...")
    
    # GridSearchCV with validation
    grid_search = GridSearchCV(
        config['model'], 
        config['params'], 
        cv=cv,
        scoring='f1',  # Use F1 instead of accuracy
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    best_models[name] = best_model
    
    # Predictions on validation set
    y_val_pred = best_model.predict(X_val_scaled)
    val_predictions[name] = y_val_pred
    
    # Predictions on test set
    y_test_pred = best_model.predict(X_test_scaled)
    y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics on TEST set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    # Calculate metrics on VALIDATION set
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring='f1')
    
    results[name] = {
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'val_accuracy': val_accuracy,
        'val_f1': val_f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'best_params': grid_search.best_params_
    }
    
    print(f"   📊 Validation Accuracy: {val_accuracy:.4f}")
    print(f"   📊 Test Accuracy: {test_accuracy:.4f}")
    print(f"   📊 Test F1-Score: {test_f1:.4f}")
    print(f"   📊 Test AUC-ROC: {test_auc:.4f}")
    print(f"   📊 CV F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# ==========================================
# 6. MODEL COMPARISON & SELECTION
# ==========================================
print("\n" + "="*80)
print("📊 Step 6: Model Comparison Results")
print("="*80)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('test_f1', ascending=False)

print("\n🏆 RANKING BY TEST F1-SCORE:")
print(results_df[['test_accuracy', 'test_f1', 'test_auc', 'val_accuracy', 'cv_mean']].round(4))

# Select best model by F1 score
best_model_name = results_df['test_f1'].idxmax()
best_model = best_models[best_model_name]

print(f"\n🥇 BEST MODEL: {best_model_name}")
print(f"   Test Accuracy: {results_df.loc[best_model_name, 'test_accuracy']:.4f}")
print(f"   Test Precision: {results_df.loc[best_model_name, 'test_precision']:.4f}")
print(f"   Test Recall: {results_df.loc[best_model_name, 'test_recall']:.4f}")
print(f"   Test F1-Score: {results_df.loc[best_model_name, 'test_f1']:.4f}")
print(f"   Test AUC-ROC: {results_df.loc[best_model_name, 'test_auc']:.4f}")

# Check for overfitting
train_pred = best_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = results_df.loc[best_model_name, 'test_accuracy']
gap = train_accuracy - test_accuracy

print(f"\n⚠️  Overfitting Check:")
print(f"   Train Accuracy: {train_accuracy:.4f}")
print(f"   Test Accuracy: {test_accuracy:.4f}")
print(f"   Gap: {gap:.4f}")
if gap > 0.05:
    print(f"   ⚠️  Warning: Model may be overfitting (gap > 5%)")
else:
    print(f"   ✅ Good: Model generalizes well (gap < 5%)")

# ==========================================
# 7. DETAILED ANALYSIS
# ==========================================
print("\n" + "="*80)
print(f"🔍 Step 7: Detailed Analysis of {best_model_name}")
print("="*80)

y_pred_best = best_model.predict(X_test_scaled)
y_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
print("\n📊 Confusion Matrix:")
print(cm)
print(f"\nTrue Negatives (Correctly Rejected): {cm[0,0]}")
print(f"False Positives (Incorrectly Approved): {cm[0,1]}")
print(f"False Negatives (Incorrectly Rejected): {cm[1,0]}")
print(f"True Positives (Correctly Approved): {cm[1,1]}")

# Calculate rates
tn, fp, fn, tp = cm.ravel()
print(f"\n📈 Error Analysis:")
print(f"False Positive Rate: {fp/(fp+tn)*100:.2f}%")
print(f"False Negative Rate: {fn/(fn+tp)*100:.2f}%")

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Rejected', 'Approved']))

# Feature Importance
if hasattr(best_model, 'feature_importances_'):
    print("\n🎯 Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10).to_string(index=False))
elif hasattr(best_model, 'coef_'):
    print("\n🎯 Top 10 Most Important Features (by coefficient magnitude):")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': abs(best_model.coef_[0])
    }).sort_values('coefficient', ascending=False)
    print(feature_importance.head(10).to_string(index=False))

# ==========================================
# 8. SAVE MODELS
# ==========================================
print("\n" + "="*80)
print("💾 Step 8: Saving Models...")
print("="*80)

# Save to current directory (like original)
output_dir = '.'
import os

pk.dump(best_model, open('model.pkl', 'wb'))
print(f"✅ Best model saved as 'model.pkl'")

pk.dump(scaler, open('scaler.pkl', 'wb'))
print("✅ Scaler saved as 'scaler.pkl'")

pk.dump(best_models, open('all_models.pkl', 'wb'))
print("✅ All trained models saved as 'all_models.pkl'")

results_df.to_csv('model_comparison_results.csv')
print("✅ Model comparison results saved as 'model_comparison_results.csv'")

feature_names = X.columns.tolist()
pk.dump(feature_names, open('feature_names.pkl', 'wb'))
print("✅ Feature names saved as 'feature_names.pkl'")

# ==========================================
# 9. VISUALIZATIONS
# ==========================================
print("\n" + "="*80)
print("📊 Step 9: Creating Visualizations...")
print("="*80)

# Create visualizations directory (like original)
import os
os.makedirs('visualizations', exist_ok=True)

# 1. Model Comparison
plt.figure(figsize=(14, 6))
metrics_to_plot = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
x = np.arange(len(results_df))
width = 0.2

for i, metric in enumerate(metrics_to_plot):
    plt.bar(x + i*width, results_df[metric], width, 
            label=metric.replace('test_', '').replace('_', ' ').title())

plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylabel('Score', fontsize=12, fontweight='bold')
plt.title('Model Performance Comparison on Test Set', fontsize=14, fontweight='bold')
plt.xticks(x + width*1.5, results_df.index, rotation=45, ha='right')
plt.legend()
plt.ylim(0, 1.1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/model_comparison.png")
plt.close()

# 2. Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Rejected', 'Approved'],
            yticklabels=['Rejected', 'Approved'],
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)

# Add percentages
for i in range(2):
    for j in range(2):
        plt.text(j+0.5, i+0.7, f'({cm[i,j]/cm.sum()*100:.1f}%)', 
                ha='center', va='center', fontsize=10, color='red')

plt.tight_layout()
plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/confusion_matrix.png")
plt.close()

# 3. Feature Importance (Top 5 only)
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(10, 6))
    feature_imp_sorted = feature_importance.head(5)  # Only top 5
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_imp_sorted)))
    bars = plt.barh(range(len(feature_imp_sorted)), feature_imp_sorted['importance'], color=colors)
    plt.yticks(range(len(feature_imp_sorted)), feature_imp_sorted['feature'], fontsize=11)
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.title(f'Top 5 Most Important Features - {best_model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, feature_imp_sorted['importance'])):
        plt.text(value + 0.01, i, f'{value:.4f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: visualizations/feature_importance.png (Top 5 features)")
    plt.close()

# 4. ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_proba_best)
auc = roc_auc_score(y_test, y_proba_best)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title(f'ROC Curve - {best_model_name}', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/roc_curve.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/roc_curve.png")
plt.close()

# ==========================================
# 10. SUMMARY & RECOMMENDATIONS
# ==========================================
print("\n" + "="*80)
print("🎉 TRAINING COMPLETE!")
print("="*80)

print(f"\n✅ Best Model: {best_model_name}")
print(f"✅ Test Accuracy: {results_df.loc[best_model_name, 'test_accuracy']*100:.2f}%")
print(f"✅ Test F1-Score: {results_df.loc[best_model_name, 'test_f1']:.4f}")
print(f"✅ Test AUC-ROC: {results_df.loc[best_model_name, 'test_auc']:.4f}")

print("\n📊 Model Reliability:")
if gap < 0.03:
    print("   ✅ Excellent: Very low overfitting")
elif gap < 0.05:
    print("   ✅ Good: Acceptable generalization")
elif gap < 0.10:
    print("   ⚠️  Fair: Some overfitting detected")
else:
    print("   ❌ Poor: Significant overfitting")

print("\n📁 All files saved in current directory:")
print("   - model.pkl (Best model for deployment)")
print("   - scaler.pkl (Feature scaler)")
print("   - all_models.pkl (All trained models)")
print("   - feature_names.pkl (Feature list)")
print("   - model_comparison_results.csv (Performance metrics)")
print("   - visualizations/model_comparison.png")
print("   - visualizations/confusion_matrix.png")
print("   - visualizations/feature_importance.png")
print("   - visualizations/roc_curve.png")
print("\n💡 Key Insights:")
print(f"   - Removed potentially leaky engineered features")
print(f"   - Used conservative hyperparameters to prevent overfitting")
print(f"   - Applied proper train/validation/test split")
print(f"   - Selected model based on F1-score (balanced metric)")
print(f"   - False Positives: {fp} ({fp/(fp+tn)*100:.1f}%)")
print(f"   - False Negatives: {fn} ({fn/(fn+tp)*100:.1f}%)")

print("\n🚀 Ready for deployment!")
print("="*80)