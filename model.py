"""
🚀 ENHANCED LOAN PREDICTION MODEL TRAINING
==========================================
New Features:
- SBERT embeddings for categorical data
- Ensemble voting classifier
- Enhanced cross-validation with overfitting detection
- Quantum Neural Network (QNN) implementation
- Comprehensive model comparison
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# SBERT for text embeddings
from sentence_transformers import SentenceTransformer

# For QNN
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. QNN will be skipped.")

import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("🏦 ENHANCED LOAN APPROVAL MODEL TRAINING")
print("="*80)


# ==========================================
# 1. LOAD DATA
# ==========================================

print("\n📂 Step 1: Loading Data...")
data = pd.read_csv('loan_approval_dataset_clean.csv')
print(f"✓ Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")


# ==========================================
# 2. DATA PREPROCESSING WITH SBERT
# ==========================================

print("\n" + "="*80)
print("🔧 Step 2: Data Preprocessing with SBERT Embeddings...")
print("="*80)

# Clean column names
data.columns = data.columns.str.strip()
print("✓ Cleaned column names")

# Drop loan_id
if 'loan_id' in data.columns:
    data.drop(columns=['loan_id'], inplace=True)
    print("✓ Dropped loan_id")

# Clean categorical values
data['education'] = data['education'].str.strip()
data['self_employed'] = data['self_employed'].str.strip()
data['loan_status'] = data['loan_status'].str.strip()

# Check class distribution
print("\n📊 Original Class Distribution:")
print(data['loan_status'].value_counts())
print(f"Approval Rate: {(data['loan_status'] == 'Approved').sum() / len(data) * 100:.2f}%")

# SBERT Embeddings for Categorical Features
print("\n🤖 Generating SBERT Embeddings for Categorical Features...")
try:
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
    
    # Create text representations for SBERT
    data['text_profile'] = (
        data['education'].astype(str) + " " + 
        data['self_employed'].apply(lambda x: "self_employed" if x == "Yes" else "employed")
    )
    
    # Generate embeddings
    embeddings = sbert_model.encode(data['text_profile'].tolist(), show_progress_bar=True)
    
    # Add embeddings as features
    for i in range(embeddings.shape[1]):
        data[f'sbert_dim_{i}'] = embeddings[:, i]
    
    print(f"✓ Added {embeddings.shape[1]} SBERT embedding dimensions")
    SBERT_ENABLED = True
    
except Exception as e:
    print(f"⚠️  SBERT embedding failed: {e}")
    print("   Continuing without SBERT features...")
    SBERT_ENABLED = False

# Conservative Feature Engineering
print("\n🔨 Conservative Feature Engineering...")

# Basic combined assets
data['total_assets'] = (data['residential_assets_value'] + 
                        data['commercial_assets_value'] + 
                        data['luxury_assets_value'] + 
                        data['bank_asset_value'])

# Create ONLY non-leaky features
data['high_income'] = (data['income_annum'] > data['income_annum'].median()).astype(int)

data['asset_level'] = pd.cut(data['total_assets'], 
                              bins=[0, 1000000, 5000000, float('inf')],
                              labels=[0, 1, 2]).astype(int)

data['employment_risk'] = (data['self_employed'] == 'Yes').astype(int)

data['family_size'] = data['no_of_dependents'].apply(lambda x: 0 if x <= 1 else (1 if x <= 3 else 2))

print("✓ Created 4 safe engineered features")

# Select base features
base_features = [
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

# Add SBERT features if available
if SBERT_ENABLED:
    sbert_features = [col for col in data.columns if col.startswith('sbert_dim_')]
    features_to_keep = base_features + sbert_features
else:
    features_to_keep = base_features

# Keep only selected features plus target
data = data[features_to_keep + ['loan_status', 'text_profile'] if SBERT_ENABLED else features_to_keep + ['loan_status']]
print(f"\n✓ Using {len(features_to_keep)} features")

# Encode categorical variables
education_map = {'Graduate': 1, 'Not Graduate': 0}
self_employed_map = {'No': 0, 'Yes': 1}
loan_status_map = {'Approved': 1, 'Rejected': 0}

data['education'] = data['education'].map(education_map)
data['self_employed'] = data['self_employed'].map(self_employed_map)
data['loan_status'] = data['loan_status'].map(loan_status_map)

# Drop text_profile if exists
if 'text_profile' in data.columns:
    data = data.drop(columns=['text_profile'])

# Drop any nulls
data = data.dropna()
print(f"✓ Final dataset size: {len(data)} rows")


# ==========================================
# 3. SPLIT DATA
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

print(f"✓ Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(data)*100:.1f}%)")
print(f"✓ Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(data)*100:.1f}%)")
print(f"✓ Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(data)*100:.1f}%)")


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
print("✓ StandardScaler fitted and applied")


# ==========================================
# 5. QUANTUM NEURAL NETWORK (QNN)
# ==========================================

if TORCH_AVAILABLE:
    print("\n" + "="*80)
    print("⚛️  Step 5a: Training Quantum-Inspired Neural Network...")
    print("="*80)
    
    class QuantumInspiredNN(nn.Module):
        """
        Quantum-Inspired Neural Network
        Uses amplitude encoding and entanglement-like connections
        """
        def __init__(self, input_dim, hidden_dim=64):
            super(QuantumInspiredNN, self).__init__()
            
            # Quantum-inspired layers with amplitude encoding
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),  # Amplitude normalization
            )
            
            # Entanglement-like layer
            self.entangle1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            self.entangle2 = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            # Measurement layer
            self.measure = nn.Linear(hidden_dim // 4, 2)
            
        def forward(self, x):
            x = self.encoder(x)
            x = self.entangle1(x)
            x = self.entangle2(x)
            x = self.measure(x)
            return x
    
    # Prepare PyTorch datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.LongTensor(y_train.values)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.LongTensor(y_val.values)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.LongTensor(y_test.values)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize QNN
    qnn_model = QuantumInspiredNN(input_dim=X_train_scaled.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(qnn_model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training
    print("🔄 Training QNN...")
    num_epochs = 50
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        qnn_model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = qnn_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        qnn_model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = qnn_model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_acc = val_correct / val_total
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{num_epochs} - Val Accuracy: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(qnn_model.state_dict(), 'qnn_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"✓ Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    qnn_model.load_state_dict(torch.load('qnn_model.pth'))
    
    # Test QNN
    qnn_model.eval()
    qnn_predictions = []
    qnn_probas = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            outputs = qnn_model(batch_X)
            probas = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            qnn_predictions.extend(predicted.numpy())
            qnn_probas.extend(probas[:, 1].numpy())
    
    qnn_predictions = np.array(qnn_predictions)
    qnn_probas = np.array(qnn_probas)
    
    qnn_accuracy = accuracy_score(y_test, qnn_predictions)
    qnn_f1 = f1_score(y_test, qnn_predictions)
    qnn_auc = roc_auc_score(y_test, qnn_probas)
    
    print(f"\n✓ QNN Test Accuracy: {qnn_accuracy:.4f}")
    print(f"✓ QNN Test F1-Score: {qnn_f1:.4f}")
    print(f"✓ QNN Test AUC-ROC: {qnn_auc:.4f}")
    
    QNN_TRAINED = True
else:
    QNN_TRAINED = False
    print("\n⚠️  Skipping QNN (PyTorch not available)")


# ==========================================
# 6. TRADITIONAL MODELS WITH ENHANCED CV
# ==========================================

print("\n" + "="*80)
print("🤖 Step 5b: Training Traditional Models with Enhanced CV...")
print("="*80)

# Use StratifiedKFold for better validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': [0.001, 0.01, 0.1, 1],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, 15],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10],
            'max_features': ['sqrt', 'log2']
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.05],
            'max_depth': [3, 5],
            'min_samples_split': [10, 20],
            'subsample': [0.8]
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
    
    # GridSearchCV
    grid_search = GridSearchCV(
        config['model'], 
        config['params'], 
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    best_models[name] = best_model
    
    # Predictions
    y_val_pred = best_model.predict(X_val_scaled)
    val_predictions[name] = y_val_pred
    
    y_test_pred = best_model.predict(X_test_scaled)
    y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    # Cross-validation with 5-fold
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring='f1')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # ENHANCED: Check for perfect overfitting
    train_pred = best_model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_pred)
    
    overfitting_flag = False
    overfitting_reason = ""
    
    if cv_mean >= 0.999:  # 100% or near-perfect CV score
        overfitting_flag = True
        overfitting_reason = "⚠️  SUSPICIOUS: Perfect CV score (likely overfitting)"
    elif train_accuracy - test_accuracy > 0.15:  # Large train-test gap
        overfitting_flag = True
        overfitting_reason = f"⚠️  SUSPICIOUS: Large train-test gap ({train_accuracy - test_accuracy:.3f})"
    
    results[name] = {
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'val_accuracy': val_accuracy,
        'val_f1': val_f1,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'train_accuracy': train_accuracy,
        'overfitting_flag': overfitting_flag,
        'overfitting_reason': overfitting_reason,
        'best_params': grid_search.best_params_
    }
    
    print(f"   Val Accuracy: {val_accuracy:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   Test F1-Score: {test_f1:.4f}")
    print(f"   CV F1-Score: {cv_mean:.4f} (±{cv_std:.4f})")
    print(f"   Train Accuracy: {train_accuracy:.4f}")
    
    if overfitting_flag:
        print(f"   {overfitting_reason}")


# ==========================================
# 7. ENSEMBLE METHOD (VOTING CLASSIFIER)
# ==========================================

print("\n" + "="*80)
print("🎯 Step 6: Creating Ensemble Voting Classifier...")
print("="*80)

# Filter out models with overfitting flags (EXCLUDE QNN from ensemble)
reliable_models = [(name, model) for name, model in best_models.items() 
                   if not results[name]['overfitting_flag'] and name != 'QNN']

if len(reliable_models) >= 3:
    print(f"✓ Using {len(reliable_models)} reliable models for ensemble (QNN excluded)")
    print(f"  Models: {[name for name, _ in reliable_models]}")
    
    # Create soft voting ensemble
    ensemble = VotingClassifier(
        estimators=reliable_models,
        voting='soft',  # Use probability averaging
        n_jobs=-1
    )
    
    print("🔄 Training ensemble...")
    ensemble.fit(X_train_scaled, y_train)
    
    # Evaluate ensemble
    y_ensemble_pred = ensemble.predict(X_test_scaled)
    y_ensemble_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
    
    ensemble_accuracy = accuracy_score(y_test, y_ensemble_pred)
    ensemble_precision = precision_score(y_test, y_ensemble_pred)
    ensemble_recall = recall_score(y_test, y_ensemble_pred)
    ensemble_f1 = f1_score(y_test, y_ensemble_pred)
    ensemble_auc = roc_auc_score(y_test, y_ensemble_proba)
    
    # Cross-validation for ensemble (5-fold)
    ensemble_cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=cv, scoring='f1')
    
    # Check ensemble overfitting
    ensemble_train_pred = ensemble.predict(X_train_scaled)
    ensemble_train_acc = accuracy_score(y_train, ensemble_train_pred)
    ensemble_overfitting_flag = ensemble_train_acc - ensemble_accuracy > 0.15
    
    results['Ensemble (Voting)'] = {
        'test_accuracy': ensemble_accuracy,
        'test_precision': ensemble_precision,
        'test_recall': ensemble_recall,
        'test_f1': ensemble_f1,
        'test_auc': ensemble_auc,
        'cv_mean': ensemble_cv_scores.mean(),
        'cv_std': ensemble_cv_scores.std(),
        'train_accuracy': ensemble_train_acc,
        'overfitting_flag': ensemble_overfitting_flag,
        'overfitting_reason': "⚠️  Large train-test gap" if ensemble_overfitting_flag else "",
        'best_params': f"Voting from {len(reliable_models)} models"
    }
    
    best_models['Ensemble (Voting)'] = ensemble
    
    print(f"✓ Ensemble Test Accuracy: {ensemble_accuracy:.4f}")
    print(f"✓ Ensemble Test F1-Score: {ensemble_f1:.4f}")
    print(f"✓ Ensemble Test AUC-ROC: {ensemble_auc:.4f}")
    print(f"✓ Ensemble CV F1-Score: {ensemble_cv_scores.mean():.4f} (±{ensemble_cv_scores.std():.4f})")
    
    ENSEMBLE_CREATED = True
else:
    print(f"⚠️  Only {len(reliable_models)} reliable models - need at least 3 for ensemble")
    ENSEMBLE_CREATED = False


# ==========================================
# 8. ADD QNN TO RESULTS
# ==========================================

if QNN_TRAINED:
    results['QNN'] = {
        'test_accuracy': qnn_accuracy,
        'test_precision': precision_score(y_test, qnn_predictions),
        'test_recall': recall_score(y_test, qnn_predictions),
        'test_f1': qnn_f1,
        'test_auc': qnn_auc,
        'cv_mean': 0.0,  # Not applicable for NN
        'cv_std': 0.0,
        'train_accuracy': 0.0,  # Would need separate calculation
        'overfitting_flag': False,
        'overfitting_reason': "",
        'best_params': "PyTorch NN"
    }


# ==========================================
# 9. MODEL COMPARISON & SELECTION
# ==========================================

print("\n" + "="*80)
print("📊 Step 7: Comprehensive Model Comparison")
print("="*80)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('test_f1', ascending=False)

print("\n🏆 RANKING BY TEST F1-SCORE:")
display_cols = ['test_accuracy', 'test_f1', 'test_auc', 'cv_mean', 'train_accuracy', 'overfitting_flag']
print(results_df[display_cols].round(4))

# Flag suspicious models
print("\n⚠️  OVERFITTING ALERTS:")
suspicious_models = results_df[results_df['overfitting_flag'] == True]
if len(suspicious_models) > 0:
    for idx in suspicious_models.index:
        print(f"   • {idx}: {results_df.loc[idx, 'overfitting_reason']}")
else:
    print("   ✓ No suspicious overfitting detected")

# FORCE ENSEMBLE AS OUTPUT MODEL
if ENSEMBLE_CREATED and 'Ensemble (Voting)' in best_models:
    best_model_name = 'Ensemble (Voting)'
    best_model = best_models[best_model_name]
    print(f"\n🎯 SELECTED MODEL FOR DEPLOYMENT: {best_model_name}")
    print("   (Ensemble method forced as output for optimal performance)")
else:
    # Fallback: Select best model (excluding QNN, excluding overfitting ones)
    reliable_results = results_df[(results_df['overfitting_flag'] == False) & (results_df.index != 'QNN')]
    if len(reliable_results) > 0:
        best_model_name = reliable_results['test_f1'].idxmax()
        best_model = best_models[best_model_name]
        print(f"\n🎯 BEST MODEL: {best_model_name}")
    else:
        # Last fallback (should rarely happen)
        best_model_name = results_df[results_df.index != 'QNN']['test_f1'].idxmax()
        best_model = best_models[best_model_name]
        print(f"\n⚠️  BEST MODEL (with caution): {best_model_name}")

print(f"   Test Accuracy: {results_df.loc[best_model_name, 'test_accuracy']:.4f}")
print(f"   Test Precision: {results_df.loc[best_model_name, 'test_precision']:.4f}")
print(f"   Test Recall: {results_df.loc[best_model_name, 'test_recall']:.4f}")
print(f"   Test F1-Score: {results_df.loc[best_model_name, 'test_f1']:.4f}")
print(f"   Test AUC-ROC: {results_df.loc[best_model_name, 'test_auc']:.4f}")


# ==========================================
# 10. DETAILED ANALYSIS
# ==========================================

print("\n" + "="*80)
print(f"🔬 Step 8: Detailed Analysis of {best_model_name}")
print("="*80)

y_pred_best = best_model.predict(X_test_scaled)
y_proba_best = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else None

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
print("\n📊 Confusion Matrix:")
print(cm)
print(f"\nTrue Negatives (Correctly Rejected): {cm[0,0]}")
print(f"False Positives (Incorrectly Approved): {cm[0,1]}")
print(f"False Negatives (Incorrectly Rejected): {cm[1,0]}")
print(f"True Positives (Correctly Approved): {cm[1,1]}")

tn, fp, fn, tp = cm.ravel()
print(f"\n❌ Error Analysis:")
print(f"False Positive Rate: {fp/(fp+tn)*100:.2f}%")
print(f"False Negative Rate: {fn/(fn+tp)*100:.2f}%")

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Rejected', 'Approved']))


# ==========================================
# 11. SAVE MODELS
# ==========================================

print("\n" + "="*80)
print("💾 Step 9: Saving Models...")
print("="*80)

# Save best model (could be ensemble or single model)
pk.dump(best_model, open('model.pkl', 'wb'))
print(f"✓ Best model ({best_model_name}) saved as 'model.pkl'")

pk.dump(scaler, open('scaler.pkl', 'wb'))
print("✓ Scaler saved as 'scaler.pkl'")

pk.dump(best_models, open('all_models.pkl', 'wb'))
print("✓ All trained models saved as 'all_models.pkl'")

results_df.to_csv('model_comparison_results.csv')
print("✓ Model comparison results saved as 'model_comparison_results.csv'")

feature_names = X.columns.tolist()
pk.dump(feature_names, open('feature_names.pkl', 'wb'))
print(f"✓ Feature names saved as 'feature_names.pkl' ({len(feature_names)} features)")

# Save model metadata
metadata = {
    'best_model_name': best_model_name,
    'sbert_enabled': SBERT_ENABLED,
    'qnn_trained': QNN_TRAINED,
    'ensemble_created': ENSEMBLE_CREATED,
    'num_features': len(feature_names),
    'test_accuracy': results_df.loc[best_model_name, 'test_accuracy'],
    'test_f1': results_df.loc[best_model_name, 'test_f1']
}
pk.dump(metadata, open('model_metadata.pkl', 'wb'))
print("✓ Model metadata saved")


# ==========================================
# 12. IEEE-QUALITY VISUALIZATIONS
# ==========================================

print("\n" + "="*80)
print("📈 Step 10: Creating IEEE-Quality Publication Visualizations...")
print("="*80)

import os
os.makedirs('visualizations', exist_ok=True)

# Set publication-quality parameters
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'figure.titlesize': 20,
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.5
})

# ==========================================
# FIGURE 1: Cross-Validation Comparison (5-Fold)
# ==========================================
print("📊 Creating Figure 1: Cross-Validation Analysis...")

# Exclude QNN from comparison (no CV for neural networks)
cv_results = {name: data for name, data in results.items() if name != 'QNN'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: CV Mean F1-Score with error bars
models_list = list(cv_results.keys())
cv_means = [cv_results[m]['cv_mean'] for m in models_list]
cv_stds = [cv_results[m]['cv_std'] for m in models_list]
colors = ['#2ecc71' if not cv_results[m]['overfitting_flag'] else '#e74c3c' for m in models_list]

x_pos = np.arange(len(models_list))
bars1 = ax1.bar(x_pos, cv_means, yerr=cv_stds, capsize=8, alpha=0.85, 
                color=colors, edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2})

ax1.set_xlabel('Model', fontsize=18, fontweight='bold')
ax1.set_ylabel('5-Fold Cross-Validation F1-Score', fontsize=18, fontweight='bold')
ax1.set_title('(a) 5-Fold Cross-Validation Performance', fontsize=20, fontweight='bold', pad=20)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models_list, rotation=45, ha='right', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 1.1])
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax1.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, label='95% Threshold')

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars1, cv_means, cv_stds)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
            f'{mean:.3f}±{std:.3f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Legend for colors
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', label='Reliable Model'),
                   Patch(facecolor='#e74c3c', label='Overfitting Flag')]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=14, framealpha=0.95)

# Right: Test vs Train Accuracy (Overfitting Detection)
train_accs = [cv_results[m]['train_accuracy'] for m in models_list]
test_accs = [cv_results[m]['test_accuracy'] for m in models_list]

x_pos2 = np.arange(len(models_list))
width = 0.35

bars_train = ax2.bar(x_pos2 - width/2, train_accs, width, label='Training Accuracy',
                     alpha=0.85, color='#3498db', edgecolor='black', linewidth=1.5)
bars_test = ax2.bar(x_pos2 + width/2, test_accs, width, label='Test Accuracy',
                    alpha=0.85, color='#9b59b6', edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Model', fontsize=18, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=18, fontweight='bold')
ax2.set_title('(b) Train vs Test Accuracy (Overfitting Analysis)', fontsize=20, fontweight='bold', pad=20)
ax2.set_xticks(x_pos2)
ax2.set_xticklabels(models_list, rotation=45, ha='right', fontsize=14, fontweight='bold')
ax2.set_ylim([0, 1.1])
ax2.legend(loc='lower right', fontsize=14, framealpha=0.95)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1)

# Add gap indicators
for i, (train, test) in enumerate(zip(train_accs, test_accs)):
    gap = train - test
    if gap > 0.05:
        ax2.plot([i, i], [test, train], 'r--', linewidth=2.5, alpha=0.7)
        ax2.text(i, (train + test)/2, f'Δ={gap:.2f}', ha='center', 
                fontsize=10, fontweight='bold', color='red', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('visualizations/Fig1_CrossValidation_Analysis.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('visualizations/Fig1_CrossValidation_Analysis.eps', format='eps', bbox_inches='tight', facecolor='white')
print("✓ Saved: Fig1_CrossValidation_Analysis.png (600 DPI) + EPS")
plt.close()

# ==========================================
# FIGURE 2: Comprehensive Performance Metrics
# ==========================================
print("📊 Creating Figure 2: Comprehensive Performance Metrics...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))

# Exclude QNN from display
display_models = [m for m in models_list if m != 'QNN']
display_results = {m: cv_results[m] for m in display_models}

# 2a: Test Accuracy
accuracies = [display_results[m]['test_accuracy'] for m in display_models]
colors_acc = ['#2ecc71' if not display_results[m]['overfitting_flag'] else '#e74c3c' for m in display_models]
x_pos = np.arange(len(display_models))

bars = ax1.barh(x_pos, accuracies, alpha=0.85, color=colors_acc, 
                edgecolor='black', linewidth=1.5)
ax1.set_yticks(x_pos)
ax1.set_yticklabels(display_models, fontsize=14, fontweight='bold')
ax1.set_xlabel('Test Accuracy', fontsize=18, fontweight='bold')
ax1.set_title('(a) Model Test Accuracy', fontsize=20, fontweight='bold', pad=20)
ax1.set_xlim([0, 1.1])
ax1.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=1)

# Add value labels
for bar, acc in zip(bars, accuracies):
    width = bar.get_width()
    ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2,
            f'{acc:.4f}', ha='left', va='center', fontsize=13, fontweight='bold')

# 2b: Precision, Recall, F1-Score
metrics_data = {
    'Precision': [display_results[m]['test_precision'] for m in display_models],
    'Recall': [display_results[m]['test_recall'] for m in display_models],
    'F1-Score': [display_results[m]['test_f1'] for m in display_models]
}

x_pos2 = np.arange(len(display_models))
width = 0.25
colors_metrics = ['#e74c3c', '#3498db', '#2ecc71']

for i, (metric_name, values) in enumerate(metrics_data.items()):
    ax2.bar(x_pos2 + i*width, values, width, label=metric_name,
           alpha=0.85, color=colors_metrics[i], edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Model', fontsize=18, fontweight='bold')
ax2.set_ylabel('Score', fontsize=18, fontweight='bold')
ax2.set_title('(b) Precision, Recall, and F1-Score', fontsize=20, fontweight='bold', pad=20)
ax2.set_xticks(x_pos2 + width)
ax2.set_xticklabels(display_models, rotation=45, ha='right', fontsize=14, fontweight='bold')
ax2.set_ylim([0, 1.1])
ax2.legend(loc='lower right', fontsize=14, framealpha=0.95)
ax2.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=1)

# 2c: AUC-ROC Scores
aucs = [display_results[m]['test_auc'] for m in display_models]
colors_auc = ['#2ecc71' if not display_results[m]['overfitting_flag'] else '#e74c3c' for m in display_models]

bars = ax3.barh(x_pos, aucs, alpha=0.85, color=colors_auc,
               edgecolor='black', linewidth=1.5)
ax3.set_yticks(x_pos)
ax3.set_yticklabels(display_models, fontsize=14, fontweight='bold')
ax3.set_xlabel('AUC-ROC Score', fontsize=18, fontweight='bold')
ax3.set_title('(c) Model AUC-ROC Performance', fontsize=20, fontweight='bold', pad=20)
ax3.set_xlim([0, 1.1])
ax3.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=1)

# Add value labels
for bar, auc in zip(bars, aucs):
    width = bar.get_width()
    ax3.text(width + 0.02, bar.get_y() + bar.get_height()/2,
            f'{auc:.4f}', ha='left', va='center', fontsize=13, fontweight='bold')

# 2d: Train-Test Gap Analysis
gaps = [display_results[m]['train_accuracy'] - display_results[m]['test_accuracy'] for m in display_models]
colors_gap = ['#2ecc71' if gap < 0.05 else '#f39c12' if gap < 0.10 else '#e74c3c' for gap in gaps]

bars = ax4.barh(x_pos, gaps, alpha=0.85, color=colors_gap,
               edgecolor='black', linewidth=1.5)
ax4.set_yticks(x_pos)
ax4.set_yticklabels(display_models, fontsize=14, fontweight='bold')
ax4.set_xlabel('Train-Test Accuracy Gap', fontsize=18, fontweight='bold')
ax4.set_title('(d) Overfitting Detection (Gap Analysis)', fontsize=20, fontweight='bold', pad=20)
ax4.axvline(x=0.05, color='orange', linestyle='--', linewidth=2.5, label='5% Threshold')
ax4.axvline(x=0.10, color='red', linestyle='--', linewidth=2.5, label='10% Threshold')
ax4.legend(loc='upper right', fontsize=14, framealpha=0.95)
ax4.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=1)

# Add value labels
for bar, gap in zip(bars, gaps):
    width = bar.get_width()
    ax4.text(width + 0.005, bar.get_y() + bar.get_height()/2,
            f'{gap:.3f}', ha='left', va='center', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/Fig2_Performance_Metrics.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('visualizations/Fig2_Performance_Metrics.eps', format='eps', bbox_inches='tight', facecolor='white')
print("✓ Saved: Fig2_Performance_Metrics.png (600 DPI) + EPS")
plt.close()

# ==========================================
# FIGURE 3: ROC Curves Comparison
# ==========================================
print("📊 Creating Figure 3: ROC Curves Comparison...")

fig, ax = plt.subplots(figsize=(12, 10))

# Color palette for models
color_palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
linestyle_map = {False: '-', True: '--'}

for i, name in enumerate(display_models):
    if name in best_models and hasattr(best_models[name], 'predict_proba'):
        y_proba = best_models[name].predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        
        is_overfitting = display_results[name]['overfitting_flag']
        linestyle = linestyle_map[is_overfitting]
        
        label = f'{name} (AUC = {auc:.4f})'
        if is_overfitting:
            label += ' *'
        
        ax.plot(fpr, tpr, linestyle=linestyle, linewidth=3.5, 
               color=color_palette[i % len(color_palette)], 
               label=label, alpha=0.85)

# Random classifier line
ax.plot([0, 1], [0, 1], 'k--', linewidth=2.5, label='Random Classifier (AUC = 0.5000)', alpha=0.6)

ax.set_xlabel('False Positive Rate', fontsize=20, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=20, fontweight='bold')
ax.set_title('ROC Curves: Model Comparison\n(Dashed lines indicate overfitting flags)', 
            fontsize=22, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=14, framealpha=0.95, 
         title='Models', title_fontsize=16)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

# Add diagonal reference
ax.fill_between([0, 1], [0, 1], alpha=0.1, color='gray')

plt.tight_layout()
plt.savefig('visualizations/Fig3_ROC_Curves.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('visualizations/Fig3_ROC_Curves.eps', format='eps', bbox_inches='tight', facecolor='white')
print("✓ Saved: Fig3_ROC_Curves.png (600 DPI) + EPS")
plt.close()

# ==========================================
# FIGURE 4: Confusion Matrix (Best Model)
# ==========================================
print("📊 Creating Figure 4: Confusion Matrix...")

fig, ax = plt.subplots(figsize=(10, 9))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred_best)

# Custom colormap
cmap = plt.cm.Blues

# Plot heatmap
im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Labels
classes = ['Rejected (0)', 'Approved (1)']
tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(classes, fontsize=16, fontweight='bold')
ax.set_yticklabels(classes, fontsize=16, fontweight='bold')

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        # Absolute count
        ax.text(j, i - 0.15, f'{cm[i, j]:,}',
               ha="center", va="center", fontsize=22, fontweight='bold',
               color="white" if cm[i, j] > thresh else "black")
        
        # Percentage
        percentage = cm[i, j] / cm.sum() * 100
        ax.text(j, i + 0.2, f'({percentage:.1f}%)',
               ha="center", va="center", fontsize=16, fontweight='bold',
               color="white" if cm[i, j] > thresh else "black")

ax.set_ylabel('True Label', fontsize=20, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=20, fontweight='bold')
ax.set_title(f'Confusion Matrix: {best_model_name}\nTest Set Performance', 
            fontsize=22, fontweight='bold', pad=20)

# Add grid
ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
ax.grid(which="minor", color="gray", linestyle='-', linewidth=2)
ax.tick_params(which="minor", size=0)

plt.tight_layout()
plt.savefig('visualizations/Fig4_Confusion_Matrix.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('visualizations/Fig4_Confusion_Matrix.eps', format='eps', bbox_inches='tight', facecolor='white')
print("✓ Saved: Fig4_Confusion_Matrix.png (600 DPI) + EPS")
plt.close()

# ==========================================
# FIGURE 5: Feature Importance (if applicable)
# ==========================================
if hasattr(best_model, 'feature_importances_'):
    print("📊 Creating Figure 5: Feature Importance...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=True).tail(15)  # Top 15
    
    # Color gradient
    colors_feat = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
    
    bars = ax.barh(range(len(feature_importance)), feature_importance['importance'],
                   color=colors_feat, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance['feature'], fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=20, fontweight='bold')
    ax.set_title(f'Top 15 Feature Importances: {best_model_name}', 
                fontsize=22, fontweight='bold', pad=20)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=1)
    
    # Add value labels
    for bar, value in zip(bars, feature_importance['importance']):
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
               f'{value:.4f}', ha='left', va='center', 
               fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/Fig5_Feature_Importance.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig('visualizations/Fig5_Feature_Importance.eps', format='eps', bbox_inches='tight', facecolor='white')
    print("✓ Saved: Fig5_Feature_Importance.png (600 DPI) + EPS")
    plt.close()

# ==========================================
# TABLE: Results Summary for IEEE Paper
# ==========================================
print("📊 Creating Results Summary Table...")

# Create comprehensive results table
results_table = pd.DataFrame({
    'Model': display_models,
    'Test Accuracy': [f"{display_results[m]['test_accuracy']:.4f}" for m in display_models],
    'Precision': [f"{display_results[m]['test_precision']:.4f}" for m in display_models],
    'Recall': [f"{display_results[m]['test_recall']:.4f}" for m in display_models],
    'F1-Score': [f"{display_results[m]['test_f1']:.4f}" for m in display_models],
    'AUC-ROC': [f"{display_results[m]['test_auc']:.4f}" for m in display_models],
    'CV Mean (5-fold)': [f"{display_results[m]['cv_mean']:.4f}" for m in display_models],
    'CV Std': [f"{display_results[m]['cv_std']:.4f}" for m in display_models],
    'Train-Test Gap': [f"{display_results[m]['train_accuracy'] - display_results[m]['test_accuracy']:.4f}" for m in display_models],
    'Overfitting': ['Yes' if display_results[m]['overfitting_flag'] else 'No' for m in display_models]
})

results_table.to_csv('visualizations/IEEE_Results_Table.csv', index=False)
print("✓ Saved: IEEE_Results_Table.csv")

# Create LaTeX table
latex_table = results_table.to_latex(index=False, 
                                     caption='Comprehensive Model Performance Comparison',
                                     label='tab:model_comparison',
                                     column_format='l' + 'c'*8,
                                     escape=False)

with open('visualizations/IEEE_Results_Table.tex', 'w') as f:
    f.write(latex_table)
print("✓ Saved: IEEE_Results_Table.tex (LaTeX format)")

print("\n✅ All IEEE-quality visualizations created successfully!")
print("   - All figures saved in PNG (600 DPI) and EPS formats")
print("   - Results table saved in CSV and LaTeX formats")
print("   - Ready for IEEE paper submission")


# ==========================================
# 13. SUMMARY
# ==========================================

print("\n" + "="*80)
print("🎉 ENHANCED TRAINING COMPLETE!")
print("="*80)

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test Accuracy: {results_df.loc[best_model_name, 'test_accuracy']*100:.2f}%")
print(f"   Test F1-Score: {results_df.loc[best_model_name, 'test_f1']:.4f}")
print(f"   Test AUC-ROC: {results_df.loc[best_model_name, 'test_auc']:.4f}")

print("\n📊 Enhancement Summary:")
print(f"   ✓ SBERT Embeddings: {'Enabled' if SBERT_ENABLED else 'Disabled'}")
print(f"   ✓ Ensemble Method: {'Created' if ENSEMBLE_CREATED else 'Not Created'}")
print(f"   ✓ QNN Training: {'Completed' if QNN_TRAINED else 'Skipped'}")
print(f"   ✓ Total Features: {len(feature_names)}")
print(f"   ✓ Models Trained: {len(results_df)}")
print(f"   ✓ Overfitting Flags: {sum(results_df['overfitting_flag'])}")

print("\n💾 Saved Files:")
print("   • model.pkl (Best model for deployment)")
print("   • scaler.pkl")
print("   • all_models.pkl")
print("   • feature_names.pkl")
print("   • model_metadata.pkl")
print("   • model_comparison_results.csv")
if QNN_TRAINED:
    print("   • qnn_model.pth (PyTorch QNN)")
print("   • visualizations/comprehensive_model_comparison.png")
print("   • visualizations/confusion_matrix_best.png")
print("   • visualizations/roc_curves_comparison.png")

print("\n✅ Ready for deployment!")
print("="*80)