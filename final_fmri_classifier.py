# ==================== final_fmri_classifier.py ====================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
output_dir = '/home/alirezarahi/ai-med-env/brain/session1/output'
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("Final fMRI Binary Classification Model")
print("="*70)

# 1. Load data
df = pd.read_csv("fmri_analysis_results/all_features.csv")

# Select only 2 main classes
target_classes = ['Response_Control', 'Correct_Task']
df_binary = df[df['trial_type'].isin(target_classes)].copy()

print(f"Binary data: {len(df_binary)} samples")
print(f"Classes: {target_classes[0]} vs {target_classes[1]}")

# 2. Data preparation
non_feature_cols = ['trial_type', 'onset', 'event_idx', 'subject', 'session', 'task']
feature_cols = [col for col in df_binary.columns if col not in non_feature_cols]

X = df_binary[feature_cols].values
y = df_binary['trial_type'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nData shape: X={X.shape}, y={y_encoded.shape}")
print(f"Class distribution:")
for i, cls in enumerate(le.classes_):
    count = sum(y_encoded == i)
    print(f"  {cls}: {count} samples ({count/len(y_encoded)*100:.1f}%)")

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, stratify=y_encoded, random_state=42
)

print(f"\nData split:")
print(f"  Training: {X_train.shape[0]} samples")
print(f"  Testing: {X_test.shape[0]} samples")

# 4. Final model with optimal parameters
print("\nCreating final model with optimized parameters...")

best_params = {
    'learning_rate': 0.1,
    'max_depth': 5,
    'n_estimators': 300
}

final_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(k_neighbors=3, random_state=42)),
    ('classifier', GradientBoostingClassifier(**best_params, random_state=42))
])

# 5. Training and evaluation
print("Training final model...")
final_pipeline.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(final_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nCross-Validation:")
print(f"  Accuracies: {cv_scores}")
print(f"  Mean: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

# Final prediction
y_pred = final_pipeline.predict(X_test)
y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

print(f"\nFinal test accuracy: {accuracy:.3f}")

# 6. Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 7. Feature importance analysis
print("\nFeature Importance Analysis:")
if hasattr(final_pipeline.named_steps['classifier'], 'feature_importances_'):
    importances = final_pipeline.named_steps['classifier'].feature_importances_
    
    # Top 10 important features
    top_indices = np.argsort(importances)[-10:]
    
    print("Top 10 Important Features:")
    for i, idx in enumerate(reversed(top_indices), 1):
        feature_name = feature_cols[idx]
        importance_val = importances[idx]
        print(f"  {i:2d}. {feature_name:30s} : {importance_val:.4f}")

# 8. Generate visual report
fig = plt.figure(figsize=(15, 10))

# 1. Confusion Matrix
ax1 = plt.subplot(2, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
           xticklabels=le.classes_, yticklabels=le.classes_)
ax1.set_title(f'Confusion Matrix\nAccuracy: {accuracy:.3f}', fontsize=14, fontweight='bold')
ax1.set_xlabel('Predicted', fontsize=12)
ax1.set_ylabel('Actual', fontsize=12)

# 2. ROC Curve
ax2 = plt.subplot(2, 3, 2)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

ax2.plot(fpr, tpr, color='darkorange', lw=2, 
        label=f'ROC curve (AUC = {roc_auc:.3f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.3)

# 3. Feature Importance
ax3 = plt.subplot(2, 3, 3)
if hasattr(final_pipeline.named_steps['classifier'], 'feature_importances_'):
    importances = final_pipeline.named_steps['classifier'].feature_importances_
    top_indices = np.argsort(importances)[-15:]
    
    feature_names_short = []
    for idx in top_indices:
        name = feature_cols[idx]
        if len(name) > 25:
            name = name[:22] + '...'
        feature_names_short.append(name)
    
    bars = ax3.barh(range(len(top_indices)), importances[top_indices], color='#2E86AB')
    ax3.set_yticks(range(len(top_indices)))
    ax3.set_yticklabels(feature_names_short, fontsize=9)
    ax3.set_xlabel('Importance', fontsize=12)
    ax3.set_title('Top 15 Important Features', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
else:
    ax3.text(0.5, 0.5, 'Feature importance not available',
            ha='center', va='center', transform=ax3.transAxes)

# 4. Prediction Confidence
ax4 = plt.subplot(2, 3, 4)
confidence = np.max(final_pipeline.predict_proba(X_test), axis=1)

correct_mask = (y_pred == y_test)
correct_conf = confidence[correct_mask]
incorrect_conf = confidence[~correct_mask]

bins = np.linspace(0.5, 1.0, 15)
ax4.hist(correct_conf, bins=bins, alpha=0.7, color='green',
        label=f'Correct ({len(correct_conf)})', density=True)
ax4.hist(incorrect_conf, bins=bins, alpha=0.7, color='red',
        label=f'Incorrect ({len(incorrect_conf)})', density=True)

ax4.set_xlabel('Prediction Confidence', fontsize=12)
ax4.set_ylabel('Density', fontsize=12)
ax4.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Class Distribution
ax5 = plt.subplot(2, 3, 5)
train_counts = [sum(y_train == 0), sum(y_train == 1)]
test_counts = [sum(y_test == 0), sum(y_test == 1)]

x = np.arange(len(le.classes_))
width = 0.35

bars1 = ax5.bar(x - width/2, train_counts, width, label='Train', color='#4ECDC4', alpha=0.8)
bars2 = ax5.bar(x + width/2, test_counts, width, label='Test', color='#FF6B6B', alpha=0.8)

ax5.set_xlabel('Class', fontsize=12)
ax5.set_ylabel('Count', fontsize=12)
ax5.set_title('Class Distribution', fontsize=14, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(le.classes_)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

# 6. Performance Summary
ax6 = plt.subplot(2, 3, 6)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
class_0_metrics = []
class_1_metrics = []

report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

class_0_metrics.append(report_dict[le.classes_[0]]['precision'])
class_0_metrics.append(report_dict[le.classes_[0]]['recall'])
class_0_metrics.append(report_dict[le.classes_[0]]['f1-score'])

class_1_metrics.append(report_dict[le.classes_[1]]['precision'])
class_1_metrics.append(report_dict[le.classes_[1]]['recall'])
class_1_metrics.append(report_dict[le.classes_[1]]['f1-score'])

x = np.arange(len(metrics))
width = 0.35

bars1 = ax6.bar(x - width/2, [accuracy] + class_0_metrics, width, 
               label=le.classes_[0], color='#2E86AB', alpha=0.8)
bars2 = ax6.bar(x + width/2, [accuracy] + class_1_metrics, width, 
               label=le.classes_[1], color='#A23B72', alpha=0.8)

ax6.set_xlabel('Metric', fontsize=12)
ax6.set_ylabel('Score', fontsize=12)
ax6.set_title('Performance by Class', fontsize=14, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(metrics)
ax6.legend()
ax6.set_ylim(0, 1.0)
ax6.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle(f'fMRI Classification Results: {accuracy:.3f} Accuracy', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Save to session1/output directory
report_path = os.path.join(output_dir, 'final_fmri_classification_report.png')
plt.savefig(report_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"\nVisual report saved to '{report_path}'.")

# 9. Save model and results to session1/output directory
model_path = os.path.join(output_dir, 'fmri_classification_model.pkl')
encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
features_path = os.path.join(output_dir, 'feature_names.pkl')

joblib.dump(final_pipeline, model_path)
joblib.dump(le, encoder_path)
joblib.dump(feature_cols, features_path)

print(f"\nModel saved to '{model_path}'.")
print(f"Label encoder saved to '{encoder_path}'.")
print(f"Feature names saved to '{features_path}'.")

# 10. Save performance summary
print("\n" + "="*70)
print("Final Performance Summary")
print("="*70)

print(f"\nOverall Performance:")
print(f"  Final Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"  Cross-Validation: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
print(f"  ROC AUC: {roc_auc:.3f}")

print(f"\nClass-wise Performance:")
for i, cls in enumerate(le.classes_):
    precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
    recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  {cls}:")
    print(f"    Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

# 11. Generate text report in session1/output directory
summary_path = os.path.join(output_dir, 'classification_summary.txt')
with open(summary_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("fMRI Binary Classification Results\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Overall Performance:\n")
    f.write(f"  Final Accuracy: {accuracy:.3f}\n")
    f.write(f"  Mean CV Accuracy: {cv_scores.mean():.3f}\n")
    f.write(f"  ROC AUC: {roc_auc:.3f}\n\n")
    
    f.write(f"Optimal Model Parameters:\n")
    for param, value in best_params.items():
        f.write(f"  {param}: {value}\n")
    
    f.write(f"\nData Statistics:\n")
    f.write(f"  Total Samples: {len(df_binary)}\n")
    f.write(f"  Training Samples: {X_train.shape[0]}\n")
    f.write(f"  Testing Samples: {X_test.shape[0]}\n")
    
    f.write(f"\nClass Distribution:\n")
    for i, cls in enumerate(le.classes_):
        count = sum(y_encoded == i)
        f.write(f"  {cls}: {count} samples ({count/len(y_encoded)*100:.1f}%)\n")
    
    f.write(f"\nFeature Importance (Top 10):\n")
    if hasattr(final_pipeline.named_steps['classifier'], 'feature_importances_'):
        importances = final_pipeline.named_steps['classifier'].feature_importances_
        top_indices = np.argsort(importances)[-10:]
        for i, idx in enumerate(reversed(top_indices), 1):
            feature_name = feature_cols[idx]
            importance_val = importances[idx]
            f.write(f"  {i:2d}. {feature_name}: {importance_val:.4f}\n")

print(f"\nText report saved to '{summary_path}'.")
print(f"\nAll outputs successfully saved to {output_dir}/")
print(f"Directory contents:")
for file in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file)
    if os.path.isfile(file_path):
        size = os.path.getsize(file_path) / 1024  # Size in KB
        print(f"  - {file} ({size:.1f} KB)")
