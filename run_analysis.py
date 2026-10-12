"""
Medical Insurance Cost Prediction - Standalone validation script
Runs all analysis steps and saves actual results to model_results.txt
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=== Medical Insurance Cost Prediction ===\n")

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('Medical Insurance Cost Prediction.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# ── Data Quality Check ────────────────────────────────────────────────────────
print(f"Missing values:\n{df.isnull().sum()}\n")
dups = df.duplicated().sum()
print(f"Duplicate rows: {dups}")
if dups > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"After removing duplicates: {df.shape}\n")

print(f"Age: {df['age'].min()}-{df['age'].max()}")
print(f"BMI: {df['bmi'].min():.2f}-{df['bmi'].max():.2f}")
print(f"Charges: {df['charges'].min():.2f}-{df['charges'].max():.2f}\n")

# ── EDA Charts ────────────────────────────────────────────────────────────────

# Chart 1: Distribution of charges
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['charges'], bins=40, color='steelblue', edgecolor='white')
axes[0].set_title('Distribution of Insurance Charges', fontsize=14)
axes[0].set_xlabel('Charges ($)')
axes[0].set_ylabel('Frequency')
axes[1].hist(np.log1p(df['charges']), bins=40, color='darkorange', edgecolor='white')
axes[1].set_title('Log-Transformed Distribution of Charges', fontsize=14)
axes[1].set_xlabel('Log(Charges)')
axes[1].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('chart_charges_distribution.png', dpi=100, bbox_inches='tight')
plt.close()
print("Chart 1 saved: chart_charges_distribution.png")

# Chart 2: Charges by smoker (use tick_labels for matplotlib>=3.9)
fig, ax = plt.subplots(figsize=(8, 5))
smoker_groups = [df[df['smoker']=='yes']['charges'].values,
                 df[df['smoker']=='no']['charges'].values]
bp = ax.boxplot(smoker_groups, patch_artist=True,
                boxprops=dict(facecolor='tomato', color='black'),
                medianprops=dict(color='black', linewidth=2))
ax.set_xticks([1, 2])
ax.set_xticklabels(['Smoker', 'Non-Smoker'])
ax.set_title('Insurance Charges: Smoker vs Non-Smoker', fontsize=14)
ax.set_ylabel('Charges ($)')
plt.tight_layout()
plt.savefig('chart_smoker_charges.png', dpi=100, bbox_inches='tight')
plt.close()
print("Chart 2 saved: chart_smoker_charges.png")

# Chart 3: Age vs Charges
colors = df['smoker'].map({'yes': 'tomato', 'no': 'steelblue'})
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(df['age'], df['charges'], c=colors, alpha=0.5, edgecolors='none', s=30)
ax.set_title('Age vs Insurance Charges (Red=Smoker, Blue=Non-Smoker)', fontsize=13)
ax.set_xlabel('Age')
ax.set_ylabel('Charges ($)')
plt.tight_layout()
plt.savefig('chart_age_vs_charges.png', dpi=100, bbox_inches='tight')
plt.close()
print("Chart 3 saved: chart_age_vs_charges.png")

# Chart 4: BMI vs Charges
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(df['bmi'], df['charges'], c=colors, alpha=0.5, edgecolors='none', s=30)
ax.set_title('BMI vs Insurance Charges (Red=Smoker, Blue=Non-Smoker)', fontsize=13)
ax.set_xlabel('BMI')
ax.set_ylabel('Charges ($)')
plt.tight_layout()
plt.savefig('chart_bmi_vs_charges.png', dpi=100, bbox_inches='tight')
plt.close()
print("Chart 4 saved: chart_bmi_vs_charges.png")

# Chart 5: Average charges by region
region_avg = df.groupby('region')['charges'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(region_avg.index, region_avg.values,
              color=['#4e79a7','#f28e2b','#e15759','#76b7b2'], edgecolor='white')
ax.set_title('Average Insurance Charges by Region', fontsize=14)
ax.set_xlabel('Region')
ax.set_ylabel('Average Charges ($)')
for bar, val in zip(bars, region_avg.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'${val:,.0f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('chart_region_charges.png', dpi=100, bbox_inches='tight')
plt.close()
print("Chart 5 saved: chart_region_charges.png")

# Chart 6: Correlation heatmap
df_corr = df.copy()
df_corr['smoker_enc'] = (df_corr['smoker'] == 'yes').astype(int)
corr_cols = ['age', 'bmi', 'children', 'smoker_enc', 'charges']
corr_matrix = df_corr[corr_cols].corr()
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
labels = ['Age', 'BMI', 'Children', 'Smoker', 'Charges']
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, f'{corr_matrix.values[i, j]:.2f}',
                ha='center', va='center', fontsize=10,
                color='white' if abs(corr_matrix.values[i, j]) > 0.5 else 'black')
ax.set_title('Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('chart_correlation_heatmap.png', dpi=100, bbox_inches='tight')
plt.close()
print("Chart 6 saved: chart_correlation_heatmap.png")
print(f"\nCorrelation with Charges:\n{corr_matrix['charges'].sort_values(ascending=False)}\n")

# Chart 7: Charges by number of children
children_avg = df.groupby('children')['charges'].mean()
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(children_avg.index.astype(str), children_avg.values,
       color='mediumseagreen', edgecolor='white')
ax.set_title('Average Insurance Charges by Number of Children', fontsize=14)
ax.set_xlabel('Number of Children')
ax.set_ylabel('Average Charges ($)')
plt.tight_layout()
plt.savefig('chart_children_charges.png', dpi=100, bbox_inches='tight')
plt.close()
print("Chart 7 saved: chart_children_charges.png")

# ── Preprocessing ─────────────────────────────────────────────────────────────
df_model = df.copy()
le = LabelEncoder()
df_model['sex']    = le.fit_transform(df_model['sex'])
df_model['smoker'] = le.fit_transform(df_model['smoker'])
df_model = pd.get_dummies(df_model, columns=['region'], drop_first=True)

X = df_model.drop('charges', axis=1)
y = df_model['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Models ────────────────────────────────────────────────────────────────────
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
print("Linear Regression trained.")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest trained.")

# ── Evaluation ────────────────────────────────────────────────────────────────
lr_pred = lr_model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test)

def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  RMSE : ${rmse:,.2f}")
    print(f"  R²   : {r2:.4f}")
    return mae, rmse, r2

lr_mae, lr_rmse, lr_r2 = evaluate("Linear Regression", y_test, lr_pred)
rf_mae, rf_rmse, rf_r2 = evaluate("Random Forest Regressor", y_test, rf_pred)

# Chart 8: Model comparison
metrics = ['MAE', 'RMSE', 'R² Score']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (metric, lr_val, rf_val) in enumerate(zip(metrics,
        [lr_mae, lr_rmse, lr_r2], [rf_mae, rf_rmse, rf_r2])):
    axes[i].bar(['Linear\nRegression', 'Random\nForest'], [lr_val, rf_val],
                color=['#4e79a7', '#f28e2b'], edgecolor='white', width=0.5)
    axes[i].set_title(metric, fontsize=13)
    axes[i].set_ylabel(metric)
    for j, val in enumerate([lr_val, rf_val]):
        label = f'{val:.4f}' if metric == 'R² Score' else f'${val:,.0f}'
        axes[i].text(j, val * 0.97, label, ha='center', va='top',
                     fontsize=10, color='white', fontweight='bold')
plt.suptitle('Model Performance Comparison', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('chart_model_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nChart 8 saved: chart_model_comparison.png")

# Chart 9: Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, pred, name, color in zip(axes,
        [lr_pred, rf_pred], ['Linear Regression', 'Random Forest'],
        ['steelblue', 'darkorange']):
    ax.scatter(y_test, pred, alpha=0.4, color=color, edgecolors='none', s=25)
    min_val = min(float(y_test.min()), float(pred.min()))
    max_val = max(float(y_test.max()), float(pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='Perfect Fit')
    ax.set_title(f'{name}: Actual vs Predicted', fontsize=13)
    ax.set_xlabel('Actual Charges ($)')
    ax.set_ylabel('Predicted Charges ($)')
    ax.legend()
plt.tight_layout()
plt.savefig('chart_actual_vs_predicted.png', dpi=100, bbox_inches='tight')
plt.close()
print("Chart 9 saved: chart_actual_vs_predicted.png")

# Chart 10: Feature Importance
feature_names = X.columns.tolist()
importances   = rf_model.feature_importances_
sorted_idx    = np.argsort(importances)[::-1]
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(range(len(feature_names)), importances[sorted_idx],
       color='mediumpurple', edgecolor='white')
ax.set_xticks(range(len(feature_names)))
ax.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=30, ha='right')
ax.set_title('Feature Importances – Random Forest', fontsize=14)
ax.set_ylabel('Importance Score')
plt.tight_layout()
plt.savefig('chart_feature_importance.png', dpi=100, bbox_inches='tight')
plt.close()
print("Chart 10 saved: chart_feature_importance.png")

# ── Business Insights ─────────────────────────────────────────────────────────
smoker_mean    = df[df['smoker']=='yes']['charges'].mean()
nonsmoker_mean = df[df['smoker']=='no']['charges'].mean()
smoker_ratio   = smoker_mean / nonsmoker_mean
corr_age    = df_corr['age'].corr(df_corr['charges'])
corr_bmi    = df_corr['bmi'].corr(df_corr['charges'])
corr_smoker = df_corr['smoker_enc'].corr(df_corr['charges'])

print(f"\n{'='*55}")
print(f"BUSINESS INSIGHTS")
print(f"{'='*55}")
print(f"Smoker mean charges   : ${smoker_mean:,.2f}")
print(f"NonSmoker mean charges: ${nonsmoker_mean:,.2f}")
print(f"Smokers pay ~{smoker_ratio:.2f}x more")
print(f"Corr(smoker, charges) : {corr_smoker:.4f}")
print(f"Corr(age, charges)    : {corr_age:.4f}")
print(f"Corr(bmi, charges)    : {corr_bmi:.4f}")
print(f"Region avg charges:\n{df.groupby('region')['charges'].mean().sort_values(ascending=False)}")

# ── Save results ──────────────────────────────────────────────────────────────
with open('model_results.txt', 'w') as f:
    f.write(f"Dataset rows after cleaning: {len(df)}\n")
    f.write(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}\n\n")
    f.write(f"Linear Regression:\n")
    f.write(f"  MAE  = {lr_mae:.4f}\n")
    f.write(f"  RMSE = {lr_rmse:.4f}\n")
    f.write(f"  R2   = {lr_r2:.4f}\n\n")
    f.write(f"Random Forest Regressor:\n")
    f.write(f"  MAE  = {rf_mae:.4f}\n")
    f.write(f"  RMSE = {rf_rmse:.4f}\n")
    f.write(f"  R2   = {rf_r2:.4f}\n\n")
    f.write(f"Smoker mean charges   = {smoker_mean:.4f}\n")
    f.write(f"NonSmoker mean charges= {nonsmoker_mean:.4f}\n")
    f.write(f"Smoker ratio          = {smoker_ratio:.4f}\n")
    f.write(f"Corr(smoker,charges)  = {corr_smoker:.4f}\n")
    f.write(f"Corr(age,charges)     = {corr_age:.4f}\n")
    f.write(f"Corr(bmi,charges)     = {corr_bmi:.4f}\n")
    region_stats = df.groupby('region')['charges'].mean().sort_values(ascending=False)
    for region, val in region_stats.items():
        f.write(f"Region {region}: {val:.4f}\n")
    fi_lines = []
    for i in sorted_idx:
        fi_lines.append(f"{feature_names[i]}: {importances[i]:.4f}")
    f.write("\nFeature Importances (Random Forest):\n")
    f.write("\n".join(fi_lines) + "\n")

print("\nmodel_results.txt saved.")
print("\n=== ALL STEPS COMPLETED SUCCESSFULLY ===")
