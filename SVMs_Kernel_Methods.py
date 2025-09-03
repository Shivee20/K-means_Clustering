
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
import shap
import time
from scipy import stats

# Read Excel
df = pd.read_excel("HIGGS_samples")

# Save as CSV
df.to_csv("HIGGS_samples.csv", index=False)

# Data loading
df = pd.read_csv("D:\Downloads\HIGGS_sample.csv")
print(df.shape)
print(df.columns.to_list())
print(df.head())

# Checking for missing values
print(df.isna().sum())
print(df.isnull().sum())

# Plotting feature distributions
num_cols = 3
num_rows = (len(df.columns) + num_cols - 1) // num_cols
plt.figure(figsize=(15, num_rows * 5))
for i, col in enumerate(df.columns):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Standard scaling
scaler = StandardScaler()
num_columns = df.select_dtypes(include=[np.number]).columns[1:]
print(num_columns)
df[num_columns] = scaler.fit_transform(df[num_columns])

# Plotting scaled distributions
num_cols = 3
num_rows = (len(df.columns) + num_cols - 1) // num_cols
plt.figure(figsize=(15, num_rows * 5))
for i, col in enumerate(df.columns):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Outlier detection
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
threshold = 4
outlier_counts = (z_scores > threshold).sum(axis=0)
outlier_summary = pd.DataFrame(outlier_counts, columns=['Number of Outliers'])
outlier_summary.index = df.select_dtypes(include=[np.number]).columns
print(outlier_summary.value_counts())
print(outlier_summary)

# Plotting outliers
num_cols = 3
num_rows = (len(df.columns) + num_cols - 1) // num_cols
plt.figure(figsize=(15, num_rows * 5))
for i, col in enumerate(df.select_dtypes(include=[np.number]).columns):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.hist(df[col], bins=30, edgecolor='black', alpha=0.7, label='Data Distribution', color='blue')
    outlier_mask = z_scores[:, i] > threshold
    outliers = df[col][outlier_mask]
    plt.hist(outliers, bins=30, edgecolor='black', alpha=0.7, label='Outliers', color='red')
    plt.title(f'Feature: {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.legend()
plt.tight_layout()
plt.show()

# Remove outliers
non_outlier_mask = np.all(z_scores <= threshold, axis=1)
df_cleaned = df[non_outlier_mask].reset_index(drop=True)
print(df.shape)
print(df_cleaned.shape)

# Correlation heatmap
df_features = df_cleaned.iloc[:, 1:]
correlation_matrix = df_features.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# Feature engineering: remove highly correlated features, add their product
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if correlation_matrix.iloc[i, j] >= 0.6:
            feature1 = correlation_matrix.columns[i]
            feature2 = correlation_matrix.columns[j]
            high_corr_pairs.append((feature1, feature2))
for feature1, feature2 in high_corr_pairs:
    df_cleaned[f'{feature1}_x_{feature2}'] = df_cleaned[feature1] * df_cleaned[feature2]
    df_cleaned = df_cleaned.drop(columns=[feature2])
    df_cleaned = df_cleaned.drop(columns=[feature1])
print(df_cleaned.columns.tolist())
print(df_cleaned.shape)

# Feature selection
target_column = '1.000000000000000000e+00'
X = df_cleaned.drop(columns=[target_column])
y = df_cleaned[target_column]
selector = SelectKBest(score_func=f_classif, k=10)
X_reduced = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
df_reduced = pd.DataFrame(X_reduced, columns=selected_features)
df_reduced[target_column] = y.values
print(df_reduced.shape)
print(df_reduced.head())

# Linear kernel SVM (SGD)
X = df_reduced.drop(columns=[target_column])
y = df_reduced[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
svm = SGDClassifier(loss='hinge')
cv_scores = cross_val_score(svm, X, y, cv=5, scoring='accuracy')
print(f"Cross-validated Accuracy : {np.mean(cv_scores):.4f}")
start_time_train = time.time()
svm.fit(X_train, y_train)
end_time_train = time.time()
start_time_pred = time.time()
y_pred = svm.predict(X_test)
end_time_pred = time.time()
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
y_pred_proba = svm.decision_function(X_test)
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc:.4f}")
train_time = end_time_train - start_time_train
pred_time = end_time_pred - start_time_pred
print(f"Training Time (seconds): {train_time:.4f}")
print(f"Prediction Time (seconds): {pred_time:.4f}")

# SHAP analysis for linear kernel
explainer = shap.LinearExplainer(svm, X_train)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", plot_size=(8, 4))
shap.summary_plot(shap_values, X_test, plot_size=(8, 4))

# Hyperparameter tuning for linear kernel (C)
n_train_samples = X.shape[0]
param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0]}
def alpha_based_on_c(C): return 1.0 / (C * n_train_samples)
sgd_svm = SGDClassifier(loss='hinge', random_state=42)
params = [{'alpha': alpha_based_on_c(c), 'C': c} for c in param_grid['C']]
best_score = 0
best_params = {}
mean_scores = []
for param in params:
    sgd_svm.set_params(alpha=param['alpha'])
    cv_scores = cross_val_score(sgd_svm, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    mean_score = np.mean(cv_scores)
    print(f"C: {param['C']}, Alpha: {param['alpha']}, Mean Accuracy: {mean_score:.4f}")
    mean_scores.append(mean_score)
    if mean_score > best_score:
        best_score = mean_score
        best_params = param
print(f"Best C: {best_params['C']}, Best Alpha: {best_params['alpha']}, Best Cross-validated Accuracy: {best_score:.4f}")
sgd_svm.set_params(alpha=best_params['alpha'])
sgd_svm.fit(X_train, y_train)
y_pred_best = sgd_svm.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)
y_pred_proba_best = sgd_svm.decision_function(X_test)
auc_best = roc_auc_score(y_test, y_pred_proba_best)
print(f"Best Model - Accuracy: {accuracy_best:.4f}")
print(f"Best Model - Precision: {precision_best:.4f}")
print(f"Best Model - Recall: {recall_best:.4f}")
print(f"Best Model - F1 Score: {f1_best:.4f}")
print(f"Best Model - AUC: {auc_best:.4f}")

# Plot effect of C on accuracy
plt.figure(figsize=(10, 6))
plt.plot(param_grid['C'], mean_scores, marker='o', linestyle='-')
plt.xscale('log')
plt.xlabel('Value of C (log scale)')
plt.ylabel('Mean Cross-validated Accuracy')
plt.title('Effect of C on Cross-validated Accuracy')
plt.grid()
plt.xticks(param_grid['C'])
plt.show()

# Polynomial kernel SVM (SGD)
degrees = [2, 3, 4]
results = {}
for degree in degrees:
    svm_poly_sgd = Pipeline([
        ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),
        ('sgd_svm', SGDClassifier(loss='hinge', n_jobs=-1))
    ])
    start_time_train = time.time()
    svm_poly_sgd.fit(X_train, y_train)
    end_time_train = time.time()
    start_time_pred = time.time()
    y_pred = svm_poly_sgd.predict(X_test)
    end_time_pred = time.time()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, svm_poly_sgd.decision_function(X_test))
    train_time = end_time_train - start_time_train
    pred_time = end_time_pred - start_time_pred
    results[degree] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc,
        'Training Time (seconds)': train_time,
        'Prediction Time (seconds)': pred_time
    }
for degree, metrics in results.items():
    print(f"Polynomial Degree: {degree}")
    print(f" - Accuracy: {metrics['Accuracy']:.4f}")
    print(f" - Precision: {metrics['Precision']:.4f}")
    print(f" - Recall: {metrics['Recall']:.4f}")
    print(f" - F1 Score: {metrics['F1 Score']:.4f}")
    print(f" - AUC: {metrics['AUC']:.4f}")
    print(f" - Training Time (seconds): {metrics['Training Time (seconds)']:.4f}")
    print(f" - Prediction Time (seconds): {metrics['Prediction Time (seconds)']:.4f}")
    print("\n")

# SHAP analysis for polynomial kernel
degrees = [2, 3, 4]
for degree in degrees:
    print(f"\n--- SHAP Analysis for Polynomial Degree {degree} ---\n")
    svm_poly_sgd = Pipeline([
        ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),
        ('sgd_svm', SGDClassifier(loss='hinge', n_jobs=-1))
    ])
    svm_poly_sgd.fit(X_train, y_train)
    explainer = shap.LinearExplainer(svm_poly_sgd.named_steps['sgd_svm'], svm_poly_sgd.named_steps['poly_features'].transform(X_train))
    X_test_poly = svm_poly_sgd.named_steps['poly_features'].transform(X_test)
    shap_values = explainer.shap_values(X_test_poly)
    print(f"SHAP Summary Plot for Polynomial Degree {degree}")
    shap.summary_plot(shap_values, X_test_poly, plot_type="bar", plot_size=(8, 4))
    shap.summary_plot(shap_values, X_test_poly, plot_size=(8, 4))

# Hyperparameter tuning for polynomial kernel
def alpha_based_on_c(C, n_train_samples): return 1.0 / (C * n_train_samples)
degrees = [2, 3, 4]
C_values = [0.01, 0.05, 0.1, 0.5, 1.0]
results = {}
accuracy_scores = {degree: [] for degree in degrees}
n_train_samples = X_train.shape[0]
for degree in degrees:
    pipeline = Pipeline([
        ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),
        ('sgd_svm', SGDClassifier(loss='hinge', n_jobs=-1))
    ])
    param_grid = {
        'sgd_svm__alpha': [alpha_based_on_c(C, n_train_samples) for C in C_values]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    accuracy_scores[degree] = grid_search.cv_results_['mean_test_score']
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, best_model.decision_function(X_test))
    results[degree] = {
        'Best C': 1 / (best_model.named_steps['sgd_svm'].alpha * n_train_samples),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc
    }
for degree, metrics in results.items():
    print(f"Polynomial Degree: {degree}")
    print(f" - Best C: {metrics['Best C']}")
    print(f" - Accuracy: {metrics['Accuracy']:.4f}")
    print(f" - Precision: {metrics['Precision']:.4f}")
    print(f" - Recall: {metrics['Recall']:.4f}")
    print(f" - F1 Score: {metrics['F1 Score']:.4f}")
    print(f" - AUC: {metrics['AUC']:.4f}")
    print("\n")

# Plot accuracy vs C for polynomial degrees
plt.figure(figsize=(10, 6))
for degree in degrees:
    plt.plot(C_values, accuracy_scores[degree], label=f'Degree {degree}', marker='o')
plt.xscale('log')
plt.xlabel('C Value (log scale)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs C Value for Different Polynomial Degrees')
plt.legend()
plt.show()

# RBF kernel SVM (SGD)
gamma = 0.1
C = 1.0
def alpha_based_on_c(C, n_train_samples): return 1.0 / (C * n_train_samples)
n_train_samples = X_train.shape[0]
svm_rbf_sgd = Pipeline([
    ('rbf_features', RBFSampler(gamma=gamma, random_state=42)),
    ('sgd_svm', SGDClassifier(loss='hinge', alpha=alpha_based_on_c(C, n_train_samples)))
])
start_time_train = time.time()
svm_rbf_sgd.fit(X_train, y_train)
end_time_train = time.time()
start_time_pred = time.time()
y_pred = svm_rbf_sgd.predict(X_test)
end_time_pred = time.time()
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, svm_rbf_sgd.decision_function(X_test))
train_time = end_time_train - start_time_train
pred_time = end_time_pred - start_time_pred
print("Approximate RBF Kernel SVM with SGD Results:")
print(f" - alpha: {alpha_based_on_c(C, n_train_samples)}")
print(f" - Accuracy: {accuracy:.4f}")
print(f" - Precision: {precision:.4f}")
print(f" - Recall: {recall:.4f}")
print(f" - F1 Score: {f1:.4f}")
print(f" - AUC: {auc:.4f}")
print(f" - Training Time (seconds): {train_time:.4f}")
print(f" - Prediction Time (seconds): {pred_time:.4f}")

# SHAP analysis for RBF kernel
explainer = shap.LinearExplainer(svm_rbf_sgd.named_steps['sgd_svm'], svm_rbf_sgd.named_steps['rbf_features'].transform(X_train))
X_test_rbf = svm_rbf_sgd.named_steps['rbf_features'].transform(X_test)
shap_values = explainer.shap_values(X_test_rbf)
print("SHAP Summary Plot for Approximate RBF Kernel")
shap.summary_plot(shap_values, X_test_rbf, plot_type="bar", plot_size=(8, 4))
shap.summary_plot(shap_values, X_test_rbf, plot_size=(8, 4))

# Hyperparameter tuning for RBF kernel
class CustomSGDClassifier(SGDClassifier):
    def __init__(self, C=1.0, gamma=0.1, **kwargs):
        alpha = alpha_based_on_c(C, n_train_samples)
        super().__init__(loss="hinge", alpha=alpha, **kwargs)
        self.C = C
        self.gamma = gamma
pipeline = Pipeline([
    ('rbf_features', RBFSampler(random_state=42)),
    ('sgd_svm', CustomSGDClassifier())
])
param_grid = {
    'rbf_features__gamma': [0.15, 0.3, 0.5, 0.7],
    'sgd_svm__C': [1, 3, 7, 10]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.decision_function(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
print("Best Hyperparameters:", grid_search.best_params_)
print("Tuned Approximate RBF Kernel SVM with SGD Results:")
print(f" - alpha: {alpha_based_on_c(best_model.named_steps['sgd_svm'].C, n_train_samples)}")
print(f" - Accuracy: {accuracy:.4f}")
print(f" - Precision: {precision:.4f}")
print(f" - Recall: {recall:.4f}")
print(f" - F1 Score: {f1:.4f}")
print(f" - AUC: {auc:.4f}")

# Plot accuracy for different gamma and C values
mean_test_scores = grid_search.cv_results_['mean_test_score']
C_values = grid_search.param_grid['sgd_svm__C']
gamma_values = grid_search.param_grid['rbf_features__gamma']
scores_matrix = mean_test_scores.reshape(len(gamma_values), len(C_values))
plt.figure(figsize=(10, 6))
for i, gamma in enumerate(gamma_values):
    plt.plot(C_values, scores_matrix[i], marker='o', label=f'Gamma: {gamma}')
plt.xlabel('C Values')
plt.ylabel('Mean Test Score (Accuracy)')
plt.title('Comparative Line Plot of Mean Test Scores for Different C and Gamma Values')
plt.legend(title='Gamma Values')
plt.xscale('log')
plt.grid(True)
plt.show()

# Custom kernel: combine RBF and polynomial features
class CombinedFeatureTransformer:
    def __init__(self, gamma=1.0, poly_degree=2, n_components=10):
        self.rbf_feature = RBFSampler(gamma=gamma, random_state=42, n_components=n_components)
        self.poly_feature = PolynomialFeatures(degree=poly_degree, include_bias=False)
    def fit(self, X, y=None):
        self.rbf_feature.fit(X)
        self.poly_feature.fit(X)
        return self
    def transform(self, X):
        rbf_transformed = self.rbf_feature.transform(X)
        poly_transformed = self.poly_feature.transform(X)
        return np.hstack((rbf_transformed, poly_transformed))
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
transformer = CombinedFeatureTransformer(gamma=0.15, poly_degree=3)
X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)
C = 0.01
alpha = 1.0 / (C * len(X_train))
svm_linear = SGDClassifier(loss='hinge', random_state=42, alpha=alpha)
start_time_train = time.time()
svm_linear.fit(X_train_transformed, y_train)
end_time_train = time.time()
start_time_pred = time.time()
y_pred = svm_linear.predict(X_test_transformed)
end_time_pred = time.time()
y_pred_proba = svm_linear.decision_function(X_test_transformed)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
train_time = end_time_train - start_time_train
pred_time = end_time_pred - start_time_pred
print("Custom Combined Kernel SVM Results:")
print(f" - alpha: {alpha:.4f}")
print(f" - Accuracy: {accuracy:.4f}")
print(f" - Precision: {precision:.4f}")
print(f" - Recall: {recall:.4f}")
print(f" - F1 Score: {f1:.4f}")
print(f" - AUC: {auc:.4f}")
print(f" - Training Time (seconds): {train_time:.4f}")
print(f" - Prediction Time (seconds): {pred_time:.4f}")

# SHAP analysis for custom kernel
explainer = shap.LinearExplainer(svm_linear, X_train_transformed)
shap_values = explainer.shap_values(X_test_transformed)
print("SHAP Summary Plot for Custom Combined Kernel")
shap.summary_plot(shap_values, X_test_transformed, plot_type="bar", plot_size=(8, 4))
shap.summary_plot(shap_values, X_test_transformed, plot_size=(8, 4))

# Manual tuning for custom kernel
values_of_c = [0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0]
accuracy_scores = []
for c in values_of_c:
    transformer = CombinedFeatureTransformer(gamma=0.15, poly_degree=3)
    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)
    alpha = 1.0 / (c * len(X_train))
    svm_linear = SGDClassifier(loss='hinge', random_state=42, alpha=alpha)
    svm_linear.fit(X_train_transformed, y_train)
    y_pred = svm_linear.predict(X_test_transformed)
    y_pred_proba = svm_linear.decision_function(X_test_transformed)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    print(f"C: {c}, Alpha: {alpha}, Accuracy: {accuracy}")

# Plot effect of C on accuracy for custom kernel
plt.figure(figsize=(10, 6))
plt.plot(values_of_c, accuracy_scores, marker='o', linestyle='-')
plt.xscale('log')
plt.xlabel('Value of C (log scale)')
plt.ylabel('Accuracy')
plt.title('Effect of C on Accuracy for Combined Kernel SVM')
plt.grid()
plt.show()