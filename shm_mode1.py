import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectFromModel
import pickle
import warnings
warnings.filterwarnings('ignore')

# 1. Data Preparation for crack_detection_dataset.csv
print("Step 1: Data Preparation")
# Load the data
df = pd.read_csv('paste.txt')
print(f"Dataset shape: {df.shape}")
print("\nSample data:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check data types and basic statistics
print("\nData types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())

# Handle missing values in crack_size_mm and crack_depth_ratio
df['crack_size_mm'] = df['crack_size_mm'].fillna(0)
df['crack_depth_ratio'] = df['crack_depth_ratio'].fillna(0)

# Extract sample_id information
df['sample_type'] = df['sample_id'].apply(lambda x: x.split('_')[0])

# 2. Exploratory Data Analysis
print("\nStep 2: Exploratory Data Analysis")

# Class distribution
print("\nClass distribution (has_crack):")
print(df['has_crack'].value_counts())
print(f"Percentage of cracked samples: {df['has_crack'].mean() * 100:.2f}%")

# Create figures directory if it doesn't exist
import os
if not os.path.exists('figures'):
    os.makedirs('figures')

# Visualize distributions of key numerical features
numerical_features = ['mean_amplitude', 'std_amplitude', 'max_amplitude', 'min_amplitude', 
                     'peak_frequency', 'energy', 'zero_crossings', 'main_freq_energy',
                     'harmonic_ratio', 'envelope_mean', 'envelope_std', 'kurtosis', 
                     'skewness', 'thickness_mm']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features[:9], 1):  # First 9 features
    plt.subplot(3, 3, i)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig('figures/numerical_distributions_1.png')
plt.close()

plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features[9:], 1):  # Remaining features
    plt.subplot(2, 3, i)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig('figures/numerical_distributions_2.png')
plt.close()

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('figures/correlation_matrix.png')
plt.close()

# Material distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='material', data=df)
plt.title('Distribution of Materials')
plt.tight_layout()
plt.savefig('figures/material_distribution.png')
plt.close()

# 3. Feature Engineering
print("\nStep 3: Feature Engineering")

# Create some new features
df['amplitude_ratio'] = df['max_amplitude'] / (df['min_amplitude'] + 1e-10)  # Avoid division by zero
df['energy_per_crossing'] = df['energy'] / (df['zero_crossings'] + 1e-10)
df['freq_energy_ratio'] = df['main_freq_energy'] / (df['energy'] + 1e-10)
df['envelope_ratio'] = df['envelope_mean'] / (df['envelope_std'] + 1e-10)

# Analyze correlation of new features with target
new_features = ['amplitude_ratio', 'energy_per_crossing', 'freq_energy_ratio', 'envelope_ratio']
all_features = numerical_features + new_features

# Calculate feature correlations with target
feature_importance = []
for feature in all_features:
    correlation = df[feature].corr(df['has_crack']) if df['has_crack'].nunique() > 1 else 0
    feature_importance.append((feature, abs(correlation)))

feature_importance.sort(key=lambda x: x[1], reverse=True)
print("\nFeature importance based on correlation with target:")
for feature, importance in feature_importance:
    print(f"{feature}: {importance:.4f}")

# 4. Model Development
print("\nStep 4: Model Development")

# Define features and target
X = df.drop(['has_crack', 'sample_id', 'crack_size_mm', 'crack_depth_ratio', 'sample_type'], axis=1)
y = df['has_crack']

# Define categorical and numerical features
categorical_features = ['material']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if y.nunique() > 1 else None)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# 5. Model Training and Evaluation
print("\nStep 5: Model Training and Evaluation")

# Define models to try
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Only calculate these if we have both classes in the test set
    if y_test.nunique() > 1 and np.unique(y_pred).size > 1:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred) if y_test.nunique() > 1 else "N/A"
    else:
        precision = recall = f1 = "N/A"
        roc_auc = "N/A"
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # For Random Forest, get feature importances
    if name == 'Random Forest':
        importances = pipeline.named_steps['classifier'].feature_importances_
        feature_names = (numerical_features + 
                        list(pipeline.named_steps['preprocessor']
                            .named_transformers_['cat']
                            .get_feature_names_out(categorical_features)))
        
        # Create a DataFrame for feature importances
        forest_importances = pd.Series(importances, index=feature_names)
        forest_importances = forest_importances.sort_values(ascending=False)
        
        print("\nRandom Forest Feature Importances:")
        print(forest_importances)
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        forest_importances[:15].plot.bar()
        plt.title('Random Forest Feature Importances (Top 15)')
        plt.tight_layout()
        plt.savefig('figures/feature_importances.png')
        plt.close()

# 6. Hyperparameter Tuning for the best model
print("\nStep 6: Hyperparameter Tuning")
 
# Define parameter grid for Random Forest
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Create pipeline with Random Forest
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Grid search with cross-validation
grid_search = GridSearchCV(
    rf_pipeline, param_grid, cv=5, n_jobs=-1, 
    scoring='accuracy', verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate best model on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
if y_test.nunique() > 1 and np.unique(y_pred).size > 1:
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred) if y_test.nunique() > 1 else "N/A"
else:
    precision = recall = f1 = "N/A"
    roc_auc = "N/A"

print("\nTuned Random Forest Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Feature Selection with the best model
print("\nStep 7: Feature Selection")

# Use selectfrommodel to select important features
selector = SelectFromModel(best_model.named_steps['classifier'], threshold='median')
selector.fit(preprocessor.fit_transform(X_train), y_train)

# Get selected feature indices
selected_indices = selector.get_support()

# Get feature names after preprocessing
feature_names = (numerical_features + 
                list(preprocessor.named_transformers_['cat']
                    .get_feature_names_out(categorical_features)))

# Get selected feature names
selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_indices[i]]
print(f"\nSelected Features ({len(selected_features)}):")
for feature in selected_features:
    print(feature)

# 8. Model Deployment Preparation
print("\nStep 8: Model Deployment Preparation")

# Save the best model
with open('best_crack_detection_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("\nBest model saved as 'best_crack_detection_model.pkl'")

# Create a simple prediction function for the API
def predict_crack(data):
    """
    Make predictions using the saved model.
    
    Parameters:
    data (dict): Dictionary containing feature values
    
    Returns:
    dict: Prediction result and probability
    """
    # Load the model
    with open('best_crack_detection_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        'has_crack': bool(prediction),
        'crack_probability': float(probability)
    }

print("\nSample usage of prediction function:")
sample_input = {
    'mean_amplitude': 0.25,
    'std_amplitude': 0.32,
    'max_amplitude': 1.0,
    'min_amplitude': 0.001,
    'peak_frequency': 10000.0,
    'energy': 40.0,
    'zero_crossings': 200,
    'main_freq_energy': 5.0,
    'harmonic_ratio': 0.0,
    'envelope_mean': 0.25,
    'envelope_std': 0.19,
    'kurtosis': 3.0,
    'skewness': 0.0,
    'material': 'Steel',
    'thickness_mm': 20.0
}

# Create a simple demo of the prediction function
print(f"Input data: {sample_input}")
print("This is simulated - actual prediction would require loading the saved model.")
print("Prediction result would look like: {'has_crack': False, 'crack_probability': 0.15}")

print("\nAnalysis complete!")

# 9. Create a simple CLI for predictions
def create_cli():
    """
    Create a simple command-line interface for making predictions.
    """
    print("\n==== Crack Detection CLI ====")
    print("Enter the following parameters:")
    
    # Collect required inputs
    data = {}
    data['mean_amplitude'] = float(input("Mean Amplitude (0-1): "))
    data['std_amplitude'] = float(input("Std Amplitude (0-1): "))
    data['max_amplitude'] = float(input("Max Amplitude (typically 1.0): "))
    data['min_amplitude'] = float(input("Min Amplitude (0-1): "))
    data['peak_frequency'] = float(input("Peak Frequency (Hz): "))
    data['energy'] = float(input("Energy: "))
    data['zero_crossings'] = int(input("Zero Crossings: "))
    data['main_freq_energy'] = float(input("Main Frequency Energy: "))
    data['harmonic_ratio'] = float(input("Harmonic Ratio: "))
    data['envelope_mean'] = float(input("Envelope Mean: "))
    data['envelope_std'] = float(input("Envelope Std: "))
    data['kurtosis'] = float(input("Kurtosis: "))
    data['skewness'] = float(input("Skewness: "))
    data['material'] = input("Material (Aluminum/Steel/Titanium/Iron/Copper): ")
    data['thickness_mm'] = float(input("Thickness (mm): "))
    
    # Make prediction
    result = predict_crack(data)
    
    # Display result
    print("\n==== Prediction Result ====")
    print(f"Has Crack: {'Yes' if result['has_crack'] else 'No'}")
    print(f"Crack Probability: {result['crack_probability']:.2%}")
    
if __name__ == "__main__":
    create_cli()