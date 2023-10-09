import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

data = pd.read_csv('data.csv')

# Dropping the unnecessary column 'id'
data.drop('id', axis=1, inplace=True)

# Encoding the target variable 'diagnosis'
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Handling missing values
data = data.dropna()

# Scaling the features
scaler = StandardScaler()
X = scaler.fit_transform(data.drop('diagnosis', axis=1))
y = data['diagnosis']

# Plotting the distribution of the target variable
sns.countplot(data['diagnosis'])

# Plotting the correlation matrix of the features
sns.heatmap(data.corr(), annot=True)

# Creating polynomial features
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Creating interaction terms
interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_interaction = interaction.fit_transform(X)

# Combining the original features, polynomial features, and interaction terms
X_combined = np.concatenate((X, X_poly, X_interaction), axis=1)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Defining the list of models to evaluate
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Evaluating each model on the training set using 10-fold cross-validation
results = []
for name, model in models.items():
    cv_results = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    results.append((name, cv_results.mean(), cv_results.std()))

# Displaying the results
for name, mean, std in results:
    print(f'{name}: {mean} ({std})')

# Tuning the hyperparameters of the best model (Random Forest)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier()
grid_search = GridSearchCV(rf, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f'Best parameters: {best_params}')

# Tuning the hyperparameters of the best model (Random Forest)
rf = RandomForestClassifier(**best_params)
rf.fit(X_train, y_train)

# Evaluating the best model on the test set
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print(f'Test accuracy: {accuracy}')
print(f'Test precision: {precision}')
print(f'Test recall: {recall}')
print(f'Test F1 score: {f1}')
print(f'Test ROC AUC score: {roc_auc}')
print(f'Test confusion matrix:\n{confusion}')

# Making predictions on new data
new_data = pd.read_csv('new_data.csv')
new_data.drop('id', axis=1, inplace=True)
new_data = new_data.dropna()
X_new = scaler.transform(new_data)

X_new_poly = poly_features.transform(X_new)
X_new_interaction = interaction.transform(X_new)
X_new_combined = np.concatenate((X_new, X_new_poly, X_new_interaction), axis=1)

y_new = rf.predict(X_new_combined)
print(f'Predictions for new data: {y_new}')


