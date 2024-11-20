import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

# Load the datasets
flight_data_path = 'flight_data_file_path'
geomagnetic_storm_data_path = 'geomagnetic_storm_data_file_path'

flight_data = pd.read_csv(flight_data_path)
geomagnetic_storm_data = pd.read_csv(geomagnetic_storm_data_path)

# Strip whitespace from column names
geomagnetic_storm_data.columns = geomagnetic_storm_data.columns.str.strip()

# Preprocess flight data
flight_data.rename(columns={'Date': 'date', 'Time': 'time', 'Total Delay': 'delay'}, inplace=True)

# Drop unnecessary columns
columns_to_drop = ['Terminal', 'Call Sign', 'Marketing Airline', 'General Aircraft Desc',
                   'Max Takeoff Wt (Lbs)', 'Max Landing Wt (Lbs)', 'Intl / Dom',
                   'Total Seats', 'Total Taxi Time', 'Direction', 'PA Airport', 'Non-PA Airport']
flight_data.drop(columns=columns_to_drop, inplace=True)

# Ensure the date columns are in the same format
flight_data['date'] = pd.to_datetime(flight_data['date'], errors='coerce')
geomagnetic_storm_data['date'] = pd.to_datetime(geomagnetic_storm_data['Date'], errors='coerce')

# Drop rows with invalid dates
flight_data.dropna(subset=['date'], inplace=True)
geomagnetic_storm_data.dropna(subset=['date'], inplace=True)

# Merge datasets on date
data = pd.merge(flight_data, geomagnetic_storm_data, on='date')

# Handle missing data
data = data.dropna()

# Convert delay to numeric
data['delay'] = pd.to_numeric(data['delay'], errors='coerce')

# Drop rows with NaN delays
data = data.dropna(subset=['delay'])

# Convert delay into a binary format: delayed (1) or not delayed (0)
data['delay'] = data['delay'].apply(lambda x: 1 if x > 0 else 0)

# Convert time to numerical format (minutes since midnight)
data['time'] = pd.to_datetime(data['time'], errors='coerce').dt.hour * 60 + pd.to_datetime(data['time'], errors='coerce').dt.minute

# Drop rows with NaN times
data = data.dropna(subset=['time'])

# Ensure all columns are numeric and handle units
columns_to_convert = ['Dst', 'Ap', 'Kp max', 'Speed', 'IMF Bt', 'IMF Bz']
for column in columns_to_convert:
    if column in data.columns:
        # Remove units from strings, if any
        data[column] = data[column].astype(str).str.replace(r'[^0-9.\-]+', '', regex=True)
        data[column] = pd.to_numeric(data[column], errors='coerce')
    else:
        print(f"Column {column} not found in data.")

# Ensure no NaN values are present in the features
data = data.dropna(subset=['time', 'Dst', 'Ap', 'Speed', 'IMF Bt', 'IMF Bz'])

# Feature selection and scaling
features = ['time', 'Dst', 'Ap', 'Speed', 'IMF Bt', 'IMF Bz']
X = data[features]
y = data['delay']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Correlation matrix
corr_matrix = data[features + ['delay']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

# Cross-validation score
cv_scores = cross_val_score(logreg, X_train_scaled, y_train, cv=5)
cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

# Predict and plot confusion matrix
y_pred = logreg.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification report
class_report = classification_report(y_test, y_pred, output_dict=True)
class_report_text = classification_report(y_test, y_pred)
print(class_report_text)

class CorrelationOutcome:
    def __init__(self, logistic_model, threshold=0.5):
        self.model = logistic_model
        self.threshold = threshold

    def evaluate(self, X, y):
        y_pred_prob = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_prob >= self.threshold).astype(int)
        accuracy = (y_pred == y).mean()
        return accuracy, y_pred

    def conclusive_sentence(self, accuracy):
        if accuracy > 0.7:  # Example threshold for determining a significant correlation
            return "True: There is a significant correlation between geomagnetic storm intensity (Dst index) and flight delays."
        else:
            return "False: There is no significant correlation between geomagnetic storm intensity (Dst index) and flight delays."

# Evaluate the model
correlation_outcome = CorrelationOutcome(logreg)
accuracy, y_pred_eval = correlation_outcome.evaluate(X_test_scaled, y_test)
conclusion = correlation_outcome.conclusive_sentence(accuracy)

# Print conclusive sentence
print(conclusion)

# Additional visualization of model performance
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
scores = [
    accuracy,
    class_report['1']['precision'] if '1' in class_report else 0,
    class_report['1']['recall'] if '1' in class_report else 0,
    class_report['1']['f1-score'] if '1' in class_report else 0
]
plt.figure(figsize=(10, 6))
sns.barplot(x=metrics, y=scores)
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.show()

# Summary of findings
summary_of_findings = f"""
Summary of Findings:
Cross-validation accuracy: {cv_mean:.2f} ± {cv_std:.2f}
Confusion Matrix:
{conf_matrix}
{classification_report(y_test, y_pred)}
"""
print(summary_of_findings)

# Logistic Regression Decision Boundary Plot (for two features)
def plot_decision_boundary(X, y, model, features):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='Spectral')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

# Select two features for decision boundary plot
selected_features = ['Dst', 'Speed']
X_selected = data[selected_features].values
y_selected = data['delay'].values

# Split the data for selected features
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y_selected, test_size=0.2, random_state=42)

# Scale the selected features
X_train_sel_scaled = scaler.fit_transform(X_train_sel)
X_test_sel_scaled = scaler.transform(X_test_sel)

# Train the model with selected features
logreg_sel = LogisticRegression(max_iter=1000)
logreg_sel.fit(X_train_sel_scaled, y_train_sel)

# Plot the decision boundary
plot_decision_boundary(X_test_sel_scaled, y_test_sel, logreg_sel, selected_features)
