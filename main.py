import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure nltk packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
flight_data = pd.read_csv('flight_delays.csv')
space_weather_data = pd.read_csv('space_weather.csv')

# Merge datasets on date or appropriate key
data = pd.merge(flight_data, space_weather_data, on='date')

# Text preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]  # Remove punctuation and numbers
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply text preprocessing to a column (assuming 'remarks' column exists)
if 'remarks' in data.columns:
    data['remarks'] = data['remarks'].apply(preprocess_text)

# Feature selection and scaling
features = ['departure_time', 'arrival_time', 'geomagnetic_storm', 'radio_blackout']
if 'remarks' in data.columns:
    features.append('remarks')  # Include preprocessed text feature if it exists

# Tokenize and vectorize the text data
from sklearn.feature_extraction.text import TfidfVectorizer

if 'remarks' in data.columns:
    vectorizer = TfidfVectorizer()
    text_features = vectorizer.fit_transform(data['remarks']).toarray()
    text_feature_names = vectorizer.get_feature_names_out()
    text_df = pd.DataFrame(text_features, columns=text_feature_names)
    data = pd.concat([data.reset_index(drop=True), text_df.reset_index(drop=True)], axis=1)
    features.extend(text_feature_names)

X = data[features]
y = data['delay']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.drop(columns='remarks', errors='ignore'))

if 'remarks' in data.columns:
    X_scaled = np.hstack((X_scaled, text_features))

# Correlation matrix
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()

# K-means clustering
kmeans = KMeans(n_clusters=2)
data['cluster'] = kmeans.fit_predict(X_scaled)

sns.scatterplot(x='departure_time', y='arrival_time', hue='cluster', data=data)
plt.title('K-means Clustering')
plt.show()

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_scaled, y)

# Predict and plot confusion matrix
y_pred = logreg.predict(X_scaled)
conf_matrix = confusion_matrix(y, y_pred)

sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print(classification_report(y, y_pred))

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
            return "True: There is a significant correlation between space weather events and flight delays."
        else:
            return "False: There is no significant correlation between space weather events and flight delays."

# Evaluate the model
correlation_outcome = CorrelationOutcome(logreg)
accuracy, y_pred = correlation_outcome.evaluate(X_scaled, y)
conclusion = correlation_outcome.conclusive_sentence(accuracy)

# Print conclusive sentence
print(conclusion)
