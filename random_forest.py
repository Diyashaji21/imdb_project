import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed data
df = pd.read_csv('IMDB_Dataset.csv')

# Preprocess the text data (assuming it's already done if the data is preprocessed)
def preprocess_text(text):
    # Add your text preprocessing steps here
    return text

df['review'] = df['review'].apply(preprocess_text)

# Convert sentiment labels to numerical
y = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split the data
X = df['review']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vect, y_train)
nb_pred = nb_model.predict(X_test_vect)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

# Random Forest with reduced n_estimators
rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf_model.fit(X_train_vect, y_train)
rf_pred = rf_model.predict(X_test_vect)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# SVM
svm_model = SVC()
svm_model.fit(X_train_vect, y_train)
svm_pred = svm_model.predict(X_test_vect)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))
