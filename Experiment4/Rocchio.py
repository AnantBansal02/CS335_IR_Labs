import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load and preprocess the dataset
from os import listdir
from os.path import isfile, join
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

X = newsgroups.data
Y = newsgroups.target
target_names = newsgroups.target_names

# Step 2: Compute TF-IDF representation of the documents
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Step 3: Implement the Rocchio algorithm
X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf, Y, test_size=0.3, random_state=42)

rocchio_model = NearestCentroid()
rocchio_model.fit(X_train, Y_train)

# Step 4: Evaluate the model
Y_pred = rocchio_model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
classification_rep = classification_report(Y_test, Y_pred, target_names=target_names)
confusion_mat = confusion_matrix(Y_test, Y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", confusion_mat)
